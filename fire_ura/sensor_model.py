from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from torch import Tensor, nn


EmbeddingActivation = Literal["none", "tanh"]


@dataclass(frozen=True)
class SensorEncoderConfig:
    """Configuration for the lightweight sensor-side temporal encoder."""

    input_channels: int = 4
    base_channels: int = 32
    hidden_channels: int = 64
    embedding_dim: int = 8
    kernel_size: int = 5
    dropout: float = 0.10
    embedding_activation: EmbeddingActivation = "tanh"

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive.")
        if self.hidden_channels <= 0:
            raise ValueError("hidden_channels must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if self.kernel_size <= 0 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.embedding_activation not in {"none", "tanh"}:
            raise ValueError(
                "embedding_activation must be either 'none' or 'tanh'."
            )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class TemporalConvBlock(nn.Module):
    """Lightweight depthwise-separable temporal convolution block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int,
        dropout: float,
    ) -> None:
        super().__init__()

        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                in_channels,
                kernel_size=kernel_size,
                padding=padding,
                groups=in_channels,
                bias=False,
            ),
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SensorEncoder(nn.Module):
    """
    Encode a multivariate sensor window into a compact feature vector.

    Input shape: ``[batch, input_channels, time]``.
    Output shape: ``[batch, embedding_dim]``.
    """

    def __init__(self, config: SensorEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        padding = config.kernel_size // 2

        self.stem = nn.Sequential(
            nn.Conv1d(
                config.input_channels,
                config.base_channels,
                kernel_size=config.kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(config.base_channels),
            nn.GELU(),
        )

        self.temporal_backbone = nn.Sequential(
            TemporalConvBlock(
                config.base_channels,
                config.base_channels,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
            TemporalConvBlock(
                config.base_channels,
                config.hidden_channels,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
            TemporalConvBlock(
                config.hidden_channels,
                config.hidden_channels,
                kernel_size=config.kernel_size,
                dropout=config.dropout,
            ),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        projection_layers: list[nn.Module] = [
            nn.Flatten(start_dim=1),
            nn.Linear(config.hidden_channels, config.hidden_channels),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_channels, config.embedding_dim),
        ]
        if config.embedding_activation == "tanh":
            projection_layers.append(nn.Tanh())

        self.projection = nn.Sequential(*projection_layers)

    def forward(self, x: Tensor) -> Tensor:
        self._validate_input(x)

        features = self.stem(x)
        features = self.temporal_backbone(features)
        pooled = self.global_pool(features)
        return self.projection(pooled)

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "SensorEncoder expects input with shape "
                f"[batch, channels, time], but received {tuple(x.shape)}."
            )
        if x.shape[1] != self.config.input_channels:
            raise ValueError(
                f"Expected {self.config.input_channels} input channels, "
                f"but received {x.shape[1]}."
            )
        if x.shape[2] < 2:
            raise ValueError("The temporal window must contain at least two samples.")

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


@dataclass(frozen=True)
class FlatMLPEncoderConfig:
    """Configuration for the flattened-window MLP baseline encoder."""

    input_channels: int = 4
    window_length: int = 12
    hidden_dim: int = 32
    embedding_dim: int = 8
    dropout: float = 0.10
    embedding_activation: EmbeddingActivation = "tanh"

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.window_length <= 1:
            raise ValueError("window_length must exceed one sample.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.embedding_activation not in {"none", "tanh"}:
            raise ValueError(
                "embedding_activation must be either 'none' or 'tanh'."
            )

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


class FlatMLPEncoder(nn.Module):
    """Flatten a complete multichannel window and encode it with a small MLP."""

    def __init__(self, config: FlatMLPEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        layers: list[nn.Module] = [
            nn.Flatten(start_dim=1),
            nn.Linear(
                config.input_channels * config.window_length,
                config.hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        ]
        if config.embedding_activation == "tanh":
            layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 3:
            raise ValueError(
                "FlatMLPEncoder expects [batch, channels, time], "
                f"received {tuple(x.shape)}."
            )
        expected = (
            self.config.input_channels,
            self.config.window_length,
        )
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected channels/time={expected}, received {tuple(x.shape[1:])}."
            )
        return self.network(x)

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


@dataclass(frozen=True)
class InvariantMLPEncoderConfig:
    """Configuration for the calibration-aware temporal-feature encoder."""

    input_channels: int = 4
    window_length: int = 12
    hidden_dim: int = 24
    embedding_dim: int = 8
    dropout: float = 0.15
    embedding_activation: EmbeddingActivation = "tanh"
    use_absolute: bool = True
    use_relative: bool = True
    use_differences: bool = True
    use_shape: bool = True
    use_statistics: bool = True
    use_level_statistics: bool = True
    use_invariant_statistics: bool = True
    epsilon: float = 1e-4

    def validate(self) -> None:
        if self.input_channels <= 0:
            raise ValueError("input_channels must be positive.")
        if self.window_length <= 1:
            raise ValueError("window_length must exceed one sample.")
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive.")
        if self.embedding_dim <= 0:
            raise ValueError("embedding_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0, 1).")
        if self.embedding_activation not in {"none", "tanh"}:
            raise ValueError(
                "embedding_activation must be either 'none' or 'tanh'."
            )
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive.")
        statistics_enabled = self.use_statistics and (
                self.use_level_statistics or self.use_invariant_statistics
        )

        if not any(
                (
                        self.use_absolute,
                        self.use_relative,
                        self.use_differences,
                        self.use_shape,
                        statistics_enabled,
                )
        ):
            raise ValueError("At least one feature family must be enabled.")

        if self.use_statistics and not (
                self.use_level_statistics or self.use_invariant_statistics
        ):
            raise ValueError(
                "When use_statistics=True, at least one of "
                "use_level_statistics or use_invariant_statistics must be enabled."
            )

    def to_dict(self) -> dict[str, int | float | str | bool]:
        return asdict(self)


class InvariantMLPEncoder(nn.Module):
    """Encode baseline-calibrated levels, changes, trends, and temporal shape.

    The input is expected to have already been normalised using statistics fitted
    on the training episodes. With per-sensor background normalisation, absolute
    values represent deviations from each sensor's own non-fire baseline. The
    encoder then supplements those calibrated values with offset-insensitive
    relative trajectories, first differences, within-window standardised shape,
    and explicit temporal statistics.
    """

    _LEVEL_STATISTICS_PER_CHANNEL = 4
    _INVARIANT_STATISTICS_PER_CHANNEL = 3

    def __init__(self, config: InvariantMLPEncoderConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config

        feature_dim = self._feature_dimension(config)
        layers: list[nn.Module] = [
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.embedding_dim),
        ]
        if config.embedding_activation == "tanh":
            layers.append(nn.Tanh())
        self.network = nn.Sequential(*layers)

        time = torch.linspace(-1.0, 1.0, config.window_length)
        self.register_buffer("slope_time", time, persistent=False)
        self.register_buffer(
            "slope_denominator",
            torch.sum(time.square()).clamp_min(config.epsilon),
            persistent=False,
        )

    @staticmethod
    def _feature_dimension(config: InvariantMLPEncoderConfig) -> int:
        channels = config.input_channels
        length = config.window_length
        dimension = 0
        if config.use_absolute:
            dimension += channels * length
        if config.use_relative:
            dimension += channels * length
        if config.use_differences:
            dimension += channels * (length - 1)
        if config.use_shape:
            dimension += channels * length
        if config.use_statistics:
            if config.use_level_statistics:
                dimension += (
                        channels
                        * InvariantMLPEncoder._LEVEL_STATISTICS_PER_CHANNEL
                )
            if config.use_invariant_statistics:
                dimension += (
                        channels
                        * InvariantMLPEncoder._INVARIANT_STATISTICS_PER_CHANNEL
                )
        return dimension

    def _validate_input(self, x: Tensor) -> None:
        if x.ndim != 3:
            raise ValueError(
                "InvariantMLPEncoder expects [batch, channels, time], "
                f"received {tuple(x.shape)}."
            )
        expected = (
            self.config.input_channels,
            self.config.window_length,
        )
        if tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"Expected channels/time={expected}, received {tuple(x.shape[1:])}."
            )

    def feature_vector(self, x: Tensor) -> Tensor:
        self._validate_input(x)
        features: list[Tensor] = []

        if self.config.use_absolute:
            features.append(x.flatten(start_dim=1))

        relative = x - x[..., :1]
        if self.config.use_relative:
            features.append(relative.flatten(start_dim=1))

        differences = x[..., 1:] - x[..., :-1]
        if self.config.use_differences:
            features.append(differences.flatten(start_dim=1))

        channel_mean = x.mean(dim=-1, keepdim=True)
        channel_std = x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(
            self.config.epsilon
        )
        if self.config.use_shape:
            shape = (x - channel_mean) / channel_std
            features.append(shape.flatten(start_dim=1))

        if self.config.use_statistics:
            mean = channel_mean.squeeze(-1)
            std = channel_std.squeeze(-1)
            minimum = x.amin(dim=-1)
            maximum = x.amax(dim=-1)
            last = x[..., -1]
            endpoint_change = x[..., -1] - x[..., 0]

            centered = x - channel_mean
            slope = torch.sum(
                centered * self.slope_time.view(1, 1, -1),
                dim=-1,
            ) / self.slope_denominator

            selected_statistics: list[Tensor] = []

            # Preserve the original ordering when both families are enabled:
            # mean, std, minimum, maximum, last, endpoint change, slope.
            if self.config.use_level_statistics:
                selected_statistics.append(mean)

            if self.config.use_invariant_statistics:
                selected_statistics.append(std)

            if self.config.use_level_statistics:
                selected_statistics.extend(
                    [
                        minimum,
                        maximum,
                        last,
                    ]
                )

            if self.config.use_invariant_statistics:
                selected_statistics.extend(
                    [
                        endpoint_change,
                        slope,
                    ]
                )

            statistics = torch.stack(
                selected_statistics,
                dim=-1,
            )
            features.append(statistics.flatten(start_dim=1))

        return torch.cat(features, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        return self.network(self.feature_vector(x))

    @property
    def embedding_dim(self) -> int:
        return self.config.embedding_dim

    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


class SensorClassifier(nn.Module):
    """
    Temporary supervised classifier used to train the sensor representation.

    After training, discard ``classification_head`` and freeze ``encoder``.
    """

    def __init__(
        self,
        encoder: nn.Module,
        *,
        head_dropout: float = 0.0,
    ) -> None:
        super().__init__()

        if not 0.0 <= head_dropout < 1.0:
            raise ValueError("head_dropout must lie in [0, 1).")

        self.encoder = encoder
        self.classification_head = nn.Sequential(
            nn.Dropout(head_dropout),
            nn.Linear(encoder.embedding_dim, 1),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Return binary logits ``[batch]`` and embeddings ``[batch, dim]``."""
        embeddings = self.encoder(x)
        logits = self.classification_head(embeddings).squeeze(-1)
        return logits, embeddings

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """Return local fire probabilities with shape ``[batch]``."""
        logits, _ = self.forward(x)
        return torch.sigmoid(logits)

    def num_trainable_parameters(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )


def build_sensor_model(
    *,
    architecture: str = "cnn",
    input_channels: int = 4,
    window_length: int = 12,
    base_channels: int = 32,
    hidden_channels: int = 64,
    embedding_dim: int = 8,
    kernel_size: int = 5,
    dropout: float = 0.10,
    embedding_activation: EmbeddingActivation = "tanh",
    head_dropout: float = 0.0,
    invariant_use_absolute: bool = True,
    invariant_use_relative: bool = True,
    invariant_use_differences: bool = True,
    invariant_use_shape: bool = True,
    invariant_use_statistics: bool = True,
    invariant_use_level_statistics: bool = True,
    invariant_use_invariant_statistics: bool = True,
) -> SensorClassifier:
    """Construct a temporal CNN, flat MLP, or invariant temporal MLP."""
    architecture_key = architecture.strip().lower()
    if architecture_key in {"cnn", "temporal_cnn"}:
        encoder_config = SensorEncoderConfig(
            input_channels=input_channels,
            base_channels=base_channels,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            kernel_size=kernel_size,
            dropout=dropout,
            embedding_activation=embedding_activation,
        )
        encoder: nn.Module = SensorEncoder(encoder_config)
    elif architecture_key in {"flat_mlp", "mlp", "flattened_mlp"}:
        encoder_config = FlatMLPEncoderConfig(
            input_channels=input_channels,
            window_length=window_length,
            hidden_dim=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            embedding_activation=embedding_activation,
        )
        encoder = FlatMLPEncoder(encoder_config)
    elif architecture_key in {
        "invariant_mlp",
        "calibrated_mlp",
        "relative_mlp",
    }:
        encoder_config = InvariantMLPEncoderConfig(
            input_channels=input_channels,
            window_length=window_length,
            hidden_dim=hidden_channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
            embedding_activation=embedding_activation,
            use_absolute=invariant_use_absolute,
            use_relative=invariant_use_relative,
            use_differences=invariant_use_differences,
            use_shape=invariant_use_shape,
            use_statistics=invariant_use_statistics,
            use_level_statistics=invariant_use_level_statistics,
            use_invariant_statistics=invariant_use_invariant_statistics,
        )
        encoder = InvariantMLPEncoder(encoder_config)
    else:
        raise ValueError(
            "architecture must be one of: cnn, flat_mlp, invariant_mlp. "
            f"Received {architecture!r}."
        )
    return SensorClassifier(encoder, head_dropout=head_dropout)


def extract_encoder_state_dict(
    checkpoint: dict[str, object] | dict[str, Tensor],
) -> dict[str, Tensor]:
    """Extract encoder weights from a full-model or encoder-only state dict."""
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        if not isinstance(state_dict, dict):
            raise TypeError("checkpoint['model_state_dict'] must be a state dict.")
    else:
        state_dict = checkpoint

    encoder_state: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        if key.startswith("encoder."):
            encoder_state[key.removeprefix("encoder.")] = value
        elif not key.startswith("classification_head."):
            encoder_state[key] = value

    if not encoder_state:
        raise ValueError("No encoder parameters were found in the checkpoint.")
    return encoder_state
