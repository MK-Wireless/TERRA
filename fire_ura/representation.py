from __future__ import annotations

import argparse
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor, nn

from fire_ura.sensor_model import (
    FlatMLPEncoder,
    FlatMLPEncoderConfig,
    InvariantMLPEncoder,
    InvariantMLPEncoderConfig,
    SensorClassifier,
    SensorEncoder,
    SensorEncoderConfig,
)


class CheckpointNormalization:
    """Reconstruct training-only normalization saved in a sensor checkpoint."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        self.payload = dict(payload)
        self.mode = str(self.payload.get("mode", "global"))

        if self.mode == "global":
            self.global_mean = dict(self.payload["mean"])
            self.global_std = dict(self.payload["std"])
            self.sensor_mean: dict[str, dict[str, float]] = {}
            self.sensor_std: dict[str, dict[str, float]] = {}
        elif self.mode == "per_sensor_background":
            self.global_mean = dict(self.payload["global_mean"])
            self.global_std = dict(self.payload["global_std"])
            self.sensor_mean = {
                str(sensor_id): dict(values)
                for sensor_id, values in self.payload.get("sensor_mean", {}).items()
            }
            self.sensor_std = {
                str(sensor_id): dict(values)
                for sensor_id, values in self.payload.get("sensor_std", {}).items()
            }
        else:
            raise ValueError(f"Unsupported checkpoint normalization mode: {self.mode!r}")

    def transform(
        self,
        values: np.ndarray,
        channels: Sequence[str],
        *,
        sensor_id: str | None = None,
    ) -> np.ndarray:
        key = str(sensor_id) if sensor_id is not None else ""
        means_map = self.sensor_mean.get(key, self.global_mean)
        stds_map = self.sensor_std.get(key, self.global_std)
        means = np.asarray([means_map[channel] for channel in channels], dtype=np.float32)
        stds = np.asarray([stds_map[channel] for channel in channels], dtype=np.float32)
        stds = np.where(np.abs(stds) < 1e-12, 1.0, stds)
        return (values.astype(np.float32) - means) / stds

    def as_dict(self) -> dict[str, Any]:
        return dict(self.payload)


def _torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint {path} did not contain a dictionary.")
    return checkpoint


def _dataclass_kwargs(config_class: type, payload: Mapping[str, Any]) -> dict[str, Any]:
    allowed = {field.name for field in fields(config_class)}
    return {key: value for key, value in payload.items() if key in allowed}


def build_encoder_from_config(payload: Mapping[str, Any]) -> nn.Module:
    """Infer the sensor encoder family from its saved dataclass configuration."""
    config = dict(payload)
    if any(key.startswith("use_") for key in config):
        encoder_config = InvariantMLPEncoderConfig(
            **_dataclass_kwargs(InvariantMLPEncoderConfig, config)
        )
        return InvariantMLPEncoder(encoder_config)
    if "base_channels" in config or "kernel_size" in config:
        encoder_config = SensorEncoderConfig(
            **_dataclass_kwargs(SensorEncoderConfig, config)
        )
        return SensorEncoder(encoder_config)
    if "window_length" in config and "hidden_dim" in config:
        encoder_config = FlatMLPEncoderConfig(
            **_dataclass_kwargs(FlatMLPEncoderConfig, config)
        )
        return FlatMLPEncoder(encoder_config)
    raise ValueError(
        "Could not infer the encoder architecture from checkpoint encoder_config. "
        f"Keys: {sorted(config)}"
    )


def derive_classifier_path(encoder_path: Path) -> Path:
    name = encoder_path.name
    if "best_encoder_" not in name:
        raise ValueError(
            "Automatic classifier-path derivation expects an encoder filename "
            "containing 'best_encoder_'."
        )
    return encoder_path.with_name(name.replace("best_encoder_", "best_classifier_", 1))


class FoldSensorProjector:
    """Load one fold-specific local model and extract receiver payloads."""

    def __init__(
        self,
        encoder_checkpoint: str | Path,
        *,
        classifier_checkpoint: str | Path | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.encoder_checkpoint_path = Path(encoder_checkpoint)
        if not self.encoder_checkpoint_path.is_file():
            raise FileNotFoundError(self.encoder_checkpoint_path)

        encoder_checkpoint_data = _torch_load(
            self.encoder_checkpoint_path,
            self.device,
        )
        encoder_config = encoder_checkpoint_data.get("encoder_config")
        if not isinstance(encoder_config, Mapping):
            raise KeyError("Encoder checkpoint is missing encoder_config.")

        self.channels = tuple(str(value) for value in encoder_checkpoint_data["channels"])
        self.window_config = dict(encoder_checkpoint_data["window"])
        self.normalization = CheckpointNormalization(
            encoder_checkpoint_data["normalization"]
        )
        self.encoder = build_encoder_from_config(encoder_config).to(self.device)
        encoder_state = encoder_checkpoint_data.get("model_state_dict")
        if not isinstance(encoder_state, Mapping):
            raise KeyError("Encoder checkpoint is missing model_state_dict.")
        self.encoder.load_state_dict(dict(encoder_state), strict=True)
        self.encoder.eval()

        candidate_classifier = (
            Path(classifier_checkpoint)
            if classifier_checkpoint is not None
            else derive_classifier_path(self.encoder_checkpoint_path)
        )
        self.classifier: SensorClassifier | None = None
        self.classifier_checkpoint_path: Path | None = None
        if candidate_classifier.is_file():
            classifier_data = _torch_load(candidate_classifier, self.device)
            classifier_encoder = build_encoder_from_config(encoder_config)
            classifier = SensorClassifier(classifier_encoder, head_dropout=0.0).to(self.device)
            state = classifier_data.get("model_state_dict")
            if not isinstance(state, Mapping):
                raise KeyError("Classifier checkpoint is missing model_state_dict.")
            classifier.load_state_dict(dict(state), strict=True)
            classifier.eval()
            self.classifier = classifier
            self.classifier_checkpoint_path = candidate_classifier

    @property
    def embedding_dim(self) -> int:
        return int(self.encoder.embedding_dim)

    @torch.no_grad()
    def project(self, x: Tensor) -> tuple[Tensor, Tensor | None]:
        x = x.to(self.device, dtype=torch.float32)
        if self.classifier is not None:
            logits, embeddings = self.classifier(x)
            return embeddings, torch.sigmoid(logits)
        embeddings = self.encoder(x)
        return embeddings, None


def temporal_slope(x: Tensor) -> Tensor:
    """Least-squares temporal slope for a batch shaped [B, C, T]."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, channels, time], received {tuple(x.shape)}")
    length = x.shape[-1]
    time_axis = torch.arange(length, device=x.device, dtype=x.dtype)
    centered_time = time_axis - time_axis.mean()
    denominator = centered_time.square().sum().clamp_min(1e-12)
    return torch.sum(x * centered_time.view(1, 1, -1), dim=-1) / denominator


def handcrafted_temporal_features(x: Tensor) -> Tensor:
    """Return the established seven statistics per channel from normalized windows."""
    if x.ndim != 3:
        raise ValueError(f"Expected [batch, channels, time], received {tuple(x.shape)}")
    statistics = (
        x.mean(dim=-1),
        x.std(dim=-1, unbiased=False),
        x.amin(dim=-1),
        x.amax(dim=-1),
        x[..., -1],
        x[..., -1] - x[..., 0],
        temporal_slope(x),
    )
    return torch.cat(statistics, dim=1)


def handcrafted_feature_names(channels: Sequence[str]) -> list[str]:
    names = ("mean", "std", "min", "max", "last", "delta", "slope")
    return [f"{channel}__{name}" for name in names for channel in channels]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect one sensor encoder checkpoint.")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    projector = FoldSensorProjector(args.checkpoint, device=args.device)
    print(f"Checkpoint:    {projector.encoder_checkpoint_path}")
    print(f"Channels:      {projector.channels}")
    print(f"Window:        {projector.window_config}")
    print(f"Embedding dim: {projector.embedding_dim}")
    print(f"Classifier:    {projector.classifier_checkpoint_path or 'not found'}")


if __name__ == "__main__":
    _started = time.perf_counter()
    try:
        _main()
    finally:
        _elapsed = time.perf_counter() - _started
        print(f"Total runtime: {_elapsed:.2f} seconds")
