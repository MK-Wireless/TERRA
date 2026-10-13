from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass

import torch
from torch import Tensor, nn


def masked_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Mean-pool valid set elements while ignoring padded positions."""
    if x.ndim != 3:
        raise ValueError(f"Expected x with shape [B,K,D], received {tuple(x.shape)}.")
    if mask.ndim != 2 or tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError(
            f"Expected mask with shape {tuple(x.shape[:2])}, received {tuple(mask.shape)}."
        )
    weights = mask.to(dtype=x.dtype).unsqueeze(-1)
    denominator = weights.sum(dim=1).clamp_min(1.0)
    return (x * weights).sum(dim=1) / denominator


@dataclass(frozen=True)
class MeanPoolMLPConfig:
    input_dim: int
    hidden_dim: int = 32
    dropout: float = 0.15

    def validate(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class MeanPoolMLP(nn.Module):
    """Nonlinear classifier applied after simple mean pooling."""

    def __init__(self, config: MeanPoolMLPConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.network = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        return self.network(masked_mean(x, mask)).squeeze(-1)


@dataclass(frozen=True)
class DeepSetsConfig:
    input_dim: int
    phi_hidden_dim: int = 32
    phi_output_dim: int = 16
    rho_hidden_dim: int = 16
    dropout: float = 0.15

    def validate(self) -> None:
        dimensions = (
            self.input_dim,
            self.phi_hidden_dim,
            self.phi_output_dim,
            self.rho_hidden_dim,
        )
        if any(value <= 0 for value in dimensions):
            raise ValueError("All DeepSets dimensions must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must lie in [0,1).")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class DeepSetsClassifier(nn.Module):
    """Permutation-invariant region classifier with masked mean aggregation."""

    def __init__(self, config: DeepSetsConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.phi = nn.Sequential(
            nn.Linear(config.input_dim, config.phi_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.phi_hidden_dim, config.phi_output_dim),
            nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(config.phi_output_dim, config.rho_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.rho_hidden_dim, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        transformed = self.phi(x)
        pooled = masked_mean(transformed, mask)
        return self.rho(pooled).squeeze(-1)


def build_region_model(
    architecture: str,
    *,
    input_dim: int,
    hidden_dim: int = 32,
    phi_hidden_dim: int = 32,
    phi_output_dim: int = 16,
    rho_hidden_dim: int = 16,
    dropout: float = 0.15,
) -> nn.Module:
    key = architecture.strip().lower()
    if key in {"mean_mlp", "mean_pool_mlp"}:
        return MeanPoolMLP(
            MeanPoolMLPConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
        )
    if key in {"deepsets", "deep_sets"}:
        return DeepSetsClassifier(
            DeepSetsConfig(
                input_dim=input_dim,
                phi_hidden_dim=phi_hidden_dim,
                phi_output_dim=phi_output_dim,
                rho_hidden_dim=rho_hidden_dim,
                dropout=dropout,
            )
        )
    raise ValueError(f"Unsupported regional architecture: {architecture!r}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a regional-model shape check.")
    parser.add_argument("--architecture", choices=("mean_mlp", "deepsets"), default="deepsets")
    parser.add_argument("--input-dim", type=int, default=8)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    model = build_region_model(args.architecture, input_dim=args.input_dim)
    x = torch.randn(4, 9, args.input_dim)
    mask = torch.ones(4, 9, dtype=torch.bool)
    logits = model(x, mask)
    print(model)
    print(f"Output shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    _started = time.perf_counter()
    try:
        _main()
    finally:
        _elapsed = time.perf_counter() - _started
        print(f"Total runtime: {_elapsed:.2f} seconds")
