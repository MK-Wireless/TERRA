from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UniformPerComponentQuantizer:
    """Uniform scalar quantizer with one fixed bit depth for every component.

    This is the main per-element scalar quantizer for the TERRA communication
    study. Multi-element reports should already be standardized using the
    frozen regional receiver's
    development-set feature_mean/feature_std. All components then share one
    predetermined clipping interval. No feature-specific ranges or bit
    allocations are learned.
    """

    dimension: int
    bits_per_component: int
    lower: float = -4.0
    upper: float = 4.0
    reconstruction_mode: str = "endpoints"

    def validate(self) -> None:
        if int(self.dimension) <= 0:
            raise ValueError("dimension must be positive.")
        if int(self.bits_per_component) <= 0:
            raise ValueError("bits_per_component must be positive.")
        if not math.isfinite(float(self.lower)) or not math.isfinite(float(self.upper)):
            raise ValueError("Quantization bounds must be finite.")
        if float(self.upper) <= float(self.lower):
            raise ValueError("upper must exceed lower.")
        if self.reconstruction_mode not in {"endpoints", "midrise"}:
            raise ValueError(
                "reconstruction_mode must be either 'endpoints' or 'midrise'."
            )

    @property
    def total_bits(self) -> int:
        self.validate()
        return int(self.dimension) * int(self.bits_per_component)

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != int(self.dimension):
            raise ValueError(
                f"Expected final dimension {self.dimension}, found {values.shape[-1]}."
            )
        if self.reconstruction_mode == "endpoints":
            reconstructed = quantize_uniform_scalar_array(
                values,
                lower=float(self.lower),
                upper=float(self.upper),
                bits=int(self.bits_per_component),
            )
        else:
            reconstructed = quantize_uniform_midrise_scalar_array(
                values,
                lower=float(self.lower),
                upper=float(self.upper),
                bits=int(self.bits_per_component),
            )
        return reconstructed.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "scheme": "uniform_per_component_fixed_clip",
            "dimension": int(self.dimension),
            "bits_per_component": int(self.bits_per_component),
            "total_bits": int(self.total_bits),
            "lower": float(self.lower),
            "upper": float(self.upper),
            "reconstruction_mode": str(self.reconstruction_mode),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "UniformPerComponentQuantizer":
        quantizer = cls(
            dimension=int(payload["dimension"]),
            bits_per_component=int(payload["bits_per_component"]),
            lower=float(payload["lower"]),
            upper=float(payload["upper"]),
            reconstruction_mode=str(payload.get("reconstruction_mode", "endpoints")),
        )
        quantizer.validate()
        return quantizer


def build_uniform_per_component_quantizer(
    *,
    dimension: int,
    bits_per_component: int,
    lower: float = -4.0,
    upper: float = 4.0,
    reconstruction_mode: str = "endpoints",
) -> UniformPerComponentQuantizer:
    quantizer = UniformPerComponentQuantizer(
        dimension=int(dimension),
        bits_per_component=int(bits_per_component),
        lower=float(lower),
        upper=float(upper),
        reconstruction_mode=str(reconstruction_mode),
    )
    quantizer.validate()
    return quantizer


def quantization_diagnostics(
    values: np.ndarray,
    quantizer: UniformPerComponentQuantizer,
) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != quantizer.dimension:
        raise ValueError(
            f"Expected [N,{quantizer.dimension}] diagnostic values, found {values.shape}."
        )
    if values.shape[0] == 0:
        raise ValueError("No values supplied for diagnostics.")
    reconstructed = quantizer.transform(values).astype(np.float64)
    clipped = np.clip(values, quantizer.lower, quantizer.upper)
    return {
        "n_reports": int(values.shape[0]),
        "dimension": int(values.shape[1]),
        "task_bits": int(quantizer.total_bits),
        "bits_per_component": int(quantizer.bits_per_component),
        "clip_fraction_elements": float(
            np.mean((values < quantizer.lower) | (values > quantizer.upper))
        ),
        "mse_to_original": float(np.mean((values - reconstructed) ** 2)),
        "mse_after_clipping_only": float(np.mean((values - clipped) ** 2)),
        "mse_quantization_given_clipped": float(np.mean((clipped - reconstructed) ** 2)),
    }


def quantize_uniform_scalar_array(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    bits: int,
) -> np.ndarray:
    """Affine uniform quantization to 2**bits reconstruction levels."""
    if bits <= 0:
        raise ValueError("bits must be positive.")
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Expected finite quantization bounds with upper > lower.")
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    effective_bits = min(int(bits), 52)
    levels_minus_one = float((1 << effective_bits) - 1)
    normalized = (clipped - lower) / (upper - lower)
    codes = np.rint(normalized * levels_minus_one)
    return lower + (codes / levels_minus_one) * (upper - lower)


def quantize_uniform_midrise_scalar_array(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    bits: int,
) -> np.ndarray:
    """Uniform mid-rise scalar quantization using bin-center reconstruction.

    For [-4,4] at 1 bit, reconstruction levels are {-2,+2}.
    """
    if bits <= 0:
        raise ValueError("bits must be positive.")
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Expected finite quantization bounds with upper > lower.")
    values = np.asarray(values, dtype=np.float64)
    clipped = np.clip(values, lower, upper)
    effective_bits = min(int(bits), 52)
    n_levels = float(1 << effective_bits)
    step = (upper - lower) / n_levels
    codes = np.floor((clipped - lower) / step)
    codes = np.clip(codes, 0.0, n_levels - 1.0)
    return lower + (codes + 0.5) * step


def quantize_probability(values: np.ndarray, payload_bits: int) -> np.ndarray:
    return quantize_uniform_scalar_array(
        np.asarray(values, dtype=np.float64),
        lower=0.0,
        upper=1.0,
        bits=int(payload_bits),
    ).astype(np.float32)


# ---------------------------------------------------------------------------
# Legacy API retained for previous experiment scripts.
# The new main TERRA study above does NOT use greedy/nonuniform allocation.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FixedPayloadUniformQuantizer:
    """LEGACY: per-dimension quantizer with greedy bit allocation."""

    lower: np.ndarray
    upper: np.ndarray
    bits_per_dimension: np.ndarray
    total_bits: int
    minimum_bits_per_dimension: int
    lower_quantile: float
    upper_quantile: float
    training_mse: float

    def validate(self) -> None:
        lower = np.asarray(self.lower)
        upper = np.asarray(self.upper)
        bits = np.asarray(self.bits_per_dimension)
        if lower.ndim != 1 or upper.ndim != 1 or bits.ndim != 1:
            raise ValueError("Quantizer parameters must be one-dimensional.")
        if not (lower.shape == upper.shape == bits.shape):
            raise ValueError("Quantizer parameter shapes disagree.")
        if lower.size == 0:
            raise ValueError("Quantizer must contain at least one dimension.")
        if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
            raise ValueError("Quantizer ranges must be finite.")
        if np.any(upper <= lower):
            raise ValueError("Every quantizer upper bound must exceed its lower bound.")
        if np.any(bits < self.minimum_bits_per_dimension):
            raise ValueError("A dimension received fewer than the configured minimum bits.")
        if int(bits.sum()) != int(self.total_bits):
            raise ValueError("Per-dimension bits do not sum to total_bits.")

    @property
    def dimension(self) -> int:
        return int(np.asarray(self.bits_per_dimension).size)

    def transform(self, values: np.ndarray) -> np.ndarray:
        self.validate()
        values = np.asarray(values, dtype=np.float64)
        if values.shape[-1] != self.dimension:
            raise ValueError(
                f"Expected final dimension {self.dimension}, found {values.shape[-1]}."
            )
        output = np.empty_like(values, dtype=np.float64)
        for index in range(self.dimension):
            output[..., index] = quantize_uniform_scalar_array(
                values[..., index],
                lower=float(self.lower[index]),
                upper=float(self.upper[index]),
                bits=int(self.bits_per_dimension[index]),
            )
        return output.astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "scheme": "legacy_greedy_fixed_payload",
            "lower": np.asarray(self.lower, dtype=float).tolist(),
            "upper": np.asarray(self.upper, dtype=float).tolist(),
            "bits_per_dimension": np.asarray(self.bits_per_dimension, dtype=int).tolist(),
            "total_bits": int(self.total_bits),
            "minimum_bits_per_dimension": int(self.minimum_bits_per_dimension),
            "lower_quantile": float(self.lower_quantile),
            "upper_quantile": float(self.upper_quantile),
            "training_mse": float(self.training_mse),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedPayloadUniformQuantizer":
        quantizer = cls(
            lower=np.asarray(payload["lower"], dtype=np.float32),
            upper=np.asarray(payload["upper"], dtype=np.float32),
            bits_per_dimension=np.asarray(payload["bits_per_dimension"], dtype=np.int64),
            total_bits=int(payload["total_bits"]),
            minimum_bits_per_dimension=int(payload["minimum_bits_per_dimension"]),
            lower_quantile=float(payload["lower_quantile"]),
            upper_quantile=float(payload["upper_quantile"]),
            training_mse=float(payload["training_mse"]),
        )
        quantizer.validate()
        return quantizer


def _dimension_mse(
    values: np.ndarray,
    *,
    lower: float,
    upper: float,
    bits: int,
) -> float:
    reconstructed = quantize_uniform_scalar_array(
        values, lower=lower, upper=upper, bits=bits
    )
    return float(
        np.mean((np.asarray(values, dtype=np.float64) - reconstructed) ** 2)
    )


def _robust_ranges(
    values: np.ndarray,
    *,
    lower_quantile: float,
    upper_quantile: float,
    minimum_span: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= lower_quantile < upper_quantile <= 1.0):
        raise ValueError("Quantiles must satisfy 0 <= lower < upper <= 1.")
    lower = np.quantile(values, lower_quantile, axis=0).astype(np.float64)
    upper = np.quantile(values, upper_quantile, axis=0).astype(np.float64)
    median = np.median(values, axis=0).astype(np.float64)
    narrow = (upper - lower) < minimum_span
    lower[narrow] = median[narrow] - 0.5 * minimum_span
    upper[narrow] = median[narrow] + 0.5 * minimum_span
    return lower, upper


def fit_fixed_payload_uniform_quantizer(
    values: np.ndarray,
    *,
    total_bits: int,
    minimum_bits_per_dimension: int = 1,
    lower_quantile: float = 0.005,
    upper_quantile: float = 0.995,
) -> FixedPayloadUniformQuantizer:
    """LEGACY fitter retained so earlier scripts remain reproducible."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("values must have shape [N_reports, D] with nonzero sizes.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Quantizer fitting values contain non-finite entries.")
    total_bits = int(total_bits)
    minimum_bits_per_dimension = int(minimum_bits_per_dimension)
    dimension = int(values.shape[1])
    minimum_payload = dimension * minimum_bits_per_dimension
    if minimum_bits_per_dimension <= 0 or total_bits < minimum_payload:
        raise ValueError("Invalid legacy total-bit budget.")

    lower, upper = _robust_ranges(
        values,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    bits = np.full(dimension, minimum_bits_per_dimension, dtype=np.int64)
    current_mse = np.asarray(
        [
            _dimension_mse(
                values[:, i],
                lower=float(lower[i]),
                upper=float(upper[i]),
                bits=int(bits[i]),
            )
            for i in range(dimension)
        ],
        dtype=np.float64,
    )

    remaining = total_bits - int(bits.sum())
    for _ in range(remaining):
        next_mse = np.asarray(
            [
                _dimension_mse(
                    values[:, i],
                    lower=float(lower[i]),
                    upper=float(upper[i]),
                    bits=int(bits[i] + 1),
                )
                for i in range(dimension)
            ],
            dtype=np.float64,
        )
        reductions = current_mse - next_mse
        chosen = int(np.argmax(reductions))
        bits[chosen] += 1
        current_mse[chosen] = next_mse[chosen]

    quantizer = FixedPayloadUniformQuantizer(
        lower=lower.astype(np.float32),
        upper=upper.astype(np.float32),
        bits_per_dimension=bits,
        total_bits=total_bits,
        minimum_bits_per_dimension=minimum_bits_per_dimension,
        lower_quantile=float(lower_quantile),
        upper_quantile=float(upper_quantile),
        training_mse=float(np.mean(current_mse)),
    )
    quantizer.validate()
    return quantizer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Uniform-per-component quantizer self-check."
    )
    parser.add_argument("--dimension", type=int, default=4)
    parser.add_argument("--bits-per-component", type=int, default=4)
    parser.add_argument("--lower", type=float, default=-4.0)
    parser.add_argument("--upper", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-json", type=Path, default=None)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)
    values = rng.normal(size=(2000, args.dimension)).astype(np.float32)
    quantizer = build_uniform_per_component_quantizer(
        dimension=args.dimension,
        bits_per_component=args.bits_per_component,
        lower=args.lower,
        upper=args.upper,
    )
    reconstructed = quantizer.transform(values)
    mse = float(np.mean((values - reconstructed) ** 2))
    print(f"Dimension:          {quantizer.dimension}")
    print(f"Bits/component:     {quantizer.bits_per_component}")
    print(f"Task payload bits:  {quantizer.total_bits}")
    print(f"Clip interval:      [{quantizer.lower}, {quantizer.upper}]")
    print(f"Measured MSE:       {mse:.8f}")
    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(
            json.dumps(quantizer.to_dict(), indent=2), encoding="utf-8"
        )
        print(f"Saved:              {args.save_json}")


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        _main()
    finally:
        print(f"Total runtime: {time.perf_counter() - started:.2f} seconds")
