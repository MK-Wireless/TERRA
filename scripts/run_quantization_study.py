#!/usr/bin/env python
from __future__ import annotations

"""
TERRA quantization / communication-rate study.

Schemes
-------
1) TERRA-Score:
   Quantize the scalar local fire probability on [0,1], then average recovered
   scores within a region.

2) TERRA-Feature:
   Use the selected learned representation (default D=4), standardize with the
   frozen regional receiver's development statistics, apply one common fixed
   clipping interval, and use the same uniform bit depth for every component.

3) Handcrafted benchmark:
   Use the selected mean+std handcrafted representation (default D=30), with
   the same independent per-element uniform scalar quantization protocol as
   TERRA-Feature.

This is post-training quantization. Regional receivers are frozen.

Bit accounting
--------------
task_bits:
    Quantized task payload only.
message_bits:
    region_tag_bits + nonce_bits + task_bits.
regional_message_bits:
    K * message_bits for an exact-K recovered-report condition.

The tag/nonce terms are configurable. If their final values are not yet fixed,
set them to zero; the task-bit results remain valid and the communication
overhead can be inserted later without retraining.

Evaluation aggregation
----------------------
subset realization -> mean within outer fold -> equal mean across 3 outer folds.

Primary metrics:
    AUPRC
    AUROC
    MDR@FAR<=1%
    MDR@FAR<=5%

No test-derived deployment threshold is exported.
"""

import argparse
import json
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from torch import nn

from fire_ura.quantization import (
    build_uniform_per_component_quantizer,
    quantization_diagnostics,
    quantize_probability,
)
from fire_ura.region_data import RegionArrays, load_region_arrays
from fire_ura.region_model import build_region_model
from fire_ura.utils import load_yaml, set_seed


MAIN_METRICS = (
    "auprc",
    "auroc",
    "mdr_at_far_1pct",
    "achieved_far_at_1pct",
    "mdr_at_far_5pct",
    "achieved_far_at_5pct",
)
LOWER_IS_BETTER = {
    "mdr_at_far_1pct",
    "mdr_at_far_5pct",
}


@dataclass(frozen=True)
class Settings:
    learned_bits_per_component: tuple[int, ...]
    handcrafted_bits_per_component: tuple[int, ...]
    score_bits: tuple[int, ...]
    k_values: tuple[int, ...]
    test_subset_seeds: int
    clip_lower: float
    clip_upper: float
    region_tag_bits: int
    nonce_bits: int
    batch_size: int
    save_predictions: bool
    hidden_dim: int
    phi_hidden_dim: int
    phi_output_dim: int
    rho_hidden_dim: int
    dropout: float
    handcrafted_stats: tuple[str, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TERRA quantization study.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate datasets/checkpoints/dimensions without evaluation.",
    )
    parser.add_argument(
        "--scalar-reconstruction",
        "--vector-quantizer",
        dest="scalar_reconstruction",
        choices=("endpoints", "midrise"),
        default="endpoints",
        help=(
            "Per-element scalar reconstruction convention for TERRA-Feature "
            "and Handcrafted-Stats. 'endpoints' reproduces the main study; "
            "'midrise' uses bin-center reconstruction."
        ),
    )
    parser.add_argument(
        "--learned-bits", type=int, nargs="+", default=None,
        help="Override TERRA-Feature bits/component from the YAML config.",
    )
    parser.add_argument(
        "--handcrafted-bits", type=int, nargs="+", default=None,
        help="Override handcrafted bits/component from the YAML config.",
    )
    parser.add_argument(
        "--score-bits", type=int, nargs="+", default=None,
        help="Override TERRA-Score scalar bit depths from the YAML config.",
    )
    parser.add_argument(
        "--k-values", type=int, nargs="+", default=None,
        help="Override exact-K values from the YAML config.",
    )
    return parser.parse_args()


def _positive_int_tuple(values: Sequence[Any], name: str) -> tuple[int, ...]:
    parsed = tuple(sorted({int(value) for value in values}))
    if not parsed or any(value <= 0 for value in parsed):
        raise ValueError(f"{name} must contain positive integers.")
    return parsed


def read_settings(config: Mapping[str, Any]) -> Settings:
    settings = Settings(
        learned_bits_per_component=_positive_int_tuple(
            config.get("learned_bits_per_component", [1, 2, 3, 4, 6, 8]),
            "learned_bits_per_component",
        ),
        handcrafted_bits_per_component=_positive_int_tuple(
            config.get("handcrafted_bits_per_component", [1, 2, 3, 4, 6, 8]),
            "handcrafted_bits_per_component",
        ),
        score_bits=_positive_int_tuple(
            config.get("score_bits", [1, 2, 3, 4, 6, 8]),
            "score_bits",
        ),
        k_values=_positive_int_tuple(
            config.get("k_values", [1, 2, 3, 5, 7, 9]),
            "k_values",
        ),
        test_subset_seeds=int(config.get("test_subset_seeds", 10)),
        clip_lower=float(config.get("clip_lower", -4.0)),
        clip_upper=float(config.get("clip_upper", 4.0)),
        region_tag_bits=int(config.get("region_tag_bits", 0)),
        nonce_bits=int(config.get("nonce_bits", 0)),
        batch_size=int(config.get("batch_size", 512)),
        save_predictions=bool(config.get("save_predictions", True)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        phi_hidden_dim=int(config.get("phi_hidden_dim", 32)),
        phi_output_dim=int(config.get("phi_output_dim", 16)),
        rho_hidden_dim=int(config.get("rho_hidden_dim", 16)),
        dropout=float(config.get("dropout", 0.15)),
        handcrafted_stats=tuple(
            str(value) for value in config.get("handcrafted_stats", ["mean", "std"])
        ),
    )
    if settings.test_subset_seeds <= 0:
        raise ValueError("test_subset_seeds must be positive.")
    if settings.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if settings.clip_upper <= settings.clip_lower:
        raise ValueError("clip_upper must exceed clip_lower.")
    if settings.region_tag_bits < 0 or settings.nonce_bits < 0:
        raise ValueError("region_tag_bits and nonce_bits must be nonnegative.")
    if not settings.handcrafted_stats:
        raise ValueError("handcrafted_stats cannot be empty.")
    return settings


def resolve_device(requested: str) -> torch.device:
    requested = requested.strip().lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(requested)


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(convert(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def safe_auprc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.size == 0:
        return float("nan")
    if np.unique(targets).size < 2:
        return float(targets.mean())
    return float(average_precision_score(targets, probabilities))


def safe_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if targets.size == 0 or np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def mdr_at_far(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_far: float,
) -> tuple[float, float]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if np.unique(targets).size < 2:
        return float("nan"), float("nan")

    fpr, tpr, _ = roc_curve(
        targets,
        probabilities,
        drop_intermediate=False,
    )
    feasible = np.flatnonzero(fpr <= float(target_far) + 1e-12)
    if feasible.size == 0:
        return float("nan"), float("nan")

    feasible_tpr = tpr[feasible]
    best_tpr = float(np.max(feasible_tpr))
    tied = feasible[
        np.isclose(feasible_tpr, best_tpr, rtol=0.0, atol=1e-12)
    ]
    best_index = int(tied[np.argmax(fpr[tied])])
    return float(1.0 - tpr[best_index]), float(fpr[best_index])


def metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    mdr1, far1 = mdr_at_far(targets, probabilities, target_far=0.01)
    mdr5, far5 = mdr_at_far(targets, probabilities, target_far=0.05)
    return {
        "n_examples": int(targets.size),
        "positive_fraction": float(targets.mean()),
        "auprc": safe_auprc(targets, probabilities),
        "auroc": safe_auroc(targets, probabilities),
        "mdr_at_far_1pct": mdr1,
        "achieved_far_at_1pct": far1,
        "mdr_at_far_5pct": mdr5,
        "achieved_far_at_5pct": far5,
    }


def exact_k_mask_numpy(base_mask: np.ndarray, k: int, seed: int) -> np.ndarray:
    base_mask = np.asarray(base_mask, dtype=bool)
    if base_mask.ndim != 2:
        raise ValueError("base_mask must have shape [N,max_reports].")
    if int(k) <= 0:
        raise ValueError("k must be positive.")

    rng = np.random.default_rng(int(seed))
    output = np.zeros_like(base_mask, dtype=bool)

    for row_index, row in enumerate(base_mask):
        valid = np.flatnonzero(row)
        keep = min(int(k), int(valid.size))
        if keep <= 0:
            raise ValueError("Every example must contain at least one report.")
        chosen = valid if keep == valid.size else rng.choice(
            valid, size=keep, replace=False
        )
        output[row_index, chosen] = True

    return output


def evaluation_seeds_for_k(
    base_mask: np.ndarray,
    k: int,
    requested_count: int,
    seed_base: int,
) -> list[int]:
    if np.all(base_mask.sum(axis=1) <= int(k)):
        return [int(seed_base)]
    return [int(seed_base + offset) for offset in range(int(requested_count))]


def read_handcrafted_feature_names(dataset_root: Path, fold: int) -> list[str]:
    path = (
        dataset_root
        / f"fold_{fold}"
        / f"region_metadata_fold{fold}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Handcrafted feature metadata not found: {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("handcrafted_feature_names")
    if not isinstance(names, list) or not names:
        raise KeyError(
            f"{path} does not contain handcrafted_feature_names."
        )
    return [str(value) for value in names]


def handcrafted_indices(
    feature_names: Sequence[str],
    requested_stats: Sequence[str],
) -> np.ndarray:
    wanted = set(str(value) for value in requested_stats)
    indices: list[int] = []

    for index, name in enumerate(feature_names):
        if "__" not in name:
            raise ValueError(
                f"Handcrafted feature name {name!r} is not channel__stat."
            )
        stat = name.rsplit("__", 1)[1]
        if stat in wanted:
            indices.append(index)

    if not indices:
        raise ValueError(
            f"No handcrafted features matched statistics {sorted(wanted)}."
        )

    found = {
        feature_names[index].rsplit("__", 1)[1]
        for index in indices
    }
    if found != wanted:
        raise ValueError(
            f"Requested handcrafted stats {sorted(wanted)}, found {sorted(found)}."
        )

    return np.asarray(indices, dtype=np.int64)


def slice_handcrafted(
    arrays: RegionArrays,
    indices: np.ndarray,
) -> np.ndarray:
    return arrays.handcrafted[:, :, indices].astype(np.float32)


def assert_aligned(
    learned_arrays: RegionArrays,
    score_hc_arrays: RegionArrays,
    *,
    fold: int,
) -> None:
    if len(learned_arrays.target) != len(score_hc_arrays.target):
        raise ValueError(f"Fold {fold}: regional example counts differ.")
    if not np.array_equal(learned_arrays.target, score_hc_arrays.target):
        raise ValueError(f"Fold {fold}: targets differ between dataset roots.")
    if not np.array_equal(learned_arrays.split_code, score_hc_arrays.split_code):
        raise ValueError(f"Fold {fold}: split codes differ between dataset roots.")
    if not np.array_equal(learned_arrays.mask, score_hc_arrays.mask):
        raise ValueError(f"Fold {fold}: report masks differ between dataset roots.")

    for column in ("episode_id", "timestamp", "region_id"):
        if (
            column in learned_arrays.metadata.columns
            and column in score_hc_arrays.metadata.columns
        ):
            left = learned_arrays.metadata[column].astype(str).to_numpy()
            right = score_hc_arrays.metadata[column].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(
                    f"Fold {fold}: metadata column {column!r} is not aligned."
                )


def build_model(
    architecture: str,
    *,
    input_dim: int,
    settings: Settings,
    device: torch.device,
) -> nn.Module:
    return build_region_model(
        architecture,
        input_dim=int(input_dim),
        hidden_dim=settings.hidden_dim,
        phi_hidden_dim=settings.phi_hidden_dim,
        phi_output_dim=settings.phi_output_dim,
        rho_hidden_dim=settings.rho_hidden_dim,
        dropout=settings.dropout,
    ).to(device)


def load_model(
    checkpoint_path: Path,
    *,
    representation: str,
    input_dim: int,
    settings: Settings,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    payload = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if str(payload.get("representation")) != representation:
        raise ValueError(
            f"Representation mismatch in {checkpoint_path}: "
            f"{payload.get('representation')!r} != {representation!r}"
        )

    architecture = str(payload.get("architecture"))
    if architecture != "deepsets":
        raise ValueError(
            f"This main study expects DeepSets checkpoints, found {architecture!r}."
        )

    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    if feature_mean.shape != (input_dim,) or feature_std.shape != (input_dim,):
        raise ValueError(
            f"{checkpoint_path}: standardizer shape mismatch for D={input_dim}: "
            f"{feature_mean.shape}, {feature_std.shape}"
        )

    model = build_model(
        architecture,
        input_dim=input_dim,
        settings=settings,
        device=device,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()

    return model, feature_mean, feature_std, payload


def standardized_all_reports(
    values: np.ndarray,
    mask: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    standardized = np.zeros_like(values, dtype=np.float32)
    standardized[mask] = (
        values[mask].astype(np.float32) - feature_mean
    ) / feature_std
    return standardized


@torch.no_grad()
def predict_vector_receiver(
    model: nn.Module,
    standardized_or_quantized_values: np.ndarray,
    *,
    indices: np.ndarray,
    subset_mask: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    values = standardized_or_quantized_values[indices].copy()
    values[~subset_mask] = 0.0

    outputs: list[np.ndarray] = []
    for start in range(0, len(indices), batch_size):
        stop = min(start + batch_size, len(indices))
        x = torch.from_numpy(values[start:stop]).to(
            device, dtype=torch.float32
        )
        mask = torch.from_numpy(subset_mask[start:stop]).to(
            device, dtype=torch.bool
        )
        logits = model(x, mask)
        outputs.append(
            torch.sigmoid(logits).cpu().numpy().astype(np.float64)
        )

    return np.concatenate(outputs)


def average_scores(
    local_probability: np.ndarray,
    *,
    indices: np.ndarray,
    subset_mask: np.ndarray,
) -> np.ndarray:
    selected = np.asarray(
        local_probability[indices],
        dtype=np.float64,
    )
    weights = subset_mask.astype(np.float64)
    denominator = weights.sum(axis=1).clip(min=1.0)
    return (selected * weights).sum(axis=1) / denominator


def message_accounting(
    *,
    task_bits: int,
    k: int,
    settings: Settings,
) -> dict[str, int]:
    overhead = int(settings.region_tag_bits) + int(settings.nonce_bits)
    message_bits = int(task_bits) + overhead
    return {
        "task_bits": int(task_bits),
        "region_tag_bits": int(settings.region_tag_bits),
        "nonce_bits": int(settings.nonce_bits),
        "shared_overhead_bits": int(overhead),
        "message_bits": int(message_bits),
        "regional_task_bits": int(k) * int(task_bits),
        "regional_message_bits": int(k) * int(message_bits),
    }


def append_condition_row(
    destination: list[dict[str, Any]],
    *,
    fold: int,
    scheme: str,
    dimension: int,
    bits_per_component: int,
    task_bits: int,
    k: int,
    subset_seed: int,
    metric_values: Mapping[str, Any],
    settings: Settings,
) -> None:
    destination.append(
        {
            "fold": int(fold),
            "scheme": str(scheme),
            "quantization_mode": "quantized",
            "dimension": int(dimension),
            "bits_per_component": int(bits_per_component),
            "k": int(k),
            "subset_seed": int(subset_seed),
            **message_accounting(
                task_bits=task_bits,
                k=k,
                settings=settings,
            ),
            **dict(metric_values),
        }
    )


def aggregate_subset_seeds(seed_table: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "fold",
        "scheme",
        "quantization_mode",
        "dimension",
        "bits_per_component",
        "task_bits",
        "region_tag_bits",
        "nonce_bits",
        "shared_overhead_bits",
        "message_bits",
        "k",
        "regional_task_bits",
        "regional_message_bits",
    ]

    rows: list[dict[str, Any]] = []

    for key_values, group in seed_table.groupby(
        keys,
        sort=True,
        dropna=False,
    ):
        row = dict(zip(keys, key_values))
        row["n_subset_seeds"] = int(len(group))
        row["n_examples"] = int(group["n_examples"].iloc[0])
        row["positive_fraction"] = float(group["positive_fraction"].iloc[0])

        for metric_name in MAIN_METRICS:
            values = group[metric_name].to_numpy(dtype=float)
            row[metric_name] = float(np.nanmean(values))
            row[f"{metric_name}_subset_std"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            )

        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_folds(fold_table: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "scheme",
        "quantization_mode",
        "dimension",
        "bits_per_component",
        "task_bits",
        "region_tag_bits",
        "nonce_bits",
        "shared_overhead_bits",
        "message_bits",
        "k",
        "regional_task_bits",
        "regional_message_bits",
    ]

    rows: list[dict[str, Any]] = []

    for key_values, group in fold_table.groupby(
        keys,
        sort=True,
        dropna=False,
    ):
        row = dict(zip(keys, key_values))
        row["n_folds"] = int(len(group))

        for metric_name in MAIN_METRICS:
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.nanmean(values))
            row[f"{metric_name}_std"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric_name}_values"] = json.dumps(values.tolist())
            row[f"{metric_name}_worst"] = (
                float(np.nanmax(values))
                if metric_name in LOWER_IS_BETTER
                else float(np.nanmin(values))
            )

        row["mean_subset_auprc_std"] = float(
            group["auprc_subset_std"].mean()
        )
        row["mean_subset_mdr1_std"] = float(
            group["mdr_at_far_1pct_subset_std"].mean()
        )
        rows.append(row)

    return pd.DataFrame(rows)


def plot_rate_curve(
    aggregate: pd.DataFrame,
    *,
    k: int,
    metric: str,
    ylabel: str,
    output_path: Path,
    x_column: str,
    percentage: bool = False,
) -> None:
    selected = aggregate[aggregate["k"] == int(k)].copy()
    if selected.empty:
        return

    figure, axis = plt.subplots(figsize=(8, 5))

    for scheme, group in selected.groupby("scheme", sort=True):
        group = group.sort_values(x_column)
        x = group[x_column].to_numpy(dtype=float)
        y = group[f"{metric}_mean"].to_numpy(dtype=float)
        std = group[f"{metric}_std"].to_numpy(dtype=float)

        if percentage:
            y = 100.0 * y
            std = 100.0 * std

        axis.errorbar(
            x,
            y,
            yerr=std,
            marker="o",
            capsize=3,
            label=str(scheme),
        )

    axis.set_xlabel(
        "Application message bits/report"
        if x_column == "message_bits"
        else "Task payload bits/report"
    )
    axis.set_ylabel(ylabel)
    axis.set_title(f"Quantized reporting, K={k}")
    axis.set_xscale("log", base=2)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def checkpoint_path(
    run_root: Path,
    *,
    fold: int,
    representation: str,
) -> Path:
    return (
        run_root
        / f"fold_{fold}"
        / representation
        / "deepsets"
        / "final"
        / "final_model.pt"
    )


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    config = load_yaml(args.config)
    study_config = dict(config.get("quantization_study", {}))
    settings = read_settings(study_config)

    if args.learned_bits is not None:
        settings = replace(
            settings,
            learned_bits_per_component=_positive_int_tuple(
                args.learned_bits, "--learned-bits"
            ),
        )
    if args.handcrafted_bits is not None:
        settings = replace(
            settings,
            handcrafted_bits_per_component=_positive_int_tuple(
                args.handcrafted_bits, "--handcrafted-bits"
            ),
        )
    if args.score_bits is not None:
        settings = replace(
            settings,
            score_bits=_positive_int_tuple(args.score_bits, "--score-bits"),
        )
    if args.k_values is not None:
        settings = replace(
            settings,
            k_values=_positive_int_tuple(args.k_values, "--k-values"),
        )

    seed = int(study_config.get("seed", 42))
    n_folds = int(study_config.get("n_folds", 3))

    learned_dataset_root = Path(
        study_config.get(
            "learned_dataset_root",
            "runs/region_dataset_all15_d4",
        )
    )
    score_hc_dataset_root = Path(
        study_config.get(
            "score_handcrafted_dataset_root",
            "runs/region_dataset_all15_d35",
        )
    )
    learned_run_root = Path(
        study_config.get(
            "learned_run_root",
            "runs/region_cv_all15_d4_empirical_fixed_budget",
        )
    )
    handcrafted_run_root = Path(
        study_config.get(
            "handcrafted_run_root",
            "runs/handcrafted_stat_ablation/receiver_runs/nested_h2",
        )
    )
    output_root = (
        args.output_root
        if args.output_root is not None
        else Path(
            study_config.get(
                "output_root",
                "runs/quantization_study_d4_h2",
            )
        )
    )
    device = resolve_device(
        args.device or str(study_config.get("device", "auto"))
    )

    print("TERRA quantization study")
    print("=" * 88)
    print(f"Learned dataset:      {learned_dataset_root}")
    print(f"Score/HC dataset:     {score_hc_dataset_root}")
    print(f"TERRA-Feature run:    {learned_run_root}")
    print(f"Handcrafted H2 run:   {handcrafted_run_root}")
    print(f"Output root:          {output_root}")
    print(f"Device:               {device}")
    print(
        "Element quantizer:    uniform per-element scalar quantization; "
        f"{args.scalar_reconstruction} reconstruction; common clip "
        f"[{settings.clip_lower}, {settings.clip_upper}] "
        "after receiver standardization"
    )
    print(
        f"Message overhead:     tag={settings.region_tag_bits} bits, "
        f"nonce={settings.nonce_bits} bits"
    )
    if settings.region_tag_bits == 0 and settings.nonce_bits == 0:
        print(
            "NOTE: tag/nonce overhead is currently zero. Task-bit results are "
            "valid; set final overhead values later for end-to-end message bits."
        )

    fold_cache: dict[int, tuple[RegionArrays, RegionArrays, np.ndarray]] = {}

    for fold in range(n_folds):
        learned_arrays = load_region_arrays(
            learned_dataset_root
            / f"fold_{fold}"
            / f"region_dataset_fold{fold}.npz",
            learned_dataset_root
            / f"fold_{fold}"
            / f"region_manifest_fold{fold}.csv",
        )
        score_hc_arrays = load_region_arrays(
            score_hc_dataset_root
            / f"fold_{fold}"
            / f"region_dataset_fold{fold}.npz",
            score_hc_dataset_root
            / f"fold_{fold}"
            / f"region_manifest_fold{fold}.csv",
        )
        feature_names = read_handcrafted_feature_names(
            score_hc_dataset_root, fold
        )
        hc_indices = handcrafted_indices(
            feature_names,
            settings.handcrafted_stats,
        )

        assert_aligned(
            learned_arrays,
            score_hc_arrays,
            fold=fold,
        )

        learned_dim = int(learned_arrays.learned.shape[2])
        hc_dim = int(hc_indices.size)

        learned_checkpoint = checkpoint_path(
            learned_run_root,
            fold=fold,
            representation="learned",
        )
        hc_checkpoint = checkpoint_path(
            handcrafted_run_root,
            fold=fold,
            representation="handcrafted",
        )
        if not learned_checkpoint.is_file():
            raise FileNotFoundError(learned_checkpoint)
        if not hc_checkpoint.is_file():
            raise FileNotFoundError(hc_checkpoint)

        if learned_dim != 4:
            print(
                f"WARNING: fold {fold} learned D={learned_dim}, not selected D=4."
            )
        if hc_dim != 30:
            print(
                f"WARNING: fold {fold} handcrafted D={hc_dim}, not selected D=30."
            )

        print(
            f"fold {fold}: OK; examples={len(learned_arrays.target)}, "
            f"TERRA-Feature D={learned_dim}, HC D={hc_dim}, "
            f"stats={settings.handcrafted_stats}"
        )

        fold_cache[fold] = (
            learned_arrays,
            score_hc_arrays,
            hc_indices,
        )

    if args.check_only:
        print("\nValidation completed successfully; no evaluation was run.")
        elapsed = time.perf_counter() - started
        print(f"Total runtime: {elapsed:.2f} seconds")
        return

    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Delete/rename it or pass --output-root."
        )
    output_root.mkdir(parents=True, exist_ok=False)

    set_seed(seed, deterministic=True)

    seed_rows: list[dict[str, Any]] = []
    float_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []

    for fold in range(n_folds):
        print(f"\n=== Fold {fold} ===")
        fold_output = output_root / f"fold_{fold}"
        fold_output.mkdir(parents=True, exist_ok=False)

        learned_arrays, score_hc_arrays, hc_indices = fold_cache[fold]

        development_indices = np.flatnonzero(
            learned_arrays.split_code == 0
        )
        test_indices = np.flatnonzero(
            learned_arrays.split_code == 1
        )
        test_targets = learned_arrays.target[test_indices]
        base_mask = learned_arrays.mask[test_indices]
        shared_seed_base = seed + 10_000_000 * fold

        learned_values = learned_arrays.learned.astype(np.float32)
        handcrafted_values = slice_handcrafted(
            score_hc_arrays, hc_indices
        )
        local_probability = score_hc_arrays.local_probability.astype(
            np.float32
        )

        learned_checkpoint = checkpoint_path(
            learned_run_root,
            fold=fold,
            representation="learned",
        )
        hc_checkpoint = checkpoint_path(
            handcrafted_run_root,
            fold=fold,
            representation="handcrafted",
        )

        learned_model, learned_mean, learned_std, learned_payload = load_model(
            learned_checkpoint,
            representation="learned",
            input_dim=learned_values.shape[2],
            settings=settings,
            device=device,
        )
        hc_model, hc_mean, hc_std, hc_payload = load_model(
            hc_checkpoint,
            representation="handcrafted",
            input_dim=handcrafted_values.shape[2],
            settings=settings,
            device=device,
        )

        learned_standardized = standardized_all_reports(
            learned_values,
            learned_arrays.mask,
            learned_mean,
            learned_std,
        )
        hc_standardized = standardized_all_reports(
            handcrafted_values,
            score_hc_arrays.mask,
            hc_mean,
            hc_std,
        )

        prediction_payload: dict[str, np.ndarray] = {
            "target": test_targets.astype(np.int64),
            "example_index": test_indices.astype(np.int64),
        }

        fold_summary: dict[str, Any] = {
            "fold": fold,
            "learned_checkpoint": str(learned_checkpoint),
            "handcrafted_checkpoint": str(hc_checkpoint),
            "learned_checkpoint_epochs": int(
                learned_payload.get("epochs", -1)
            ),
            "handcrafted_checkpoint_epochs": int(
                hc_payload.get("epochs", -1)
            ),
            "learned_dimension": int(learned_values.shape[2]),
            "handcrafted_dimension": int(handcrafted_values.shape[2]),
            "handcrafted_stats": list(settings.handcrafted_stats),
            "quantizers": [],
        }

        # Float references with exactly the same K/subset masks.
        for k in settings.k_values:
            seeds = evaluation_seeds_for_k(
                base_mask,
                k,
                settings.test_subset_seeds,
                shared_seed_base + 1000 * k,
            )
            for subset_seed in seeds:
                subset_mask = exact_k_mask_numpy(
                    base_mask, k, subset_seed
                )

                learned_float = predict_vector_receiver(
                    learned_model,
                    learned_standardized,
                    indices=test_indices,
                    subset_mask=subset_mask,
                    batch_size=settings.batch_size,
                    device=device,
                )
                hc_float = predict_vector_receiver(
                    hc_model,
                    hc_standardized,
                    indices=test_indices,
                    subset_mask=subset_mask,
                    batch_size=settings.batch_size,
                    device=device,
                )
                score_float = average_scores(
                    local_probability,
                    indices=test_indices,
                    subset_mask=subset_mask,
                )

                for scheme, dimension, probabilities in (
                    ("TERRA-Feature", int(learned_values.shape[2]), learned_float),
                    ("Handcrafted-Stats", int(handcrafted_values.shape[2]), hc_float),
                    ("TERRA-Score", 1, score_float),
                ):
                    float_rows.append(
                        {
                            "fold": fold,
                            "scheme": scheme,
                            "dimension": dimension,
                            "k": int(k),
                            "subset_seed": int(subset_seed),
                            **metrics(test_targets, probabilities),
                        }
                    )

        # TERRA-Score.
        for bits in settings.score_bits:
            quantized_score = quantize_probability(
                local_probability,
                payload_bits=bits,
            )
            print(f"  TERRA-Score: {bits} task bits/report")

            for k in settings.k_values:
                seeds = evaluation_seeds_for_k(
                    base_mask,
                    k,
                    settings.test_subset_seeds,
                    shared_seed_base + 1000 * k,
                )
                for subset_seed in seeds:
                    subset_mask = exact_k_mask_numpy(
                        base_mask, k, subset_seed
                    )
                    probabilities = average_scores(
                        quantized_score,
                        indices=test_indices,
                        subset_mask=subset_mask,
                    )
                    append_condition_row(
                        seed_rows,
                        fold=fold,
                        scheme="TERRA-Score",
                        dimension=1,
                        bits_per_component=bits,
                        task_bits=bits,
                        k=k,
                        subset_seed=subset_seed,
                        metric_values=metrics(
                            test_targets, probabilities
                        ),
                        settings=settings,
                    )
                    if settings.save_predictions:
                        key = (
                            f"terra_score__b{bits}__k{k}__s{subset_seed}"
                        )
                        prediction_payload[key] = probabilities.astype(
                            np.float32
                        )

        # Multi-element report schemes; each element is scalar-quantized independently.
        report_specs = (
            (
                "TERRA-Feature",
                learned_standardized,
                learned_model,
                learned_arrays.mask,
                settings.learned_bits_per_component,
            ),
            (
                "Handcrafted-Stats",
                hc_standardized,
                hc_model,
                score_hc_arrays.mask,
                settings.handcrafted_bits_per_component,
            ),
        )

        for (
            scheme,
            standardized_values,
            model,
            full_mask,
            bit_depths,
        ) in report_specs:
            dimension = int(standardized_values.shape[2])
            valid_dev = standardized_values[development_indices][
                full_mask[development_indices]
            ]

            for bits_per_component in bit_depths:
                quantizer = build_uniform_per_component_quantizer(
                    dimension=dimension,
                    bits_per_component=bits_per_component,
                    lower=settings.clip_lower,
                    upper=settings.clip_upper,
                    reconstruction_mode=args.scalar_reconstruction,
                )
                task_bits = quantizer.total_bits

                diagnostic = quantization_diagnostics(
                    valid_dev,
                    quantizer,
                )
                diagnostic_rows.append(
                    {
                        "fold": fold,
                        "scheme": scheme,
                        **diagnostic,
                    }
                )
                fold_summary["quantizers"].append(
                    {
                        "scheme": scheme,
                        **quantizer.to_dict(),
                        "development_diagnostics": diagnostic,
                    }
                )

                quantized_values = np.zeros_like(
                    standardized_values,
                    dtype=np.float32,
                )
                quantized_values[full_mask] = quantizer.transform(
                    standardized_values[full_mask]
                )

                print(
                    f"  {scheme}: {bits_per_component} bits/component "
                    f"x D={dimension} -> {task_bits} task bits/report; "
                    f"dev clip fraction={diagnostic['clip_fraction_elements']:.4%}"
                )

                for k in settings.k_values:
                    seeds = evaluation_seeds_for_k(
                        base_mask,
                        k,
                        settings.test_subset_seeds,
                        shared_seed_base + 1000 * k,
                    )
                    for subset_seed in seeds:
                        subset_mask = exact_k_mask_numpy(
                            base_mask,
                            k,
                            subset_seed,
                        )
                        probabilities = predict_vector_receiver(
                            model,
                            quantized_values,
                            indices=test_indices,
                            subset_mask=subset_mask,
                            batch_size=settings.batch_size,
                            device=device,
                        )
                        append_condition_row(
                            seed_rows,
                            fold=fold,
                            scheme=scheme,
                            dimension=dimension,
                            bits_per_component=bits_per_component,
                            task_bits=task_bits,
                            k=k,
                            subset_seed=subset_seed,
                            metric_values=metrics(
                                test_targets,
                                probabilities,
                            ),
                            settings=settings,
                        )

                        if settings.save_predictions:
                            slug = (
                                "terra_feature"
                                if scheme == "TERRA-Feature"
                                else "handcrafted_stats"
                            )
                            key = (
                                f"{slug}__bpc{bits_per_component}"
                                f"__k{k}__s{subset_seed}"
                            )
                            prediction_payload[key] = probabilities.astype(
                                np.float32
                            )

        if settings.save_predictions:
            np.savez_compressed(
                fold_output
                / f"quantization_predictions_fold{fold}.npz",
                **prediction_payload,
            )

        save_json(
            fold_summary,
            fold_output / f"quantization_results_fold{fold}.json",
        )
        fold_summaries.append(fold_summary)

    seed_table = pd.DataFrame(seed_rows)
    seed_table.to_csv(
        output_root / "quantization_seed_metrics.csv",
        index=False,
    )

    fold_table = aggregate_subset_seeds(seed_table)
    fold_table.to_csv(
        output_root / "quantization_fold_metrics.csv",
        index=False,
    )

    aggregate_table = aggregate_folds(fold_table)
    aggregate_table.to_csv(
        output_root / "quantization_aggregate_metrics.csv",
        index=False,
    )

    pd.DataFrame(diagnostic_rows).to_csv(
        output_root / "quantization_development_diagnostics.csv",
        index=False,
    )

    float_seed_table = pd.DataFrame(float_rows)
    float_seed_table.to_csv(
        output_root / "quantization_float_reference_seed_metrics.csv",
        index=False,
    )

    float_fold_rows: list[dict[str, Any]] = []
    for (fold, scheme, k), group in float_seed_table.groupby(
        ["fold", "scheme", "k"],
        sort=True,
    ):
        row: dict[str, Any] = {
            "fold": int(fold),
            "scheme": scheme,
            "k": int(k),
            "n_subset_seeds": int(len(group)),
        }
        for metric_name in MAIN_METRICS:
            row[metric_name] = float(
                np.nanmean(group[metric_name].to_numpy(dtype=float))
            )
        float_fold_rows.append(row)

    float_fold = pd.DataFrame(float_fold_rows)
    float_aggregate_rows: list[dict[str, Any]] = []

    for (scheme, k), group in float_fold.groupby(
        ["scheme", "k"], sort=True
    ):
        row: dict[str, Any] = {
            "scheme": scheme,
            "k": int(k),
            "n_folds": int(len(group)),
        }
        for metric_name in MAIN_METRICS:
            values = group[metric_name].to_numpy(dtype=float)
            row[f"{metric_name}_mean"] = float(np.nanmean(values))
            row[f"{metric_name}_std"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            )
            row[f"{metric_name}_values"] = json.dumps(values.tolist())
        float_aggregate_rows.append(row)

    float_aggregate = pd.DataFrame(float_aggregate_rows)
    float_aggregate.to_csv(
        output_root / "quantization_float_reference_metrics.csv",
        index=False,
    )

    x_column = (
        "message_bits"
        if settings.region_tag_bits + settings.nonce_bits > 0
        else "task_bits"
    )

    for k in (1, 3, 5, 9):
        if k not in settings.k_values:
            continue

        plot_rate_curve(
            aggregate_table,
            k=k,
            metric="auprc",
            ylabel="AUPRC",
            output_path=output_root / f"quantization_auprc_k{k}.png",
            x_column=x_column,
        )
        plot_rate_curve(
            aggregate_table,
            k=k,
            metric="mdr_at_far_1pct",
            ylabel="MDR at FAR <= 1% (%)",
            output_path=output_root / f"quantization_mdr1_k{k}.png",
            x_column=x_column,
            percentage=True,
        )

    summary = {
        "configuration": {
            "learned_dataset_root": str(learned_dataset_root),
            "score_handcrafted_dataset_root": str(score_hc_dataset_root),
            "learned_run_root": str(learned_run_root),
            "handcrafted_run_root": str(handcrafted_run_root),
            "handcrafted_stats": list(settings.handcrafted_stats),
            "learned_bits_per_component": list(
                settings.learned_bits_per_component
            ),
            "handcrafted_bits_per_component": list(
                settings.handcrafted_bits_per_component
            ),
            "score_bits": list(settings.score_bits),
            "k_values": list(settings.k_values),
            "test_subset_seeds": settings.test_subset_seeds,
            "clip_lower": settings.clip_lower,
            "clip_upper": settings.clip_upper,
            "scalar_reconstruction_mode": args.scalar_reconstruction,
            "region_tag_bits": settings.region_tag_bits,
            "nonce_bits": settings.nonce_bits,
        },
        "methodology": {
            "TERRA-Score": (
                "The local fire probability produced by the selected sensor-side model "
                "is uniformly quantized on [0,1] and averaged across exact-K recovered reports."
            ),
            "TERRA-Feature": (
                "The selected D=4 learned representation is standardized using "
                "the frozen regional receiver's development statistics, clipped "
                "to one common fixed interval, and uniformly scalar-quantized "
                "independently per element with the same bit depth on every component."
            ),
            "Handcrafted-Stats": (
                "The selected mean+std handcrafted benchmark is sliced from the "
                "full handcrafted report, standardized with its frozen regional "
                "receiver statistics, and independently scalar-quantized per element "
                "with the identical rule."
            ),
            "receiver": (
                "All regional receivers are frozen; no quantization-aware training "
                "or receiver fine-tuning is performed."
            ),
            "aggregation": (
                "Metrics are computed per subset realization, averaged within each "
                "outer fold, then averaged equally across the three folds."
            ),
            "operational_metrics": (
                "MDR@FAR<=1% and MDR@FAR<=5% are ROC performance metrics; "
                "no test-derived deployment threshold is exported."
            ),
            "bit_accounting": (
                "task_bits count only the quantized task payload. message_bits add "
                "region-tag and nonce overhead. regional_message_bits equal K times "
                "message_bits for the exact-K condition."
            ),
            "ura_note": (
                "This study assumes selected reports are recovered correctly. URA "
                "missed detections, false alarms, and payload/tag errors are handled "
                "in the subsequent channel-impairment study."
            ),
        },
        "folds": fold_summaries,
    }
    save_json(
        summary,
        output_root / "quantization_summary.json",
    )

    print("\n=== Quantized aggregate, K=9 ===")
    selected = aggregate_table[
        aggregate_table["k"] == 9
    ][
        [
            "scheme",
            "dimension",
            "bits_per_component",
            "task_bits",
            "message_bits",
            "auprc_mean",
            "auroc_mean",
            "mdr_at_far_1pct_mean",
            "mdr_at_far_5pct_mean",
        ]
    ].sort_values(["scheme", "task_bits"])
    print(selected.to_string(index=False))

    print("\n=== Float references, K=9 ===")
    print(
        float_aggregate[
            float_aggregate["k"] == 9
        ].to_string(index=False)
    )

    print(f"\nSaved quantization study to: {output_root}")

    elapsed = time.perf_counter() - started
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(
        "\nTotal runtime: "
        f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
        f"({elapsed:.2f} seconds)"
    )


if __name__ == "__main__":
    main()
