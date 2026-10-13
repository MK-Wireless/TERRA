#!/usr/bin/env python
from __future__ import annotations

"""
Run the TERRA synthetic URA MD/FA impairment study.

Locked application operating points
------------------------------------
TERRA-Score:
    scalar local fire probability, 6 task bits/report, endpoint reconstruction
    on [0,1], regional receiver = arithmetic mean.

TERRA-Feature:
    selected D=4 learned representation, 6 bits/element -> 24 task bits/report,
    receiver-standardized, fixed clip [-4,4], mid-rise reconstruction,
    regional receiver = frozen DeepSets.

Handcrafted-Stats:
    selected mean+std representation over 15 channels, D=30,
    1 bit/element -> 30 task bits/report, receiver-standardized,
    fixed clip [-4,4], mid-rise reconstruction, regional receiver = frozen
    DeepSets.

Synthetic URA abstraction
-------------------------
Only list-level missed detections (MD) and false alarms (FA) are modeled.

* There is NO separate payload-bit-error process.
* P_MD and P_FA are GLOBAL URA list-recovery quantities.
* FA messages receive uniformly random VALID region tags.
* FA task payloads are uniformly random valid quantized task words.
* A false task payload may equal a genuine task payload. Full-message
  uniqueness is guaranteed by a synthetic nonce/message-ID namespace.
* Physical nonce LENGTH does not affect the imposed synthetic MD/FA rates.
* The number of valid regions DOES matter because random FA tags determine
  which regional receiver a false message contaminates.
* Other regions need not be materialized by the dataset. Their true messages
  are represented through the configured activity count when determining the
  global decoded-list size and global FA count.

For a requested P_FA, the default count construction uses ratio-matched
randomized rounding so that, conditional on the number T of correctly
recovered global true messages,

    E[F / (T+F) | T] = P_FA.

This matches the list-level FA definition directly.

Outputs
-------
<output_root>/
    ura_impairment_seed_metrics.csv
    ura_impairment_fold_metrics.csv
    ura_impairment_aggregate_metrics.csv
    ura_impairment_summary.json
    heatmap_<scheme>_<metric>.png

The primary paper metrics are AUPRC, AUROC, MDR@FAR<=1%, and MDR@FAR<=5%.
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
    quantize_probability,
)
from fire_ura.region_data import RegionArrays, load_region_arrays
from fire_ura.region_model import build_region_model
from fire_ura.ura_impairments import (
    RecoveryPlan,
    sample_recovery_plan,
    sample_uniform_task_payload,
)
from fire_ura.utils import load_yaml, set_seed


SCHEMES = (
    "TERRA-Score",
    "TERRA-Feature",
    "Handcrafted-Stats",
)

METRIC_COLUMNS = (
    "auprc",
    "auroc",
    "mdr_at_far_1pct",
    "achieved_far_at_1pct",
    "mdr_at_far_5pct",
    "achieved_far_at_5pct",
)


@dataclass(frozen=True)
class Settings:
    learned_dataset_root: Path
    learned_run_root: Path
    score_handcrafted_dataset_root: Path
    handcrafted_run_root: Path
    handcrafted_stats: tuple[str, ...]

    feature_bits_per_component: int
    score_bits: int
    handcrafted_bits_per_component: int
    clip_lower: float
    clip_upper: float
    vector_reconstruction_mode: str

    active_users_per_region: int
    num_regions: int | None
    p_md_values: tuple[float, ...]
    p_fa_values: tuple[float, ...]
    monte_carlo_seeds: int

    # Reporting only. This value does NOT affect synthetic error generation.
    nonce_bits: int | None

    empty_region_probability: float
    batch_size: int
    hidden_dim: int
    phi_hidden_dim: int
    phi_output_dim: int
    rho_hidden_dim: int
    dropout: float

    seed: int
    n_folds: int
    device: str
    output_root: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TERRA synthetic URA MD/FA impairment study."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
    )
    parser.add_argument(
        "--num-regions",
        type=int,
        default=None,
        help=(
            "Override the number of valid region tags. Required for a full run "
            "if num_regions is left null in the YAML."
        ),
    )
    parser.add_argument(
        "--p-md",
        type=float,
        nargs="+",
        default=None,
        help="Override P_MD grid.",
    )
    parser.add_argument(
        "--p-fa",
        type=float,
        nargs="+",
        default=None,
        help="Override P_FA grid.",
    )
    parser.add_argument(
        "--mc-seeds",
        type=int,
        default=None,
        help="Override Monte Carlo realizations per (P_MD,P_FA) point.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate datasets/checkpoints/configuration without evaluation.",
    )
    return parser.parse_args()


def _float_tuple(
    values: Sequence[Any],
    name: str,
    *,
    allow_one: bool,
) -> tuple[float, ...]:
    parsed = tuple(sorted({float(v) for v in values}))
    if not parsed:
        raise ValueError(f"{name} cannot be empty.")
    for value in parsed:
        upper_ok = value <= 1.0 if allow_one else value < 1.0
        if not math.isfinite(value) or value < 0.0 or not upper_ok:
            interval = "[0,1]" if allow_one else "[0,1)"
            raise ValueError(f"{name} values must lie in {interval}.")
    return parsed


def read_settings(config: Mapping[str, Any]) -> Settings:
    num_regions_raw = config.get("num_regions", None)
    num_regions = (
        None if num_regions_raw is None else int(num_regions_raw)
    )
    nonce_raw = config.get("nonce_bits", None)
    nonce_bits = None if nonce_raw is None else int(nonce_raw)

    settings = Settings(
        learned_dataset_root=Path(
            config.get("learned_dataset_root", "runs/region_dataset_all15_d4")
        ),
        learned_run_root=Path(
            config.get(
                "learned_run_root",
                "runs/region_cv_all15_d4_empirical_fixed_budget",
            )
        ),
        score_handcrafted_dataset_root=Path(
            config.get(
                "score_handcrafted_dataset_root",
                "runs/region_dataset_all15_d35",
            )
        ),
        handcrafted_run_root=Path(
            config.get(
                "handcrafted_run_root",
                "runs/handcrafted_stat_ablation/receiver_runs/nested_h2",
            )
        ),
        handcrafted_stats=tuple(
            str(v) for v in config.get("handcrafted_stats", ["mean", "std"])
        ),
        feature_bits_per_component=int(
            config.get("feature_bits_per_component", 6)
        ),
        score_bits=int(config.get("score_bits", 6)),
        handcrafted_bits_per_component=int(
            config.get("handcrafted_bits_per_component", 1)
        ),
        clip_lower=float(config.get("clip_lower", -4.0)),
        clip_upper=float(config.get("clip_upper", 4.0)),
        vector_reconstruction_mode=str(
            config.get("vector_reconstruction_mode", "midrise")
        ),
        active_users_per_region=int(
            config.get("active_users_per_region", 9)
        ),
        num_regions=num_regions,
        p_md_values=_float_tuple(
            config.get(
                "p_md_values",
                [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50],
            ),
            "p_md_values",
            allow_one=True,
        ),
        p_fa_values=_float_tuple(
            config.get(
                "p_fa_values",
                [0.0, 0.01, 0.05, 0.10, 0.20, 0.30],
            ),
            "p_fa_values",
            allow_one=False,
        ),
        monte_carlo_seeds=int(config.get("monte_carlo_seeds", 10)),
        nonce_bits=nonce_bits,
        empty_region_probability=float(
            config.get("empty_region_probability", 0.0)
        ),
        batch_size=int(config.get("batch_size", 512)),
        hidden_dim=int(config.get("hidden_dim", 32)),
        phi_hidden_dim=int(config.get("phi_hidden_dim", 32)),
        phi_output_dim=int(config.get("phi_output_dim", 16)),
        rho_hidden_dim=int(config.get("rho_hidden_dim", 16)),
        dropout=float(config.get("dropout", 0.15)),
        seed=int(config.get("seed", 42)),
        n_folds=int(config.get("n_folds", 3)),
        device=str(config.get("device", "auto")),
        output_root=Path(
            config.get(
                "output_root",
                "runs/ura_impairment_study_d4_h2",
            )
        ),
    )

    if settings.feature_bits_per_component <= 0:
        raise ValueError("feature_bits_per_component must be positive.")
    if settings.score_bits <= 0:
        raise ValueError("score_bits must be positive.")
    if settings.handcrafted_bits_per_component <= 0:
        raise ValueError("handcrafted_bits_per_component must be positive.")
    if settings.clip_upper <= settings.clip_lower:
        raise ValueError("clip_upper must exceed clip_lower.")
    if settings.vector_reconstruction_mode not in {"endpoints", "midrise"}:
        raise ValueError(
            "vector_reconstruction_mode must be endpoints or midrise."
        )
    if settings.active_users_per_region <= 0:
        raise ValueError("active_users_per_region must be positive.")
    if settings.num_regions is not None and settings.num_regions <= 0:
        raise ValueError("num_regions must be positive when specified.")
    if settings.monte_carlo_seeds <= 0:
        raise ValueError("monte_carlo_seeds must be positive.")
    if settings.nonce_bits is not None and settings.nonce_bits < 0:
        raise ValueError("nonce_bits must be nonnegative when specified.")
    if not 0.0 <= settings.empty_region_probability <= 1.0:
        raise ValueError("empty_region_probability must lie in [0,1].")
    if settings.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if settings.n_folds <= 0:
        raise ValueError("n_folds must be positive.")
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


def read_handcrafted_feature_names(dataset_root: Path, fold: int) -> list[str]:
    path = (
        dataset_root
        / f"fold_{fold}"
        / f"region_metadata_fold{fold}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    names = payload.get("handcrafted_feature_names")
    if not isinstance(names, list) or not names:
        raise KeyError(
            f"{path} does not contain handcrafted_feature_names."
        )
    return [str(v) for v in names]


def handcrafted_indices(
    feature_names: Sequence[str],
    requested_stats: Sequence[str],
) -> np.ndarray:
    wanted = {str(v) for v in requested_stats}
    indices: list[int] = []
    for index, name in enumerate(feature_names):
        if "__" not in name:
            raise ValueError(
                f"Unexpected handcrafted feature name {name!r}."
            )
        if name.rsplit("__", 1)[1] in wanted:
            indices.append(index)

    found = {
        feature_names[i].rsplit("__", 1)[1]
        for i in indices
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
        raise ValueError(f"Fold {fold}: example counts differ.")
    if not np.array_equal(learned_arrays.target, score_hc_arrays.target):
        raise ValueError(f"Fold {fold}: targets differ.")
    if not np.array_equal(
        learned_arrays.split_code,
        score_hc_arrays.split_code,
    ):
        raise ValueError(f"Fold {fold}: split codes differ.")
    if not np.array_equal(learned_arrays.mask, score_hc_arrays.mask):
        raise ValueError(f"Fold {fold}: report masks differ.")

    for column in ("episode_id", "timestamp", "region_id"):
        if (
            column in learned_arrays.metadata.columns
            and column in score_hc_arrays.metadata.columns
        ):
            left = learned_arrays.metadata[column].astype(str).to_numpy()
            right = score_hc_arrays.metadata[column].astype(str).to_numpy()
            if not np.array_equal(left, right):
                raise ValueError(
                    f"Fold {fold}: metadata column {column!r} differs."
                )


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
    checkpoint: Path,
    *,
    representation: str,
    input_dim: int,
    settings: Settings,
    device: torch.device,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    payload = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=False,
    )
    if str(payload.get("representation")) != representation:
        raise ValueError(
            f"{checkpoint}: expected representation={representation!r}, "
            f"found {payload.get('representation')!r}."
        )
    architecture = str(payload.get("architecture"))
    if architecture != "deepsets":
        raise ValueError(
            f"{checkpoint}: expected DeepSets, found {architecture!r}."
        )

    feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
    if feature_mean.shape != (input_dim,) or feature_std.shape != (input_dim,):
        raise ValueError(
            f"{checkpoint}: standardizer shape mismatch for D={input_dim}."
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


def standardize_all(
    values: np.ndarray,
    mask: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    output = np.zeros_like(values, dtype=np.float32)
    output[mask] = (
        values[mask].astype(np.float32) - mean
    ) / std
    return output


@torch.no_grad()
def predict_variable_deepsets(
    model: nn.Module,
    report_lists: Sequence[np.ndarray],
    *,
    input_dim: int,
    batch_size: int,
    device: torch.device,
    empty_region_probability: float,
) -> np.ndarray:
    """Run a frozen DeepSets receiver on variable-cardinality recovered lists."""
    output = np.full(
        len(report_lists),
        float(empty_region_probability),
        dtype=np.float64,
    )

    nonempty = np.asarray(
        [i for i, values in enumerate(report_lists) if len(values) > 0],
        dtype=np.int64,
    )
    if nonempty.size == 0:
        return output

    model.eval()
    for start in range(0, len(nonempty), batch_size):
        batch_indices = nonempty[start : start + batch_size]
        max_reports = max(len(report_lists[int(i)]) for i in batch_indices)

        x_np = np.zeros(
            (len(batch_indices), max_reports, input_dim),
            dtype=np.float32,
        )
        mask_np = np.zeros(
            (len(batch_indices), max_reports),
            dtype=bool,
        )

        for row, example_index in enumerate(batch_indices):
            values = np.asarray(
                report_lists[int(example_index)],
                dtype=np.float32,
            )
            if values.ndim != 2 or values.shape[1] != input_dim:
                raise ValueError(
                    f"Expected report list [L,{input_dim}], found {values.shape}."
                )
            count = len(values)
            x_np[row, :count] = values
            mask_np[row, :count] = True

        x = torch.from_numpy(x_np).to(
            device,
            dtype=torch.float32,
        )
        mask = torch.from_numpy(mask_np).to(
            device,
            dtype=torch.bool,
        )
        probabilities = torch.sigmoid(model(x, mask)).cpu().numpy()
        output[batch_indices] = probabilities.astype(np.float64)

    return output


def average_variable_scores(
    report_lists: Sequence[np.ndarray],
    *,
    empty_region_probability: float,
) -> np.ndarray:
    output = np.full(
        len(report_lists),
        float(empty_region_probability),
        dtype=np.float64,
    )
    for index, values in enumerate(report_lists):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        if values.size > 0:
            output[index] = float(np.mean(values))
    return output


def _condition_seed(
    *,
    base_seed: int,
    fold: int,
    md_index: int,
    fa_index: int,
    realization: int,
) -> int:
    return int(
        base_seed
        + 10_000_000 * fold
        + 100_000 * md_index
        + 1_000 * fa_index
        + realization
    )


def build_corrupted_lists(
    *,
    base_mask: np.ndarray,
    score_values: np.ndarray,
    feature_values: np.ndarray,
    handcrafted_values: np.ndarray,
    num_regions: int,
    active_users_per_region: int,
    p_md: float,
    p_fa: float,
    seed: int,
    score_bits: int,
    feature_bits_per_component: int,
    handcrafted_bits_per_component: int,
    clip_lower: float,
    clip_upper: float,
    vector_reconstruction_mode: str,
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    dict[str, float],
]:
    """Apply one paired global MD/FA realization to all three schemes."""
    plan_rng = np.random.default_rng(seed)
    score_rng = np.random.default_rng(seed + 100_000_003)
    feature_rng = np.random.default_rng(seed + 200_000_033)
    handcrafted_rng = np.random.default_rng(seed + 300_000_077)

    score_lists: list[np.ndarray] = []
    feature_lists: list[np.ndarray] = []
    handcrafted_lists: list[np.ndarray] = []

    realized_global_md: list[float] = []
    realized_global_fa: list[float] = []
    realized_target_md: list[float] = []
    target_true_recovered: list[int] = []
    target_fa_count: list[int] = []
    target_decoded_count: list[int] = []

    for example_index, row_mask in enumerate(base_mask):
        valid_slots = np.flatnonzero(row_mask)
        if len(valid_slots) != active_users_per_region:
            raise ValueError(
                f"Example {example_index}: found {len(valid_slots)} valid reports, "
                f"but active_users_per_region={active_users_per_region}. "
                "Use the full-report regional dataset/condition or update the "
                "configuration deliberately."
            )

        plan: RecoveryPlan = sample_recovery_plan(
            target_true_count=len(valid_slots),
            active_users_per_other_region=active_users_per_region,
            num_regions=num_regions,
            p_md=p_md,
            p_fa=p_fa,
            rng=plan_rng,
        )

        kept_slots = valid_slots[plan.target_keep_mask]

        kept_score = np.asarray(
            score_values[example_index, kept_slots],
            dtype=np.float32,
        ).reshape(-1)
        kept_feature = np.asarray(
            feature_values[example_index, kept_slots],
            dtype=np.float32,
        )
        kept_hc = np.asarray(
            handcrafted_values[example_index, kept_slots],
            dtype=np.float32,
        )

        n_fa = int(plan.target_fa_count)

        # Uniform random valid task words. Task-payload equality with a genuine
        # report is permitted because the FA full message has a distinct
        # synthetic nonce/message ID.
        fa_score = sample_uniform_task_payload(
            n_messages=n_fa,
            dimension=1,
            bits_per_component=score_bits,
            lower=0.0,
            upper=1.0,
            reconstruction_mode="endpoints",
            rng=score_rng,
        ).reshape(-1)

        fa_feature = sample_uniform_task_payload(
            n_messages=n_fa,
            dimension=feature_values.shape[2],
            bits_per_component=feature_bits_per_component,
            lower=clip_lower,
            upper=clip_upper,
            reconstruction_mode=vector_reconstruction_mode,
            rng=feature_rng,
        )

        fa_hc = sample_uniform_task_payload(
            n_messages=n_fa,
            dimension=handcrafted_values.shape[2],
            bits_per_component=handcrafted_bits_per_component,
            lower=clip_lower,
            upper=clip_upper,
            reconstruction_mode=vector_reconstruction_mode,
            rng=handcrafted_rng,
        )

        score_lists.append(
            np.concatenate([kept_score, fa_score]).astype(np.float32)
        )
        feature_lists.append(
            np.concatenate([kept_feature, fa_feature], axis=0).astype(np.float32)
        )
        handcrafted_lists.append(
            np.concatenate([kept_hc, fa_hc], axis=0).astype(np.float32)
        )

        realized_global_md.append(plan.realized_global_md)
        realized_global_fa.append(plan.realized_global_fa)
        realized_target_md.append(plan.realized_target_md)
        target_true_recovered.append(plan.target_true_recovered_count)
        target_fa_count.append(plan.target_fa_count)
        target_decoded_count.append(plan.target_decoded_count)

    diagnostics = {
        "realized_global_md": float(np.mean(realized_global_md)),
        "realized_global_fa": float(np.mean(realized_global_fa)),
        "realized_target_md": float(np.mean(realized_target_md)),
        "mean_target_true_recovered": float(np.mean(target_true_recovered)),
        "mean_target_fa_count": float(np.mean(target_fa_count)),
        "mean_target_decoded_count": float(np.mean(target_decoded_count)),
        "empty_target_fraction": float(
            np.mean(np.asarray(target_decoded_count) == 0)
        ),
    }
    return score_lists, feature_lists, handcrafted_lists, diagnostics


def aggregate_seed_metrics(seed_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouping = ["fold", "scheme", "p_md", "p_fa"]
    numeric = [
        *METRIC_COLUMNS,
        "realized_global_md",
        "realized_global_fa",
        "realized_target_md",
        "mean_target_true_recovered",
        "mean_target_fa_count",
        "mean_target_decoded_count",
        "empty_target_fraction",
    ]

    fold_rows: list[dict[str, Any]] = []
    for keys, group in seed_df.groupby(grouping, sort=True):
        row = dict(zip(grouping, keys))
        row["n_realizations"] = int(len(group))
        for column in numeric:
            row[column] = float(group[column].mean())
        fold_rows.append(row)

    fold_df = pd.DataFrame(fold_rows)

    aggregate_rows: list[dict[str, Any]] = []
    for keys, group in fold_df.groupby(
        ["scheme", "p_md", "p_fa"],
        sort=True,
    ):
        row = {
            "scheme": keys[0],
            "p_md": float(keys[1]),
            "p_fa": float(keys[2]),
            "n_folds": int(len(group)),
        }
        for column in numeric:
            values = group[column].to_numpy(dtype=float)
            row[f"{column}_mean"] = float(np.nanmean(values))
            row[f"{column}_std"] = (
                float(np.nanstd(values, ddof=1))
                if len(values) > 1
                else 0.0
            )
        aggregate_rows.append(row)

    return fold_df, pd.DataFrame(aggregate_rows)


def plot_heatmaps(
    aggregate_df: pd.DataFrame,
    *,
    output_root: Path,
) -> None:
    metrics_to_plot = (
        "auprc_mean",
        "mdr_at_far_1pct_mean",
    )

    for scheme in SCHEMES:
        scheme_df = aggregate_df[aggregate_df["scheme"] == scheme]
        for metric in metrics_to_plot:
            pivot = scheme_df.pivot(
                index="p_md",
                columns="p_fa",
                values=metric,
            ).sort_index().sort_index(axis=1)

            matrix = pivot.to_numpy(dtype=float)
            fig, ax = plt.subplots(figsize=(7.2, 5.2))
            image = ax.imshow(
                matrix,
                aspect="auto",
                origin="lower",
            )
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels(
                [f"{float(v):.2f}" for v in pivot.columns]
            )
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels(
                [f"{float(v):.2f}" for v in pivot.index]
            )
            ax.set_xlabel(r"$P_{\rm FA}$")
            ax.set_ylabel(r"$P_{\rm MD}$")
            ax.set_title(f"{scheme}: {metric.replace('_mean', '')}")

            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    value = matrix[row, col]
                    if np.isfinite(value):
                        ax.text(
                            col,
                            row,
                            f"{value:.3f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                        )

            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            slug = scheme.lower().replace("-", "_")
            fig.savefig(
                output_root / f"heatmap_{slug}_{metric}.png",
                dpi=220,
                bbox_inches="tight",
            )
            plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    config = load_yaml(args.config)
    study_config = config.get("ura_impairment_study", config)
    settings = read_settings(study_config)

    if args.output_root is not None:
        settings = replace(settings, output_root=args.output_root)
    if args.device is not None:
        settings = replace(settings, device=args.device)
    if args.num_regions is not None:
        settings = replace(settings, num_regions=int(args.num_regions))
    if args.p_md is not None:
        settings = replace(
            settings,
            p_md_values=_float_tuple(
                args.p_md,
                "--p-md",
                allow_one=True,
            ),
        )
    if args.p_fa is not None:
        settings = replace(
            settings,
            p_fa_values=_float_tuple(
                args.p_fa,
                "--p-fa",
                allow_one=False,
            ),
        )
    if args.mc_seeds is not None:
        if int(args.mc_seeds) <= 0:
            raise ValueError("--mc-seeds must be positive.")
        settings = replace(
            settings,
            monte_carlo_seeds=int(args.mc_seeds),
        )

    device = resolve_device(settings.device)

    print("TERRA synthetic URA MD/FA impairment study")
    print("=" * 88)
    print(f"Device:                    {device}")
    print(f"TERRA-Score:               {settings.score_bits} task bits")
    print(
        "TERRA-Feature:             "
        f"D=4 x {settings.feature_bits_per_component} bits = "
        f"{4 * settings.feature_bits_per_component} task bits"
    )
    print(
        "Handcrafted-Stats:         "
        f"D=30 x {settings.handcrafted_bits_per_component} bits = "
        f"{30 * settings.handcrafted_bits_per_component} task bits"
    )
    print(f"P_MD grid:                 {settings.p_md_values}")
    print(f"P_FA grid:                 {settings.p_fa_values}")
    print(f"Monte Carlo seeds:         {settings.monte_carlo_seeds}")
    print(f"Active users/region:       {settings.active_users_per_region}")
    print(f"Valid regions:             {settings.num_regions}")
    print(
        "FA tags:                   uniform over valid region tags "
        "(global P_FA before grouping)"
    )
    print(
        "FA task payloads:          uniform valid quantized task words"
    )
    print(
        "Payload-bit-error axis:    none; wrong decodes are represented by MD+FA"
    )
    print(
        "Nonce in emulator:         synthetic unique IDs only; physical length "
        "does not affect imposed MD/FA"
    )
    print(
        f"Empty target region output: p(fire)="
        f"{settings.empty_region_probability:.3f}"
    )

    learned_root = settings.learned_dataset_root
    score_hc_root = settings.score_handcrafted_dataset_root

    fold_cache: dict[int, tuple[RegionArrays, RegionArrays, np.ndarray]] = {}

    for fold in range(settings.n_folds):
        learned_arrays = load_region_arrays(
            learned_root
            / f"fold_{fold}"
            / f"region_dataset_fold{fold}.npz",
            learned_root
            / f"fold_{fold}"
            / f"region_manifest_fold{fold}.csv",
        )
        score_hc_arrays = load_region_arrays(
            score_hc_root
            / f"fold_{fold}"
            / f"region_dataset_fold{fold}.npz",
            score_hc_root
            / f"fold_{fold}"
            / f"region_manifest_fold{fold}.csv",
        )
        feature_names = read_handcrafted_feature_names(
            score_hc_root,
            fold,
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

        if learned_arrays.learned.shape[2] != 4:
            raise ValueError(
                f"Fold {fold}: expected locked TERRA-Feature D=4, found "
                f"D={learned_arrays.learned.shape[2]}."
            )
        if len(hc_indices) != 30:
            raise ValueError(
                f"Fold {fold}: expected locked handcrafted D=30, found "
                f"D={len(hc_indices)}."
            )

        test_indices = np.flatnonzero(learned_arrays.split_code == 1)
        report_counts = learned_arrays.mask[test_indices].sum(axis=1)
        if not np.all(report_counts == settings.active_users_per_region):
            unique = np.unique(report_counts).tolist()
            raise ValueError(
                f"Fold {fold}: test report counts {unique} do not all equal "
                f"active_users_per_region={settings.active_users_per_region}."
            )

        for checkpoint in (
            checkpoint_path(
                settings.learned_run_root,
                fold=fold,
                representation="learned",
            ),
            checkpoint_path(
                settings.handcrafted_run_root,
                fold=fold,
                representation="handcrafted",
            ),
        ):
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)

        fold_cache[fold] = (
            learned_arrays,
            score_hc_arrays,
            hc_indices,
        )
        print(
            f"fold {fold}: OK; test examples={len(test_indices)}, "
            f"reports/example={settings.active_users_per_region}"
        )

    if args.check_only:
        if settings.num_regions is None:
            print(
                "\nNOTE: num_regions is still null. Dataset/checkpoint validation "
                "passed, but a full run requires --num-regions N or a YAML value."
            )
        else:
            tag_bits = int(math.ceil(math.log2(settings.num_regions))) \
                if settings.num_regions > 1 else 0
            print(
                f"\nConfiguration validation passed. "
                f"{settings.num_regions} valid regions require at least "
                f"{tag_bits} tag bits."
            )
        print(f"Total runtime: {time.perf_counter() - started:.2f} seconds")
        return

    if settings.num_regions is None:
        raise ValueError(
            "num_regions is required for the synthetic FA-tag model. "
            "Set it in the YAML or pass --num-regions N."
        )

    num_regions = int(settings.num_regions)
    tag_bits = int(math.ceil(math.log2(num_regions))) if num_regions > 1 else 0

    output_root = settings.output_root
    if output_root.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_root}\n"
            "Delete/rename it or pass --output-root."
        )
    output_root.mkdir(parents=True, exist_ok=False)

    set_seed(settings.seed, deterministic=True)

    seed_rows: list[dict[str, Any]] = []
    fold_metadata: list[dict[str, Any]] = []

    for fold in range(settings.n_folds):
        print(f"\n=== Fold {fold} ===")
        learned_arrays, score_hc_arrays, hc_indices = fold_cache[fold]
        test_indices = np.flatnonzero(learned_arrays.split_code == 1)
        test_targets = learned_arrays.target[test_indices].astype(np.int64)
        base_mask = learned_arrays.mask[test_indices].astype(bool)

        learned_values = learned_arrays.learned.astype(np.float32)
        hc_values = slice_handcrafted(score_hc_arrays, hc_indices)
        local_probability = score_hc_arrays.local_probability.astype(np.float32)

        learned_checkpoint = checkpoint_path(
            settings.learned_run_root,
            fold=fold,
            representation="learned",
        )
        hc_checkpoint = checkpoint_path(
            settings.handcrafted_run_root,
            fold=fold,
            representation="handcrafted",
        )

        learned_model, learned_mean, learned_std, learned_payload = load_model(
            learned_checkpoint,
            representation="learned",
            input_dim=4,
            settings=settings,
            device=device,
        )
        hc_model, hc_mean, hc_std, hc_payload = load_model(
            hc_checkpoint,
            representation="handcrafted",
            input_dim=30,
            settings=settings,
            device=device,
        )

        learned_standardized = standardize_all(
            learned_values,
            learned_arrays.mask,
            learned_mean,
            learned_std,
        )
        hc_standardized = standardize_all(
            hc_values,
            score_hc_arrays.mask,
            hc_mean,
            hc_std,
        )

        feature_quantizer = build_uniform_per_component_quantizer(
            dimension=4,
            bits_per_component=settings.feature_bits_per_component,
            lower=settings.clip_lower,
            upper=settings.clip_upper,
            reconstruction_mode=settings.vector_reconstruction_mode,
        )
        hc_quantizer = build_uniform_per_component_quantizer(
            dimension=30,
            bits_per_component=settings.handcrafted_bits_per_component,
            lower=settings.clip_lower,
            upper=settings.clip_upper,
            reconstruction_mode=settings.vector_reconstruction_mode,
        )

        quantized_feature = np.zeros_like(
            learned_standardized,
            dtype=np.float32,
        )
        quantized_feature[learned_arrays.mask] = feature_quantizer.transform(
            learned_standardized[learned_arrays.mask]
        )

        quantized_hc = np.zeros_like(
            hc_standardized,
            dtype=np.float32,
        )
        quantized_hc[score_hc_arrays.mask] = hc_quantizer.transform(
            hc_standardized[score_hc_arrays.mask]
        )

        quantized_score = quantize_probability(
            local_probability,
            payload_bits=settings.score_bits,
        )

        fold_metadata.append(
            {
                "fold": fold,
                "learned_checkpoint": str(learned_checkpoint),
                "handcrafted_checkpoint": str(hc_checkpoint),
                "learned_checkpoint_epochs": int(
                    learned_payload.get("epochs", -1)
                ),
                "handcrafted_checkpoint_epochs": int(
                    hc_payload.get("epochs", -1)
                ),
                "n_test_examples": int(len(test_indices)),
            }
        )

        # Restrict arrays to test examples once, so build_corrupted_lists uses
        # local example indices 0..N_test-1.
        test_score = quantized_score[test_indices]
        test_feature = quantized_feature[test_indices]
        test_hc = quantized_hc[test_indices]

        for md_index, p_md in enumerate(settings.p_md_values):
            for fa_index, p_fa in enumerate(settings.p_fa_values):
                print(
                    f"  P_MD={p_md:.3f}, P_FA={p_fa:.3f}",
                    end="",
                    flush=True,
                )

                for realization in range(settings.monte_carlo_seeds):
                    condition_seed = _condition_seed(
                        base_seed=settings.seed,
                        fold=fold,
                        md_index=md_index,
                        fa_index=fa_index,
                        realization=realization,
                    )

                    (
                        score_lists,
                        feature_lists,
                        hc_lists,
                        diagnostics,
                    ) = build_corrupted_lists(
                        base_mask=base_mask,
                        score_values=test_score,
                        feature_values=test_feature,
                        handcrafted_values=test_hc,
                        num_regions=num_regions,
                        active_users_per_region=settings.active_users_per_region,
                        p_md=p_md,
                        p_fa=p_fa,
                        seed=condition_seed,
                        score_bits=settings.score_bits,
                        feature_bits_per_component=(
                            settings.feature_bits_per_component
                        ),
                        handcrafted_bits_per_component=(
                            settings.handcrafted_bits_per_component
                        ),
                        clip_lower=settings.clip_lower,
                        clip_upper=settings.clip_upper,
                        vector_reconstruction_mode=(
                            settings.vector_reconstruction_mode
                        ),
                    )

                    probabilities_by_scheme = {
                        "TERRA-Score": average_variable_scores(
                            score_lists,
                            empty_region_probability=(
                                settings.empty_region_probability
                            ),
                        ),
                        "TERRA-Feature": predict_variable_deepsets(
                            learned_model,
                            feature_lists,
                            input_dim=4,
                            batch_size=settings.batch_size,
                            device=device,
                            empty_region_probability=(
                                settings.empty_region_probability
                            ),
                        ),
                        "Handcrafted-Stats": predict_variable_deepsets(
                            hc_model,
                            hc_lists,
                            input_dim=30,
                            batch_size=settings.batch_size,
                            device=device,
                            empty_region_probability=(
                                settings.empty_region_probability
                            ),
                        ),
                    }

                    for scheme, probabilities in probabilities_by_scheme.items():
                        seed_rows.append(
                            {
                                "fold": fold,
                                "scheme": scheme,
                                "p_md": float(p_md),
                                "p_fa": float(p_fa),
                                "realization": int(realization),
                                "seed": int(condition_seed),
                                **diagnostics,
                                **metrics(test_targets, probabilities),
                            }
                        )

                print(" done")

    seed_df = pd.DataFrame(seed_rows)
    fold_df, aggregate_df = aggregate_seed_metrics(seed_df)

    seed_path = output_root / "ura_impairment_seed_metrics.csv"
    fold_path = output_root / "ura_impairment_fold_metrics.csv"
    aggregate_path = output_root / "ura_impairment_aggregate_metrics.csv"

    seed_df.to_csv(seed_path, index=False)
    fold_df.to_csv(fold_path, index=False)
    aggregate_df.to_csv(aggregate_path, index=False)

    plot_heatmaps(
        aggregate_df,
        output_root=output_root,
    )

    task_bits = {
        "TERRA-Score": int(settings.score_bits),
        "TERRA-Feature": int(4 * settings.feature_bits_per_component),
        "Handcrafted-Stats": int(
            30 * settings.handcrafted_bits_per_component
        ),
    }

    if settings.nonce_bits is None:
        message_bits = {scheme: None for scheme in SCHEMES}
    else:
        message_bits = {
            scheme: int(
                tag_bits + task_bits[scheme] + settings.nonce_bits
            )
            for scheme in SCHEMES
        }

    summary = {
        "configuration": {
            "num_regions": num_regions,
            "minimum_region_tag_bits": tag_bits,
            "active_users_per_region": settings.active_users_per_region,
            "p_md_values": list(settings.p_md_values),
            "p_fa_values": list(settings.p_fa_values),
            "monte_carlo_seeds": settings.monte_carlo_seeds,
            "empty_region_probability": settings.empty_region_probability,
            "task_bits": task_bits,
            "nonce_bits_for_reporting_only": settings.nonce_bits,
            "message_bits_if_nonce_known": message_bits,
            "clip_lower": settings.clip_lower,
            "clip_upper": settings.clip_upper,
            "vector_reconstruction_mode": (
                settings.vector_reconstruction_mode
            ),
            "score_reconstruction_mode": "endpoints",
        },
        "methodology": {
            "error_abstraction": (
                "Only global URA list-level missed detections and false alarms "
                "are emulated. No separate payload-bit-error axis is used."
            ),
            "p_md": (
                "Each transmitted full message is independently missed with "
                "probability P_MD."
            ),
            "p_fa": (
                "For each slot, the integer number of false decoded full "
                "messages is ratio-matched so that conditional expected "
                "F/(T+F) equals the requested global P_FA."
            ),
            "fa_region_tags": (
                "Each FA has a uniformly random valid region tag. The "
                "implementation uses the equivalent binomial thinning of the "
                "global FA count to determine contamination of the target "
                "regional receiver."
            ),
            "fa_task_payload": (
                "Uniformly random valid quantized task word for each scheme. "
                "A false task payload may equal a genuine task payload."
            ),
            "full_message_uniqueness": (
                "Synthetic disjoint nonce/message IDs guarantee that FA full "
                "messages were not transmitted. Their physical bit length is "
                "not used by the synthetic impairment generator."
            ),
            "other_regions": (
                "The dataset materializes the target region only. Other valid "
                "regions are assumed to have the same configured number of "
                "active users and enter through their aggregate recovered-true "
                "count when generating the global FA count."
            ),
            "aggregation": (
                "Metrics are computed for each Monte Carlo realization, "
                "averaged within each outer fold, then averaged equally across "
                "the three folds."
            ),
            "operational_metrics": (
                "MDR@FAR<=1% and MDR@FAR<=5% are ROC operating metrics; no "
                "test-derived deployment threshold is exported."
            ),
        },
        "folds": fold_metadata,
        "outputs": {
            "seed_metrics": str(seed_path),
            "fold_metrics": str(fold_path),
            "aggregate_metrics": str(aggregate_path),
        },
    }
    save_json(
        summary,
        output_root / "ura_impairment_summary.json",
    )

    print("\nSaved:")
    print(f"  {seed_path}")
    print(f"  {fold_path}")
    print(f"  {aggregate_path}")
    print(f"  {output_root / 'ura_impairment_summary.json'}")
    print(f"  heatmaps under {output_root}")

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
