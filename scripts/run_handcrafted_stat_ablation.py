#!/usr/bin/env python
"""
Handcrafted temporal-statistics ablation for the TERRA paper.

Purpose
-------
Give the non-AI handcrafted benchmark the same opportunity to become compact
before the fixed-budget quantization study.

The existing regional dataset stores seven statistics for each sensor channel:
    mean, std, min, max, last, delta, slope

The source implementation stores them in statistic-major order, but this script
does NOT rely on that ordering. It reads `handcrafted_feature_names` from the
fold metadata and selects dimensions by name.

Study design
------------
A) Predefined nested size ablation (used for size selection):
    H1: mean                                            -> 15 dims
    H2: mean, std                                       -> 30 dims
    H3: mean, std, slope                                -> 45 dims
    H4: mean, std, slope, last                          -> 60 dims
    H5: mean, std, slope, last, max                     -> 75 dims
    H6: mean, std, slope, last, max, min                -> 90 dims
    H7: mean, std, slope, last, max, min, delta         -> 105 dims

B) Leave-one-statistic-out interpretation:
    Full H7 minus each of the seven statistics.

H6 is exactly "leave out delta", so that condition is trained only once and
reused in both tables.

Receiver protocol
-----------------
For every unique handcrafted subset:
  - handcrafted representation only
  - DeepSets only
  - fixed-budget empirical regional sampling
  - validation-AUPRC checkpoint selection
  - full reports (K=9)
  - same 3 outer folds and inner manifests as the locked regional study

Primary size-selection rule (nested variants only)
--------------------------------------------------
Choose the SMALLEST nested representation satisfying:
  mean AUPRC >= best nested mean AUPRC - 0.005
and
  mean MDR@FAR<=1% <= best nested mean MDR@FAR<=1% + 0.02

The 0.02 MDR tolerance is two absolute percentage points.

The leave-one-out results are interpretive only and are NOT used to construct a
post-hoc optimized subset.

The script reuses the existing full 105-D handcrafted + DeepSets result when
available. It creates temporary sliced regional datasets for the other variants,
runs the existing `scripts/run_region_cv.py`, and removes the sliced datasets
after successful training unless --keep-sliced-datasets is set.

Typical commands
----------------
Dry run:
    python scripts/run_handcrafted_stat_ablation.py --prepare-only

Full study:
    python scripts/run_handcrafted_stat_ablation.py

Outputs
-------
runs/handcrafted_stat_ablation/
    generated_configs/
    receiver_runs/
    handcrafted_stat_fold_metrics.csv
    handcrafted_stat_nested_aggregate.csv
    handcrafted_stat_leave_one_out_aggregate.csv
    handcrafted_stat_selection.json
    handcrafted_stat_nested_auprc.png
    handcrafted_stat_nested_low_far.png
    handcrafted_stat_leave_one_out_auprc.png

All executable steps report runtime, including this orchestration script.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

from fire_ura.region_data import RegionArrays, load_region_arrays, save_region_arrays


ALL_STATS = ("mean", "std", "min", "max", "last", "delta", "slope")

# Fixed BEFORE looking at this ablation's results.
NESTED_VARIANTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("H1", ("mean",)),
    ("H2", ("mean", "std")),
    ("H3", ("mean", "std", "slope")),
    ("H4", ("mean", "std", "slope", "last")),
    ("H5", ("mean", "std", "slope", "last", "max")),
    ("H6", ("mean", "std", "slope", "last", "max", "min")),
    ("H7", ("mean", "std", "slope", "last", "max", "min", "delta")),
)


@dataclass(frozen=True)
class Variant:
    variant_id: str
    family: str
    label: str
    included_stats: tuple[str, ...]
    omitted_stat: str | None = None
    reused_from: str | None = None

    @property
    def n_stats(self) -> int:
        return len(self.included_stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run handcrafted-statistics size and leave-one-out ablations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("runs/region_dataset_all15_d35"),
        help=(
            "Existing all-15-channel regional dataset containing the full 105-D "
            "handcrafted representation."
        ),
    )
    parser.add_argument(
        "--sensor-cv-run-root",
        type=Path,
        default=Path("runs/sensor_cv_all15_d35"),
        help="Sensor CV run containing the locked inner split manifests.",
    )
    parser.add_argument(
        "--full-baseline-run-root",
        type=Path,
        default=Path("runs/region_cv_all15_d35_empirical_fixed_budget"),
        help=(
            "Completed locked regional CV run whose handcrafted+DeepSets H7 result "
            "can be reused."
        ),
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("runs/handcrafted_stat_ablation"),
        help="Output root for this ablation.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device passed to run_region_cv.py.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.005,
        help="Absolute mean-AUPRC tolerance from the best nested variant.",
    )
    parser.add_argument(
        "--mdr-tolerance",
        type=float,
        default=0.02,
        help=(
            "Absolute MDR@FAR<=1%% tolerance from the best nested variant. "
            "Default 0.02 = two percentage points."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate inputs, generate configs, and print the plan without training.",
    )
    parser.add_argument(
        "--keep-sliced-datasets",
        action="store_true",
        help="Keep temporary sliced regional datasets after successful runs.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help=(
            "Do not reuse completed ablation receiver runs. Existing run directories "
            "must be removed manually first."
        ),
    )
    parser.add_argument(
        "--rerun-full-h7",
        action="store_true",
        help=(
            "Do not reuse the existing full 105-D baseline; retrain H7 inside this "
            "ablation using the locked protocol."
        ),
    )
    return parser.parse_args()


def save_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def convert(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): convert(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(v) for v in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return convert(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(convert(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp.replace(path)


def write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n" + "=" * 96)
    print("RUN:")
    print(" ".join(command))
    print("=" * 96)
    subprocess.run(command, cwd=cwd, check=True)


def nested_variants() -> list[Variant]:
    return [
        Variant(
            variant_id=f"nested_{name.lower()}",
            family="nested",
            label=name,
            included_stats=stats,
        )
        for name, stats in NESTED_VARIANTS
    ]


def leave_one_out_variants() -> list[Variant]:
    outputs: list[Variant] = []
    for omitted in ALL_STATS:
        included = tuple(stat for stat in ALL_STATS if stat != omitted)
        if omitted == "delta":
            # Exactly the same set as nested H6, only in a different order.
            outputs.append(
                Variant(
                    variant_id="loo_without_delta",
                    family="leave_one_out",
                    label="without_delta",
                    included_stats=included,
                    omitted_stat=omitted,
                    reused_from="nested_h6",
                )
            )
        else:
            outputs.append(
                Variant(
                    variant_id=f"loo_without_{omitted}",
                    family="leave_one_out",
                    label=f"without_{omitted}",
                    included_stats=included,
                    omitted_stat=omitted,
                )
            )
    return outputs


def all_conceptual_variants() -> list[Variant]:
    return nested_variants() + leave_one_out_variants()


def unique_train_variants() -> list[Variant]:
    """
    Unique receiver trainings required.

    H7 may be reused from the locked full baseline at runtime.
    LOO-without-delta is reused from nested H6.
    """
    outputs: list[Variant] = []
    seen_sets: set[frozenset[str]] = set()
    for variant in all_conceptual_variants():
        key = frozenset(variant.included_stats)
        if key in seen_sets:
            continue
        seen_sets.add(key)
        outputs.append(variant)
    return outputs


def validate_region_script(project_root: Path) -> None:
    path = project_root / "scripts" / "run_region_cv.py"
    if not path.is_file():
        raise FileNotFoundError(path)

    text = path.read_text(encoding="utf-8", errors="replace")
    if "empirical_fixed_budget" not in text:
        raise RuntimeError(
            "scripts/run_region_cv.py does not contain 'empirical_fixed_budget'. "
            "This ablation must use the locked fixed-budget empirical sampler."
        )
    if "best_auprc" not in text and "validation_auprc" not in text:
        raise RuntimeError(
            "Could not verify validation-AUPRC checkpoint selection in "
            "scripts/run_region_cv.py."
        )


def read_fold_feature_names(dataset_root: Path, fold: int) -> list[str]:
    metadata_path = (
        dataset_root
        / f"fold_{fold}"
        / f"region_metadata_fold{fold}.json"
    )
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"Missing regional metadata needed to identify handcrafted columns:\n"
            f"  {metadata_path}"
        )
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    names = payload.get("handcrafted_feature_names")
    if not isinstance(names, list) or not names:
        raise KeyError(
            f"{metadata_path} lacks a nonempty handcrafted_feature_names list."
        )
    return [str(name) for name in names]


def parse_feature_name(name: str) -> tuple[str, str]:
    if "__" not in name:
        raise ValueError(
            f"Handcrafted feature name {name!r} does not follow channel__stat."
        )
    channel, stat = name.rsplit("__", 1)
    return channel, stat


def validate_full_handcrafted_dataset(
    dataset_root: Path,
    *,
    n_folds: int = 3,
) -> dict[str, Any]:
    reference_names: list[str] | None = None
    reference_channels: list[str] | None = None
    fold_dimensions: list[int] = []

    for fold in range(n_folds):
        fold_dir = dataset_root / f"fold_{fold}"
        npz_path = fold_dir / f"region_dataset_fold{fold}.npz"
        manifest_path = fold_dir / f"region_manifest_fold{fold}.csv"
        arrays = load_region_arrays(npz_path, manifest_path)
        names = read_fold_feature_names(dataset_root, fold)

        if len(names) != arrays.handcrafted.shape[2]:
            raise ValueError(
                f"Fold {fold}: metadata has {len(names)} handcrafted names but "
                f"array dimension is {arrays.handcrafted.shape[2]}."
            )

        parsed = [parse_feature_name(name) for name in names]
        stats_found = {stat for _, stat in parsed}
        if stats_found != set(ALL_STATS):
            raise ValueError(
                f"Fold {fold}: expected handcrafted statistics {ALL_STATS}, "
                f"found {sorted(stats_found)}."
            )

        channels = []
        for channel, _ in parsed:
            if channel not in channels:
                channels.append(channel)

        counts = {
            stat: sum(1 for _, feature_stat in parsed if feature_stat == stat)
            for stat in ALL_STATS
        }
        if len(set(counts.values())) != 1:
            raise ValueError(
                f"Fold {fold}: inconsistent channel counts by statistic: {counts}"
            )
        n_channels = next(iter(counts.values()))
        if n_channels != len(channels):
            raise ValueError(
                f"Fold {fold}: parsed {len(channels)} channels but each statistic "
                f"contains {n_channels} dimensions."
            )

        if reference_names is None:
            reference_names = names
            reference_channels = channels
        else:
            if names != reference_names:
                raise ValueError(
                    f"Fold {fold}: handcrafted feature-name ordering differs from fold 0."
                )
            if channels != reference_channels:
                raise ValueError(
                    f"Fold {fold}: channel list differs from fold 0."
                )

        fold_dimensions.append(int(arrays.handcrafted.shape[2]))

    assert reference_names is not None
    assert reference_channels is not None

    return {
        "feature_names": reference_names,
        "channels": reference_channels,
        "n_channels": len(reference_channels),
        "full_dimension": len(reference_names),
        "fold_dimensions": fold_dimensions,
    }


def selected_feature_indices(
    feature_names: Sequence[str],
    included_stats: Sequence[str],
) -> np.ndarray:
    requested = set(str(stat) for stat in included_stats)
    unknown = requested - set(ALL_STATS)
    if unknown:
        raise ValueError(f"Unknown handcrafted statistics: {sorted(unknown)}")

    indices = [
        index
        for index, name in enumerate(feature_names)
        if parse_feature_name(str(name))[1] in requested
    ]
    if not indices:
        raise ValueError("No handcrafted dimensions were selected.")
    return np.asarray(indices, dtype=np.int64)


def selected_feature_names(
    feature_names: Sequence[str],
    indices: np.ndarray,
) -> list[str]:
    return [str(feature_names[int(index)]) for index in indices]


def build_sliced_dataset(
    *,
    source_root: Path,
    destination_root: Path,
    variant: Variant,
    feature_names: Sequence[str],
    n_folds: int = 3,
) -> int:
    if destination_root.exists():
        raise FileExistsError(
            f"Temporary sliced dataset already exists:\n  {destination_root}"
        )

    indices = selected_feature_indices(feature_names, variant.included_stats)
    destination_root.mkdir(parents=True, exist_ok=False)

    for fold in range(n_folds):
        source_fold = source_root / f"fold_{fold}"
        destination_fold = destination_root / f"fold_{fold}"
        destination_fold.mkdir(parents=True, exist_ok=False)

        source_npz = source_fold / f"region_dataset_fold{fold}.npz"
        source_manifest = source_fold / f"region_manifest_fold{fold}.csv"
        arrays = load_region_arrays(source_npz, source_manifest)

        sliced = RegionArrays(
            learned=arrays.learned,
            handcrafted=arrays.handcrafted[:, :, indices],
            local_probability=arrays.local_probability,
            mask=arrays.mask,
            target=arrays.target,
            split_code=arrays.split_code,
            metadata=arrays.metadata.copy(),
        )

        save_region_arrays(
            sliced,
            destination_fold / f"region_dataset_fold{fold}.npz",
            destination_fold / f"region_manifest_fold{fold}.csv",
        )

        # Keep useful metadata with the temporary dataset as a safety check.
        metadata_source = source_fold / f"region_metadata_fold{fold}.json"
        metadata_payload = (
            json.loads(metadata_source.read_text(encoding="utf-8"))
            if metadata_source.is_file()
            else {}
        )
        metadata_payload["handcrafted_dimension"] = int(indices.size)
        metadata_payload["handcrafted_feature_names"] = selected_feature_names(
            feature_names, indices
        )
        metadata_payload["handcrafted_ablation_variant"] = variant.variant_id
        metadata_payload["handcrafted_included_stats"] = list(
            variant.included_stats
        )
        save_json(
            metadata_payload,
            destination_fold / f"region_metadata_fold{fold}.json",
        )

    save_json(
        {
            "source_dataset_root": str(source_root),
            "variant": variant.variant_id,
            "family": variant.family,
            "included_stats": list(variant.included_stats),
            "dimension": int(indices.size),
            "feature_names": selected_feature_names(feature_names, indices),
        },
        destination_root / "ablation_dataset_summary.json",
    )

    return int(indices.size)


def make_region_config(
    *,
    sensor_run_root: Path,
    dataset_root: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    return {
        "sensor_cv_run_root": str(sensor_run_root),
        "region_dataset_root": str(dataset_root),
        "region_training": {
            "output_root": str(output_root),
            "device": device,
            "seed": 42,
            "n_folds": 3,
            "representations": ["handcrafted"],
            "include_average_local_probability": False,
            "include_mean_logreg": False,
            "neural_methods": ["deepsets"],
            "sampling_mode": "empirical_fixed_budget",
            "batch_size": 256,
            "num_workers": 0,
            "samples_per_epoch": 6000,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 30,
            "early_stopping_patience": 6,
            "minimum_improvement": 0.0001,
            "gradient_clip_norm": 5.0,
            "hidden_dim": 32,
            "phi_hidden_dim": 32,
            "phi_output_dim": 16,
            "rho_hidden_dim": 16,
            "dropout": 0.15,
        },
    }


def region_run_complete(root: Path) -> bool:
    return (
        (root / "region_fold_metrics.csv").is_file()
        and all(
            (
                root
                / f"fold_{fold}"
                / f"region_predictions_fold{fold}.csv"
            ).is_file()
            for fold in range(3)
        )
    )


def full_baseline_usable(root: Path) -> bool:
    if not region_run_complete(root):
        return False
    table = pd.read_csv(root / "region_fold_metrics.csv")
    selected = table[
        (table["representation"].astype(str) == "handcrafted")
        & (table["method"].astype(str) == "deepsets")
    ]
    return len(selected) == 3


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


def collect_receiver_run(
    *,
    run_root: Path,
    variant: Variant,
    dimension: int,
) -> list[dict[str, Any]]:
    metric_path = run_root / "region_fold_metrics.csv"
    table = pd.read_csv(metric_path)
    selected = table[
        (table["representation"].astype(str) == "handcrafted")
        & (table["method"].astype(str) == "deepsets")
    ].copy()

    if len(selected) != 3:
        raise ValueError(
            f"{run_root}: expected 3 handcrafted+deepsets rows, found {len(selected)}."
        )

    rows: list[dict[str, Any]] = []
    for _, metric_row in selected.sort_values("fold").iterrows():
        fold = int(metric_row["fold"])
        prediction_path = (
            run_root
            / f"fold_{fold}"
            / f"region_predictions_fold{fold}.csv"
        )
        predictions = pd.read_csv(prediction_path)

        probability_column = "probability__handcrafted__deepsets"
        if probability_column not in predictions.columns:
            raise KeyError(
                f"{prediction_path} lacks {probability_column!r}."
            )
        if "target" not in predictions.columns:
            raise KeyError(f"{prediction_path} lacks 'target'.")

        targets = predictions["target"].to_numpy(dtype=np.int64)
        probabilities = predictions[probability_column].to_numpy(dtype=np.float64)

        mdr1, far1 = mdr_at_far(
            targets, probabilities, target_far=0.01
        )
        mdr5, far5 = mdr_at_far(
            targets, probabilities, target_far=0.05
        )

        rows.append(
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "label": variant.label,
                "included_stats": ",".join(variant.included_stats),
                "omitted_stat": variant.omitted_stat or "",
                "dimension": int(dimension),
                "fold": fold,
                "auprc": safe_auprc(targets, probabilities),
                "auroc": safe_auroc(targets, probabilities),
                "mdr_at_far_1pct": mdr1,
                "achieved_far_at_1pct": far1,
                "mdr_at_far_5pct": mdr5,
                "achieved_far_at_5pct": far5,
            }
        )

    return rows


def aggregate_variants(fold_table: pd.DataFrame) -> pd.DataFrame:
    metric_names = (
        "auprc",
        "auroc",
        "mdr_at_far_1pct",
        "achieved_far_at_1pct",
        "mdr_at_far_5pct",
        "achieved_far_at_5pct",
    )

    keys = (
        "variant_id",
        "family",
        "label",
        "included_stats",
        "omitted_stat",
        "dimension",
    )

    rows: list[dict[str, Any]] = []

    for key_values, group in fold_table.groupby(
        list(keys), sort=True, dropna=False
    ):
        row = dict(zip(keys, key_values))
        row["n_folds"] = int(len(group))

        for metric in metric_names:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=1))
                if finite.size >= 2
                else 0.0
                if finite.size == 1
                else float("nan")
            )
            row[f"{metric}_values"] = json.dumps(values.tolist())

        rows.append(row)

    return pd.DataFrame(rows)


def recommend_nested_variant(
    nested: pd.DataFrame,
    *,
    auprc_tolerance: float,
    mdr_tolerance: float,
) -> dict[str, Any]:
    if nested.empty:
        raise ValueError("Nested ablation table is empty.")

    best_auprc = float(nested["auprc_mean"].max())
    best_mdr = float(nested["mdr_at_far_1pct_mean"].min())
    auprc_cutoff = best_auprc - float(auprc_tolerance)
    mdr_cutoff = best_mdr + float(mdr_tolerance)

    eligible = nested[
        (nested["auprc_mean"] >= auprc_cutoff)
        & (nested["mdr_at_far_1pct_mean"] <= mdr_cutoff)
    ].copy()

    fallback_used = False
    if eligible.empty:
        fallback_used = True
        eligible = nested[nested["auprc_mean"] >= auprc_cutoff].copy()

    if eligible.empty:
        raise RuntimeError("No nested variant satisfies the recommendation rule.")

    eligible = eligible.sort_values(["dimension", "variant_id"])
    selected = eligible.iloc[0]

    return {
        "selected_variant": str(selected["variant_id"]),
        "selected_label": str(selected["label"]),
        "selected_dimension": int(selected["dimension"]),
        "selected_stats": str(selected["included_stats"]).split(","),
        "eligible_variants": eligible["variant_id"].astype(str).tolist(),
        "best_mean_auprc": best_auprc,
        "best_mean_auprc_variant": str(
            nested.loc[nested["auprc_mean"].idxmax(), "variant_id"]
        ),
        "auprc_cutoff": auprc_cutoff,
        "best_mean_mdr_at_far_1pct": best_mdr,
        "best_mean_mdr_variant": str(
            nested.loc[nested["mdr_at_far_1pct_mean"].idxmin(), "variant_id"]
        ),
        "mdr_cutoff": mdr_cutoff,
        "fallback_to_auprc_only": fallback_used,
        "rule": (
            "Choose the smallest predefined nested handcrafted representation "
            f"with mean AUPRC within {auprc_tolerance:.6f} of the best nested "
            "variant and mean MDR@FAR<=1% within "
            f"{mdr_tolerance:.6f} of the best nested variant."
        ),
        "methodology_note": (
            "Leave-one-statistic-out results are interpretive only and are not "
            "used to construct a post-hoc optimized handcrafted subset. The three "
            "outer folds have already been explored during development, so this "
            "is an ablation/development choice rather than a pristine unbiased "
            "model-selection estimate."
        ),
    }


def plot_nested(
    nested: pd.DataFrame,
    *,
    value_column: str,
    std_column: str,
    ylabel: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    ordered = nested.sort_values("dimension")
    x = ordered["dimension"].to_numpy(dtype=int)
    y = ordered[value_column].to_numpy(dtype=float)
    std = ordered[std_column].to_numpy(dtype=float)

    if percentage:
        y = 100.0 * y
        std = 100.0 * std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=std, marker="o", capsize=4)
    ax.set_xlabel("Handcrafted representation dimension")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_leave_one_out(loo: pd.DataFrame, output_path: Path) -> None:
    ordered = loo.sort_values("omitted_stat")
    x = np.arange(len(ordered), dtype=int)
    y = ordered["auprc_mean"].to_numpy(dtype=float)
    std = ordered["auprc_std"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, y, yerr=std, capsize=4)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"-{value}" for value in ordered["omitted_stat"].astype(str)],
        rotation=30,
        ha="right",
    )
    ax.set_ylabel("Regional AUPRC")
    ax.set_xlabel("Omitted statistic")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    project_root = Path.cwd().resolve()
    dataset_root = args.dataset_root.resolve()
    sensor_run_root = args.sensor_cv_run_root.resolve()
    baseline_run_root = args.full_baseline_run_root.resolve()
    study_root = args.study_root.resolve()

    config_root = study_root / "generated_configs"
    receiver_root = study_root / "receiver_runs"
    sliced_root = study_root / "_temporary_sliced_datasets"

    if args.selection_tolerance < 0 or args.mdr_tolerance < 0:
        raise ValueError("Selection tolerances must be nonnegative.")
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)
    if not sensor_run_root.is_dir():
        raise FileNotFoundError(sensor_run_root)

    validate_region_script(project_root)

    dataset_info = validate_full_handcrafted_dataset(dataset_root)
    feature_names = dataset_info["feature_names"]
    n_channels = int(dataset_info["n_channels"])

    if n_channels != 15:
        print(
            f"WARNING: detected {n_channels} channels, not the expected 15. "
            "Dimensions will be derived from the actual metadata."
        )

    study_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)
    receiver_root.mkdir(parents=True, exist_ok=True)

    conceptual = all_conceptual_variants()
    unique_variants = unique_train_variants()

    print("Handcrafted temporal-statistics ablation")
    print("=" * 96)
    print(f"Project root:       {project_root}")
    print(f"Dataset root:       {dataset_root}")
    print(f"Sensor CV root:     {sensor_run_root}")
    print(f"Full baseline root: {baseline_run_root}")
    print(f"Study root:         {study_root}")
    print(f"Detected channels:  {n_channels}")
    print(f"Full HC dimension:  {dataset_info['full_dimension']}")
    print(
        f"Recommendation: AUPRC tolerance={args.selection_tolerance}, "
        f"MDR@1% tolerance={args.mdr_tolerance}"
    )
    print(
        "Conceptual conditions: "
        f"{len(conceptual)} (7 nested + 7 leave-one-out); "
        f"unique feature subsets: {len(unique_variants)}."
    )
    print(
        "Note: H6 and leave-one-out-delta are the same feature set and are "
        "trained only once."
    )

    h7_can_reuse = (
        not args.rerun_full_h7
        and full_baseline_usable(baseline_run_root)
    )
    print(
        "H7 full baseline: "
        + ("will be reused" if h7_can_reuse else "will be trained in this study")
    )

    planned: list[dict[str, Any]] = []

    for variant in unique_variants:
        indices = selected_feature_indices(feature_names, variant.included_stats)
        dimension = int(indices.size)
        variant_run_root = receiver_root / variant.variant_id
        variant_dataset_root = sliced_root / variant.variant_id
        config_path = config_root / f"{variant.variant_id}.yaml"

        is_h7 = frozenset(variant.included_stats) == frozenset(ALL_STATS)
        source_dataset_for_run = dataset_root if is_h7 else variant_dataset_root

        config = make_region_config(
            sensor_run_root=sensor_run_root,
            dataset_root=source_dataset_for_run,
            output_root=variant_run_root,
            device=args.device,
        )
        write_yaml(config, config_path)

        planned.append(
            {
                "variant_id": variant.variant_id,
                "family": variant.family,
                "label": variant.label,
                "included_stats": list(variant.included_stats),
                "dimension": dimension,
                "receiver_run_root": str(variant_run_root),
                "dataset_root_for_run": str(source_dataset_for_run),
                "config": str(config_path),
                "reuse_full_baseline": bool(is_h7 and h7_can_reuse),
            }
        )

    save_json(
        {
            "dataset_info": dataset_info,
            "nested_variants": [
                {
                    "variant_id": v.variant_id,
                    "label": v.label,
                    "included_stats": list(v.included_stats),
                }
                for v in nested_variants()
            ],
            "leave_one_out_variants": [
                {
                    "variant_id": v.variant_id,
                    "label": v.label,
                    "included_stats": list(v.included_stats),
                    "omitted_stat": v.omitted_stat,
                    "reused_from": v.reused_from,
                }
                for v in leave_one_out_variants()
            ],
            "planned_unique_runs": planned,
        },
        study_root / "handcrafted_stat_study_plan.json",
    )

    if args.prepare_only:
        print("\n=== Planned unique receiver conditions ===")
        for item in planned:
            suffix = " [reuse H7 baseline]" if item["reuse_full_baseline"] else ""
            print(
                f"  {item['variant_id']:<24} "
                f"D={item['dimension']:<3d} "
                f"stats={','.join(item['included_stats'])}{suffix}"
            )
        print(
            "\nPreparation only: no sliced datasets or receiver training were run."
        )
        elapsed = time.perf_counter() - started
        print(f"Total runtime: {elapsed:.2f} seconds")
        return

    unique_results: dict[frozenset[str], list[dict[str, Any]]] = {}
    unique_dimensions: dict[frozenset[str], int] = {}

    for variant in unique_variants:
        key = frozenset(variant.included_stats)
        indices = selected_feature_indices(feature_names, variant.included_stats)
        dimension = int(indices.size)
        unique_dimensions[key] = dimension

        is_h7 = key == frozenset(ALL_STATS)

        print("\n" + "#" * 96)
        print(
            f"# {variant.variant_id}: D={dimension}, "
            f"stats={','.join(variant.included_stats)}"
        )
        print("#" * 96)

        if is_h7 and h7_can_reuse:
            print(f"Reusing locked full H7 baseline: {baseline_run_root}")
            unique_results[key] = collect_receiver_run(
                run_root=baseline_run_root,
                variant=variant,
                dimension=dimension,
            )
            continue

        run_root = receiver_root / variant.variant_id
        config_path = config_root / f"{variant.variant_id}.yaml"

        completed = region_run_complete(run_root)
        if completed and not args.rerun_existing:
            print(f"Reusing completed ablation run: {run_root}")
        else:
            if run_root.exists():
                raise FileExistsError(
                    f"Receiver output exists but cannot be overwritten:\n  {run_root}\n"
                    "Delete/rename it before using --rerun-existing."
                )

            if is_h7:
                run_dataset_root = dataset_root
            else:
                run_dataset_root = sliced_root / variant.variant_id
                if run_dataset_root.exists():
                    shutil.rmtree(run_dataset_root)
                built_dimension = build_sliced_dataset(
                    source_root=dataset_root,
                    destination_root=run_dataset_root,
                    variant=variant,
                    feature_names=feature_names,
                )
                if built_dimension != dimension:
                    raise RuntimeError(
                        f"Internal dimension mismatch: {built_dimension} != {dimension}"
                    )

            command = [
                sys.executable,
                "scripts/run_region_cv.py",
                "--config",
                str(config_path),
                "--dataset-root",
                str(run_dataset_root),
                "--output-root",
                str(run_root),
                "--device",
                args.device,
            ]
            run_command(command, cwd=project_root)

            if not is_h7 and not args.keep_sliced_datasets:
                try:
                    shutil.rmtree(run_dataset_root)
                    print(f"Removed temporary sliced dataset: {run_dataset_root}")
                except PermissionError as exc:
                    print(
                        "WARNING: Could not remove temporary sliced dataset "
                        f"because Windows reported that it is in use:\n"
                        f"  {run_dataset_root}\n"
                        f"  {exc}\n"
                        "The completed receiver result is unaffected. "
                        "The temporary directory can be deleted manually later."
                    )

        unique_results[key] = collect_receiver_run(
            run_root=run_root,
            variant=variant,
            dimension=dimension,
        )

    # Expand unique results into all 14 conceptual rows, including the deliberate
    # H6 == leave-one-out-delta reuse.
    all_fold_rows: list[dict[str, Any]] = []

    for variant in conceptual:
        key = frozenset(variant.included_stats)
        source_rows = unique_results[key]
        dimension = unique_dimensions[key]

        for source in source_rows:
            row = dict(source)
            row.update(
                {
                    "variant_id": variant.variant_id,
                    "family": variant.family,
                    "label": variant.label,
                    "included_stats": ",".join(variant.included_stats),
                    "omitted_stat": variant.omitted_stat or "",
                    "dimension": int(dimension),
                    "reused_from": variant.reused_from or "",
                }
            )
            all_fold_rows.append(row)

    fold_table = pd.DataFrame(all_fold_rows).sort_values(
        ["family", "dimension", "variant_id", "fold"]
    ).reset_index(drop=True)

    aggregate = aggregate_variants(fold_table)
    nested = aggregate[aggregate["family"] == "nested"].copy()
    nested["nested_rank"] = (
        nested["label"]
        .astype(str)
        .str.extract(r"H(\d+)", expand=False)
        .astype(int)
    )
    nested = nested.sort_values("nested_rank").reset_index(drop=True)

    loo = aggregate[aggregate["family"] == "leave_one_out"].copy()
    loo = loo.sort_values("omitted_stat").reset_index(drop=True)

    recommendation = recommend_nested_variant(
        nested,
        auprc_tolerance=args.selection_tolerance,
        mdr_tolerance=args.mdr_tolerance,
    )

    fold_path = study_root / "handcrafted_stat_fold_metrics.csv"
    nested_path = study_root / "handcrafted_stat_nested_aggregate.csv"
    loo_path = study_root / "handcrafted_stat_leave_one_out_aggregate.csv"
    selection_path = study_root / "handcrafted_stat_selection.json"

    fold_table.to_csv(fold_path, index=False)
    nested.to_csv(nested_path, index=False)
    loo.to_csv(loo_path, index=False)

    save_json(
        {
            "selection": recommendation,
            "protocol": {
                "full_handcrafted_statistics": list(ALL_STATS),
                "nested_order_fixed_before_results": True,
                "nested_variants_used_for_size_selection": True,
                "leave_one_out_used_for_interpretation_only": True,
                "regional_receiver": "handcrafted + deepsets",
                "regional_sampling": "empirical_fixed_budget",
                "regional_checkpoint_selection": "validation_auprc",
                "full_report_count": 9,
                "primary_metric": "regional_auprc",
                "secondary_operational_metric": "regional_mdr_at_far_le_1pct",
                "selection_tolerance_auprc": args.selection_tolerance,
                "selection_tolerance_mdr_at_far_1pct": args.mdr_tolerance,
                "h7_reused_from_locked_baseline": h7_can_reuse,
                "outer_folds_already_explored": True,
            },
            "nested": nested.to_dict(orient="records"),
            "leave_one_out": loo.to_dict(orient="records"),
        },
        selection_path,
    )

    plot_nested(
        nested,
        value_column="auprc_mean",
        std_column="auprc_std",
        ylabel="Regional AUPRC",
        output_path=study_root / "handcrafted_stat_nested_auprc.png",
    )
    plot_nested(
        nested,
        value_column="mdr_at_far_1pct_mean",
        std_column="mdr_at_far_1pct_std",
        ylabel="Regional MDR at FAR <= 1% (%)",
        output_path=study_root / "handcrafted_stat_nested_low_far.png",
        percentage=True,
    )
    plot_leave_one_out(
        loo,
        study_root / "handcrafted_stat_leave_one_out_auprc.png",
    )

    print("\n=== Nested handcrafted-size ablation ===")
    print(
        nested[
            [
                "label",
                "dimension",
                "included_stats",
                "auprc_mean",
                "auprc_std",
                "mdr_at_far_1pct_mean",
                "mdr_at_far_1pct_std",
                "mdr_at_far_5pct_mean",
            ]
        ].to_string(index=False)
    )

    print("\n=== Leave-one-statistic-out interpretation ===")
    print(
        loo[
            [
                "omitted_stat",
                "dimension",
                "auprc_mean",
                "auprc_std",
                "mdr_at_far_1pct_mean",
            ]
        ].to_string(index=False)
    )

    print("\n=== Handcrafted size recommendation ===")
    print(
        f"Selected {recommendation['selected_label']} "
        f"(D={recommendation['selected_dimension']}): "
        f"{recommendation['selected_stats']}"
    )
    print(recommendation["rule"])
    if recommendation["fallback_to_auprc_only"]:
        print(
            "WARNING: no nested variant satisfied both criteria; selection fell "
            "back to the AUPRC near-saturation rule."
        )

    print("\nSaved:")
    print(f"  {fold_path}")
    print(f"  {nested_path}")
    print(f"  {loo_path}")
    print(f"  {selection_path}")
    print(f"  {study_root / 'handcrafted_stat_nested_auprc.png'}")
    print(f"  {study_root / 'handcrafted_stat_nested_low_far.png'}")
    print(f"  {study_root / 'handcrafted_stat_leave_one_out_auprc.png'}")

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
