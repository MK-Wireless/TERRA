#!/usr/bin/env python
"""
Run and summarize the TERRA-Feature embedding-dimension ablation.

Default dimensions:
    D in {8, 16, 24, 35, 48, 64}

For each D, this script:
  1) copies the existing all-15-channel sensor-CV configuration and changes
     ONLY sensor_model.embedding_dim, sensor_model.output_tag, and output roots;
  2) runs the local sensor CV;
  3) builds the corresponding regional dataset;
  4) runs ONLY learned + DeepSets at the regional receiver, using the locked
     empirical-fixed-budget / validation-AUPRC protocol in run_region_cv.py;
  5) collects full-report regional AUPRC/AUROC and MDR@FAR<=1%/5%;
  6) records the local TERRA-Score AUPRC as a diagnostic versus D;
  7) recommends the smallest near-saturated D.

IMPORTANT METHODOLOGY NOTE
--------------------------
The current project has already used the three outer folds extensively during
development. Therefore this script should be treated as an ablation/development
study, not as a fresh unbiased model-selection estimate. It applies the same
outer folds to every D and selects the smallest near-saturated dimension.

Default recommendation rule:
  - mean regional AUPRC >= (best mean AUPRC - 0.005), AND
  - mean MDR@FAR<=1% <= (best mean MDR@FAR<=1% + 0.02),
  - then choose the SMALLEST eligible D.

The 0.02 MDR tolerance means two absolute percentage points.

Typical command:
    python scripts/run_feature_dimension_study.py ^
      --base-sensor-config runs\\sensor_cv_all15_d35\\config_all15_d35.yaml

Every executable step prints its own runtime; this orchestration script also
prints the total runtime.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


DEFAULT_DIMS = (8, 16, 24, 35, 48, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run TERRA-Feature embedding-dimension ablation."
    )
    parser.add_argument(
        "--base-sensor-config",
        type=Path,
        default=Path("runs/sensor_cv_all15_d35/config_all15_d35.yaml"),
        help=(
            "Existing all-15-channel D=35 sensor config used as the template. "
            "Only embedding_dim/output_tag/output roots are changed."
        ),
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        nargs="+",
        default=list(DEFAULT_DIMS),
        help="Embedding dimensions to evaluate.",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root directory for sensor/dataset/regional runs.",
    )
    parser.add_argument(
        "--study-root",
        type=Path,
        default=Path("runs/feature_dimension_study"),
        help="Directory for generated configs and dimension-study summaries.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default="auto",
        help="Device override passed to all three stages.",
    )
    parser.add_argument(
        "--selection-tolerance",
        type=float,
        default=0.005,
        help=(
            "Absolute mean-AUPRC tolerance from the best dimension for the "
            "smallest-near-saturated recommendation."
        ),
    )
    parser.add_argument(
        "--mdr-tolerance",
        type=float,
        default=0.02,
        help=(
            "Absolute MDR@FAR<=1%% tolerance from the best dimension. "
            "Default 0.02 = two percentage points."
        ),
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Generate configs and print planned commands without running training.",
    )
    parser.add_argument(
        "--rerun-existing",
        action="store_true",
        help=(
            "Re-run completed dimensions. Existing run directories are NOT deleted; "
            "use this only after manually removing/renaming those directories."
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

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(convert(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp.replace(path)


def load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a YAML mapping.")
    return payload


def write_yaml(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> None:
    print("\n" + "=" * 88)
    print("RUN:")
    print(" ".join(command))
    print("=" * 88)
    subprocess.run(command, cwd=cwd, check=True)


def mdr_at_far(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_far: float,
) -> tuple[float, float]:
    """Return (MDR, achieved FAR) at the best ROC point with FAR <= constraint."""
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if np.unique(targets).size < 2:
        return float("nan"), float("nan")

    fpr, tpr, _ = roc_curve(targets, probabilities, drop_intermediate=False)
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


def sensor_root_for(runs_root: Path, dim: int) -> Path:
    return runs_root / f"sensor_cv_all15_d{dim}"


def dataset_root_for(runs_root: Path, dim: int) -> Path:
    return runs_root / f"region_dataset_all15_d{dim}"


def region_root_for(runs_root: Path, dim: int) -> Path:
    return runs_root / f"region_cv_all15_d{dim}_empirical_fixed_budget"


def sensor_complete(root: Path, dim: int) -> bool:
    return (
        (root / f"summary_all15_d{dim}.json").is_file()
        and all((root / f"fold_{fold}" / "final").is_dir() for fold in range(3))
    )


def dataset_complete(root: Path) -> bool:
    return all(
        (root / f"fold_{fold}" / f"region_dataset_fold{fold}.npz").is_file()
        and (root / f"fold_{fold}" / f"region_manifest_fold{fold}.csv").is_file()
        for fold in range(3)
    )


def region_complete(root: Path) -> bool:
    return (
        (root / "region_fold_metrics.csv").is_file()
        and all(
            (root / f"fold_{fold}" / f"region_predictions_fold{fold}.csv").is_file()
            for fold in range(3)
        )
    )


def assert_no_partial_existing(root: Path, *, complete: bool, stage: str) -> None:
    if root.exists() and not complete:
        raise FileExistsError(
            f"{stage} directory exists but does not look complete:\n  {root}\n"
            "Rename/delete the partial directory before rerunning."
        )


def make_sensor_config(
    base: dict[str, Any], *, dim: int, sensor_root: Path
) -> dict[str, Any]:
    config = json.loads(json.dumps(base))
    if "sensor_model" not in config or "sensor_cv" not in config:
        raise KeyError(
            "Base sensor config must contain sensor_model and sensor_cv sections."
        )
    config["sensor_model"]["embedding_dim"] = int(dim)
    config["sensor_model"]["output_tag"] = f"all15_d{dim}"
    config["sensor_cv"]["output_root"] = str(sensor_root)
    return config


def make_region_config(
    *,
    sensor_root: Path,
    sensor_config_path: Path,
    dataset_root: Path,
    region_root: Path,
    device: str,
) -> dict[str, Any]:
    return {
        "sensor_cv_run_root": str(sensor_root),
        "sensor_config_path": str(sensor_config_path),
        "region_dataset_root": str(dataset_root),
        "region_dataset": {
            "output_root": str(dataset_root),
            "device": device,
            "batch_size": 1024,
            "num_workers": 0,
            "pin_memory": True,
            "expected_max_sensors": 9,
        },
        "region_training": {
            "output_root": str(region_root),
            "device": device,
            "seed": 42,
            "n_folds": 3,
            "representations": ["learned"],
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


def validate_region_script(project_root: Path) -> None:
    path = project_root / "scripts" / "run_region_cv.py"
    if not path.is_file():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    if "empirical_fixed_budget" not in text:
        raise RuntimeError(
            "Your local scripts/run_region_cv.py does not contain "
            "'empirical_fixed_budget'. This dimension study must use the locked "
            "fixed-budget empirical sampler, so update run_region_cv.py first."
        )
    if "best_auprc" not in text and "validation_auprc" not in text:
        raise RuntimeError(
            "Could not verify validation-AUPRC checkpoint selection in "
            "scripts/run_region_cv.py. Inspect that file before running this study."
        )


def find_sensor_fold_metrics(sensor_root: Path, dim: int) -> Path:
    preferred = sensor_root / f"fold_metrics_all15_d{dim}.csv"
    if preferred.is_file():
        return preferred
    candidates = sorted(sensor_root.glob("fold_metrics_*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Could not uniquely find sensor fold metrics in {sensor_root}. "
            f"Candidates: {candidates}"
        )
    return candidates[0]


def collect_dimension(
    *, dim: int, sensor_root: Path, region_root: Path
) -> list[dict[str, Any]]:
    sensor_metrics = pd.read_csv(find_sensor_fold_metrics(sensor_root, dim))
    if "fold" not in sensor_metrics.columns:
        sensor_metrics = sensor_metrics.copy()
        sensor_metrics["fold"] = np.arange(len(sensor_metrics), dtype=int)

    region_metrics = pd.read_csv(region_root / "region_fold_metrics.csv")
    subset = region_metrics[
        (region_metrics["representation"].astype(str) == "learned")
        & (region_metrics["method"].astype(str) == "deepsets")
    ].copy()
    if len(subset) != 3:
        raise ValueError(
            f"D={dim}: expected 3 learned+deepsets regional fold rows, found {len(subset)}."
        )

    sensor_by_fold = {
        int(row["fold"]): row for _, row in sensor_metrics.iterrows()
    }
    rows: list[dict[str, Any]] = []

    for _, region_row in subset.sort_values("fold").iterrows():
        fold = int(region_row["fold"])
        pred_path = (
            region_root / f"fold_{fold}" / f"region_predictions_fold{fold}.csv"
        )
        predictions = pd.read_csv(pred_path)
        probability_column = "probability__learned__deepsets"
        if probability_column not in predictions.columns:
            raise KeyError(f"{pred_path} lacks {probability_column!r}.")
        if "target" not in predictions.columns:
            raise KeyError(f"{pred_path} lacks 'target'.")

        targets = predictions["target"].to_numpy(dtype=np.int64)
        probabilities = predictions[probability_column].to_numpy(dtype=np.float64)
        mdr1, far1 = mdr_at_far(targets, probabilities, target_far=0.01)
        mdr5, far5 = mdr_at_far(targets, probabilities, target_far=0.05)

        local_row = sensor_by_fold.get(fold)
        if local_row is None:
            raise KeyError(f"D={dim}: missing local sensor fold {fold}.")

        rows.append(
            {
                "dimension": int(dim),
                "fold": fold,
                "regional_auprc": float(region_row["auprc"]),
                "regional_auroc": float(region_row["auroc"]),
                "regional_mdr_at_far_1pct": mdr1,
                "regional_achieved_far_at_1pct": far1,
                "regional_mdr_at_far_5pct": mdr5,
                "regional_achieved_far_at_5pct": far5,
                "local_score_auprc": float(local_row["auprc"]),
                "local_score_auroc": float(local_row["auroc"]),
                "local_score_positive_fraction": float(
                    local_row.get("positive_fraction", np.nan)
                ),
            }
        )
    return rows


def aggregate_dimensions(fold_table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_names = (
        "regional_auprc",
        "regional_auroc",
        "regional_mdr_at_far_1pct",
        "regional_achieved_far_at_1pct",
        "regional_mdr_at_far_5pct",
        "regional_achieved_far_at_5pct",
        "local_score_auprc",
        "local_score_auroc",
    )

    for dim, group in fold_table.groupby("dimension", sort=True):
        row: dict[str, Any] = {"dimension": int(dim), "n_folds": int(len(group))}
        for metric in metric_names:
            values = group[metric].to_numpy(dtype=np.float64)
            finite = values[np.isfinite(values)]
            row[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=1))
                if finite.size >= 2
                else 0.0 if finite.size == 1 else float("nan")
            )
            row[f"{metric}_values"] = json.dumps(values.tolist())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("dimension").reset_index(drop=True)


def recommend_dimension(
    aggregate: pd.DataFrame,
    *,
    auprc_tolerance: float,
    mdr_tolerance: float,
) -> dict[str, Any]:
    best_auprc = float(aggregate["regional_auprc_mean"].max())
    best_mdr = float(aggregate["regional_mdr_at_far_1pct_mean"].min())
    auprc_cutoff = best_auprc - float(auprc_tolerance)
    mdr_cutoff = best_mdr + float(mdr_tolerance)

    eligible = aggregate[
        (aggregate["regional_auprc_mean"] >= auprc_cutoff)
        & (aggregate["regional_mdr_at_far_1pct_mean"] <= mdr_cutoff)
    ].copy()

    fallback_used = False
    if eligible.empty:
        fallback_used = True
        eligible = aggregate[
            aggregate["regional_auprc_mean"] >= auprc_cutoff
        ].copy()

    selected_dim = int(eligible["dimension"].min())
    selected_row = aggregate[aggregate["dimension"] == selected_dim].iloc[0]

    return {
        "selected_dimension": selected_dim,
        "rule": (
            "Choose the smallest dimension with mean regional AUPRC within "
            f"{auprc_tolerance:.6f} of the best and mean MDR@FAR<=1% within "
            f"{mdr_tolerance:.6f} of the best."
        ),
        "best_mean_regional_auprc": best_auprc,
        "best_auprc_dimension": int(
            aggregate.loc[aggregate["regional_auprc_mean"].idxmax(), "dimension"]
        ),
        "auprc_cutoff": auprc_cutoff,
        "best_mean_mdr_at_far_1pct": best_mdr,
        "best_mdr_dimension": int(
            aggregate.loc[
                aggregate["regional_mdr_at_far_1pct_mean"].idxmin(), "dimension"
            ]
        ),
        "mdr_cutoff": mdr_cutoff,
        "fallback_to_auprc_only": fallback_used,
        "eligible_dimensions": sorted(int(v) for v in eligible["dimension"].tolist()),
        "selected_metrics": {
            "regional_auprc_mean": float(selected_row["regional_auprc_mean"]),
            "regional_auprc_std": float(selected_row["regional_auprc_std"]),
            "regional_mdr_at_far_1pct_mean": float(
                selected_row["regional_mdr_at_far_1pct_mean"]
            ),
            "regional_mdr_at_far_1pct_std": float(
                selected_row["regional_mdr_at_far_1pct_std"]
            ),
            "local_score_auprc_mean": float(selected_row["local_score_auprc_mean"]),
        },
        "methodology_note": (
            "This is a development/ablation choice using the already-explored "
            "three outer folds. It should not be described as a fresh unbiased "
            "model-selection estimate."
        ),
    }


def plot_dimension_curve(
    aggregate: pd.DataFrame,
    *,
    value_column: str,
    std_column: str,
    ylabel: str,
    output_path: Path,
    percentage: bool = False,
) -> None:
    x = aggregate["dimension"].to_numpy(dtype=int)
    y = aggregate[value_column].to_numpy(dtype=float)
    std = aggregate[std_column].to_numpy(dtype=float)
    if percentage:
        y = 100.0 * y
        std = 100.0 * std

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(x, y, yerr=std, marker="o", capsize=4)
    ax.set_xlabel("TERRA-Feature embedding dimension, D")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    project_root = Path.cwd().resolve()
    runs_root = args.runs_root.resolve()
    study_root = args.study_root.resolve()
    config_root = study_root / "generated_configs"

    dimensions = sorted(set(int(v) for v in args.dimensions))
    if not dimensions or any(v <= 0 for v in dimensions):
        raise ValueError("--dimensions must contain positive integers.")
    if args.selection_tolerance < 0 or args.mdr_tolerance < 0:
        raise ValueError("Selection tolerances must be nonnegative.")

    base_config_path = args.base_sensor_config.resolve()
    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base sensor config not found:\n  {base_config_path}")
    base_sensor_config = load_yaml(base_config_path)

    channels = base_sensor_config.get("data", {}).get("channels", [])
    if len(channels) != 15:
        print(
            f"WARNING: base config contains {len(channels)} channels, not 15. "
            "Check that you supplied the all-channel config."
        )

    validate_region_script(project_root)
    study_root.mkdir(parents=True, exist_ok=True)
    config_root.mkdir(parents=True, exist_ok=True)

    print("TERRA-Feature dimension study")
    print("=" * 88)
    print(f"Project root:       {project_root}")
    print(f"Base sensor config: {base_config_path}")
    print(f"Dimensions:         {dimensions}")
    print(f"Study root:         {study_root}")
    print(
        f"Recommendation: AUPRC tolerance={args.selection_tolerance}, "
        f"MDR@1% tolerance={args.mdr_tolerance}"
    )

    all_fold_rows: list[dict[str, Any]] = []

    for dim in dimensions:
        print("\n" + "#" * 88)
        print(f"# EMBEDDING DIMENSION D={dim}")
        print("#" * 88)

        sensor_root = sensor_root_for(runs_root, dim)
        dataset_root = dataset_root_for(runs_root, dim)
        region_root = region_root_for(runs_root, dim)

        generated_sensor_config = config_root / f"sensor_cv_all15_d{dim}.yaml"
        generated_region_config = config_root / f"region_cv_all15_d{dim}.yaml"

        sensor_config = make_sensor_config(
            base_sensor_config, dim=dim, sensor_root=sensor_root
        )
        write_yaml(sensor_config, generated_sensor_config)
        write_yaml(
            make_region_config(
                sensor_root=sensor_root,
                sensor_config_path=generated_sensor_config,
                dataset_root=dataset_root,
                region_root=region_root,
                device=args.device,
            ),
            generated_region_config,
        )

        sensor_is_complete = sensor_complete(sensor_root, dim)
        dataset_is_complete = dataset_complete(dataset_root)
        region_is_complete = region_complete(region_root)

        if not args.rerun_existing:
            assert_no_partial_existing(
                sensor_root, complete=sensor_is_complete, stage=f"D={dim} sensor CV"
            )
            assert_no_partial_existing(
                dataset_root,
                complete=dataset_is_complete,
                stage=f"D={dim} regional dataset",
            )
            assert_no_partial_existing(
                region_root,
                complete=region_is_complete,
                stage=f"D={dim} regional CV",
            )

        sensor_cmd = [
            sys.executable,
            "scripts/run_sensor_cv.py",
            "--config",
            str(generated_sensor_config),
            "--output-root",
            str(sensor_root),
            "--device",
            args.device,
        ]
        dataset_cmd = [
            sys.executable,
            "scripts/build_region_dataset.py",
            "--config",
            str(generated_region_config),
            "--output-root",
            str(dataset_root),
            "--device",
            args.device,
        ]
        region_cmd = [
            sys.executable,
            "scripts/run_region_cv.py",
            "--config",
            str(generated_region_config),
            "--dataset-root",
            str(dataset_root),
            "--output-root",
            str(region_root),
            "--device",
            args.device,
        ]

        if args.prepare_only:
            print("\nPrepared configs:")
            print(f"  {generated_sensor_config}")
            print(f"  {generated_region_config}")
            print("\nPlanned commands:")
            print(" ".join(sensor_cmd))
            print(" ".join(dataset_cmd))
            print(" ".join(region_cmd))
            continue

        if sensor_is_complete and not args.rerun_existing:
            print(f"Reusing completed sensor run: {sensor_root}")
        else:
            run_command(sensor_cmd, cwd=project_root)

        if dataset_is_complete and not args.rerun_existing:
            print(f"Reusing completed regional dataset: {dataset_root}")
        else:
            run_command(dataset_cmd, cwd=project_root)

        if region_is_complete and not args.rerun_existing:
            print(f"Reusing completed regional run: {region_root}")
        else:
            run_command(region_cmd, cwd=project_root)

        all_fold_rows.extend(
            collect_dimension(dim=dim, sensor_root=sensor_root, region_root=region_root)
        )

    if args.prepare_only:
        elapsed = time.perf_counter() - started
        print(f"\nPreparation complete. Total runtime: {elapsed:.2f} seconds")
        return

    fold_table = pd.DataFrame(all_fold_rows).sort_values(
        ["dimension", "fold"]
    ).reset_index(drop=True)
    aggregate_table = aggregate_dimensions(fold_table)
    recommendation = recommend_dimension(
        aggregate_table,
        auprc_tolerance=args.selection_tolerance,
        mdr_tolerance=args.mdr_tolerance,
    )

    fold_path = study_root / "feature_dimension_fold_metrics.csv"
    aggregate_path = study_root / "feature_dimension_aggregate_metrics.csv"
    selection_path = study_root / "feature_dimension_selection.json"

    fold_table.to_csv(fold_path, index=False)
    aggregate_table.to_csv(aggregate_path, index=False)
    save_json(
        {
            "dimensions": dimensions,
            "selection": recommendation,
            "aggregate": aggregate_table.to_dict(orient="records"),
            "protocol": {
                "sensor_config_template": str(base_config_path),
                "regional_receiver": "learned + deepsets",
                "regional_sampling": "empirical_fixed_budget",
                "regional_checkpoint_selection": "validation_auprc",
                "full_report_count": 9,
                "primary_metric": "regional_auprc",
                "secondary_operational_metric": "regional_mdr_at_far_le_1pct",
                "selection_tolerance_auprc": args.selection_tolerance,
                "selection_tolerance_mdr_at_far_1pct": args.mdr_tolerance,
                "outer_folds_already_explored": True,
            },
        },
        selection_path,
    )

    plot_dimension_curve(
        aggregate_table,
        value_column="regional_auprc_mean",
        std_column="regional_auprc_std",
        ylabel="Regional AUPRC",
        output_path=study_root / "feature_dimension_regional_auprc.png",
    )
    plot_dimension_curve(
        aggregate_table,
        value_column="regional_mdr_at_far_1pct_mean",
        std_column="regional_mdr_at_far_1pct_std",
        ylabel="Regional MDR at FAR ≤ 1% (%)",
        output_path=study_root / "feature_dimension_low_far.png",
        percentage=True,
    )
    plot_dimension_curve(
        aggregate_table,
        value_column="local_score_auprc_mean",
        std_column="local_score_auprc_std",
        ylabel="Local TERRA-Score AUPRC",
        output_path=study_root / "feature_dimension_local_score_auprc.png",
    )

    print("\n=== Dimension ablation aggregate ===")
    display_columns = [
        "dimension",
        "regional_auprc_mean",
        "regional_auprc_std",
        "regional_mdr_at_far_1pct_mean",
        "regional_mdr_at_far_1pct_std",
        "local_score_auprc_mean",
    ]
    print(aggregate_table[display_columns].to_string(index=False))

    print("\n=== Recommendation ===")
    print(
        f"Selected D={recommendation['selected_dimension']} "
        f"from eligible dimensions {recommendation['eligible_dimensions']}."
    )
    print(recommendation["rule"])
    if recommendation["fallback_to_auprc_only"]:
        print(
            "WARNING: no dimension satisfied both criteria, so the script "
            "fell back to the AUPRC-only near-saturation rule."
        )

    print("\nSaved:")
    print(f"  {fold_path}")
    print(f"  {aggregate_path}")
    print(f"  {selection_path}")
    print(f"  {study_root / 'feature_dimension_regional_auprc.png'}")
    print(f"  {study_root / 'feature_dimension_low_far.png'}")
    print(f"  {study_root / 'feature_dimension_local_score_auprc.png'}")

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
