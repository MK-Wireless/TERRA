#!/usr/bin/env python
from __future__ import annotations

"""
Diagnose the information content of the selected 1-bit handcrafted report.

The selected handcrafted benchmark contains, for each sensor channel:
    mean, std

Before quantization, the frozen regional receiver standardizes each component:
    z_j = (x_j - mu_j) / sigma_j

For the symmetric 1-bit mid-rise quantizer on [-4, 4], the transmitted bit is
therefore exactly:
    bit_j = 1[z_j >= 0] = 1[x_j >= mu_j]

and the reconstruction is -2 or +2.

This script checks whether those 30 transmitted bits are genuinely informative
or nearly constant. It reports, per fold and per component:

  * raw positive/zero fractions
  * transmitted-one fraction
  * bit entropy
  * P(bit=1 | fire)
  * P(bit=1 | no-fire)
  * absolute class-conditional gap
  * fraction of all-zero and all-one 30-bit reports
  * number/effective number of unique 30-bit report patterns
  * within-region diversity of the nine sensor reports
  * expected Hamming separation between fire and no-fire reports

No model is trained and no receiver inference is needed. The script reads only
the existing regional dataset and frozen handcrafted receiver checkpoint.

Typical run
-----------
python scripts/diagnose_handcrafted_1bit.py

Optional:
python scripts/diagnose_handcrafted_1bit.py ^
  --split test ^
  --output-root runs/handcrafted_1bit_diagnostic

Outputs
-------
runs/handcrafted_1bit_diagnostic/
    handcrafted_1bit_fold_summary.csv
    handcrafted_1bit_component_stats.csv
    handcrafted_1bit_top_patterns.csv
    handcrafted_1bit_summary.json

Every executable path reports total runtime.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

from fire_ura.region_data import RegionArrays, load_region_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose information content of 1-bit handcrafted reports."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("runs/region_dataset_all15_d35"),
        help="Regional dataset containing the full handcrafted feature vector.",
    )
    parser.add_argument(
        "--receiver-root",
        type=Path,
        default=Path(
            "runs/handcrafted_stat_ablation/receiver_runs/nested_h2"
        ),
        help="Selected H2 handcrafted+DeepSets receiver root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("runs/handcrafted_1bit_diagnostic"),
    )
    parser.add_argument(
        "--split",
        choices=("dev", "test", "both"),
        default="test",
        help="Which outer split to diagnose.",
    )
    parser.add_argument(
        "--stats",
        nargs="+",
        default=["mean", "std"],
        help="Handcrafted statistics to retain. Default: mean std.",
    )
    parser.add_argument(
        "--top-patterns",
        type=int,
        default=20,
        help="Number of most frequent binary report patterns saved per fold.",
    )
    parser.add_argument(
        "--near-constant-threshold",
        type=float,
        default=0.01,
        help=(
            "A bit is called near-constant if P(1) <= t or P(1) >= 1-t. "
            "Default: 0.01."
        ),
    )
    return parser.parse_args()


def save_json(payload: Any, path: Path) -> None:
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

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(convert(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def read_feature_names(dataset_root: Path, fold: int) -> list[str]:
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
    return [str(name) for name in names]


def select_handcrafted_indices(
    feature_names: Sequence[str],
    requested_stats: Sequence[str],
) -> np.ndarray:
    wanted = {str(stat) for stat in requested_stats}
    indices: list[int] = []

    for index, name in enumerate(feature_names):
        if "__" not in name:
            raise ValueError(
                f"Unexpected handcrafted feature name {name!r}; "
                "expected channel__stat."
            )
        stat = name.rsplit("__", 1)[1]
        if stat in wanted:
            indices.append(index)

    found = {
        feature_names[index].rsplit("__", 1)[1]
        for index in indices
    }
    if found != wanted:
        raise ValueError(
            f"Requested stats {sorted(wanted)}, but found {sorted(found)}."
        )
    if not indices:
        raise ValueError("No handcrafted dimensions selected.")

    return np.asarray(indices, dtype=np.int64)


def checkpoint_path(receiver_root: Path, fold: int) -> Path:
    return (
        receiver_root
        / f"fold_{fold}"
        / "handcrafted"
        / "deepsets"
        / "final"
        / "final_model.pt"
    )


def load_standardizer(
    receiver_root: Path,
    *,
    fold: int,
    expected_dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    path = checkpoint_path(receiver_root, fold)
    if not path.is_file():
        raise FileNotFoundError(path)

    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if str(payload.get("representation")) != "handcrafted":
        raise ValueError(
            f"{path}: expected handcrafted checkpoint, found "
            f"{payload.get('representation')!r}."
        )

    mean = np.asarray(payload["feature_mean"], dtype=np.float64)
    std = np.asarray(payload["feature_std"], dtype=np.float64)

    if mean.shape != (expected_dimension,):
        raise ValueError(
            f"{path}: feature_mean shape {mean.shape}, expected "
            f"({expected_dimension},)."
        )
    if std.shape != (expected_dimension,):
        raise ValueError(
            f"{path}: feature_std shape {std.shape}, expected "
            f"({expected_dimension},)."
        )
    if np.any(std <= 0) or np.any(~np.isfinite(std)):
        raise ValueError(f"{path}: invalid standard deviations.")

    return mean, std


def split_indices(arrays: RegionArrays, split: str) -> np.ndarray:
    if split == "dev":
        return np.flatnonzero(arrays.split_code == 0)
    if split == "test":
        return np.flatnonzero(arrays.split_code == 1)
    if split == "both":
        return np.arange(len(arrays.target), dtype=np.int64)
    raise ValueError(split)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=np.float64)
    output = np.zeros_like(p)
    mask = (p > 0.0) & (p < 1.0)
    output[mask] = -(
        p[mask] * np.log2(p[mask])
        + (1.0 - p[mask]) * np.log2(1.0 - p[mask])
    )
    return output


def encode_bitstrings(bits: np.ndarray) -> np.ndarray:
    return np.asarray(
        ["".join("1" if value else "0" for value in row) for row in bits],
        dtype=object,
    )


def expected_cross_class_hamming(
    p_fire: np.ndarray,
    p_no_fire: np.ndarray,
) -> float:
    """Expected Hamming distance between independent fire/no-fire reports."""
    per_bit = (
        p_fire * (1.0 - p_no_fire)
        + (1.0 - p_fire) * p_no_fire
    )
    return float(np.sum(per_bit))


def expected_within_class_hamming(p: np.ndarray) -> float:
    """Expected Hamming distance between two independent reports in one class."""
    return float(np.sum(2.0 * p * (1.0 - p)))


def effective_pattern_count(counts: np.ndarray) -> float:
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    entropy = -np.sum(
        probabilities[probabilities > 0]
        * np.log2(probabilities[probabilities > 0])
    )
    return float(2.0 ** entropy)


def within_region_diversity(
    bits_by_example: np.ndarray,
    valid_mask: np.ndarray,
) -> dict[str, float]:
    unique_counts: list[int] = []
    pairwise_hamming: list[float] = []

    for example_index in range(bits_by_example.shape[0]):
        rows = bits_by_example[example_index][valid_mask[example_index]]
        if len(rows) == 0:
            continue

        unique_counts.append(
            int(np.unique(rows, axis=0).shape[0])
        )

        if len(rows) >= 2:
            distances: list[float] = []
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    distances.append(
                        float(np.mean(rows[i] != rows[j]))
                    )
            pairwise_hamming.append(float(np.mean(distances)))

    if not unique_counts:
        return {
            "mean_unique_patterns_per_region": float("nan"),
            "median_unique_patterns_per_region": float("nan"),
            "fraction_regions_single_pattern": float("nan"),
            "mean_within_region_normalized_hamming": float("nan"),
        }

    return {
        "mean_unique_patterns_per_region": float(np.mean(unique_counts)),
        "median_unique_patterns_per_region": float(np.median(unique_counts)),
        "fraction_regions_single_pattern": float(
            np.mean(np.asarray(unique_counts) == 1)
        ),
        "mean_within_region_normalized_hamming": (
            float(np.mean(pairwise_hamming))
            if pairwise_hamming
            else float("nan")
        ),
    }


def diagnose_fold(
    *,
    fold: int,
    arrays: RegionArrays,
    feature_names: Sequence[str],
    selected_indices: np.ndarray,
    receiver_root: Path,
    split: str,
    near_constant_threshold: float,
    top_patterns: int,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    example_indices = split_indices(arrays, split)

    selected_names = [
        feature_names[int(index)]
        for index in selected_indices
    ]
    dimension = len(selected_names)

    feature_mean, feature_std = load_standardizer(
        receiver_root,
        fold=fold,
        expected_dimension=dimension,
    )

    raw = arrays.handcrafted[
        example_indices
    ][:, :, selected_indices].astype(np.float64)

    valid_mask = arrays.mask[example_indices].astype(bool)
    targets_by_example = arrays.target[example_indices].astype(np.int64)

    # Flatten only valid reports.
    raw_valid = raw[valid_mask]
    report_targets = np.repeat(
        targets_by_example[:, None],
        valid_mask.shape[1],
        axis=1,
    )[valid_mask]

    standardized_valid = (
        raw_valid - feature_mean[None, :]
    ) / feature_std[None, :]

    # Exactly equivalent to 1-bit symmetric mid-rise quantization on [-4,4].
    bits_valid = standardized_valid >= 0.0

    # Keep the per-example structure for regional diversity checks.
    bits_by_example = np.zeros_like(raw, dtype=bool)
    standardized_all = (
        raw - feature_mean[None, None, :]
    ) / feature_std[None, None, :]
    bits_by_example[valid_mask] = standardized_all[valid_mask] >= 0.0

    if not np.any(report_targets == 1):
        raise ValueError(f"Fold {fold}: no fire reports in selected split.")
    if not np.any(report_targets == 0):
        raise ValueError(f"Fold {fold}: no no-fire reports in selected split.")

    p_one = bits_valid.mean(axis=0)
    p_one_fire = bits_valid[report_targets == 1].mean(axis=0)
    p_one_no_fire = bits_valid[report_targets == 0].mean(axis=0)
    entropy = binary_entropy(p_one)

    raw_positive = (raw_valid > 0.0).mean(axis=0)
    raw_zero = np.isclose(raw_valid, 0.0, atol=1e-12).mean(axis=0)

    component_rows: list[dict[str, Any]] = []
    for index, name in enumerate(selected_names):
        channel, stat = name.rsplit("__", 1)
        component_rows.append(
            {
                "fold": fold,
                "split": split,
                "component_index": index,
                "feature_name": name,
                "channel": channel,
                "statistic": stat,
                "development_mean": float(feature_mean[index]),
                "development_std": float(feature_std[index]),
                "raw_positive_fraction": float(raw_positive[index]),
                "raw_zero_fraction": float(raw_zero[index]),
                "transmitted_one_fraction": float(p_one[index]),
                "bit_entropy_bits": float(entropy[index]),
                "p_one_given_fire": float(p_one_fire[index]),
                "p_one_given_no_fire": float(p_one_no_fire[index]),
                "absolute_class_gap": float(
                    abs(p_one_fire[index] - p_one_no_fire[index])
                ),
                "near_constant": bool(
                    p_one[index] <= near_constant_threshold
                    or p_one[index] >= 1.0 - near_constant_threshold
                ),
            }
        )

    bitstrings = encode_bitstrings(bits_valid)
    unique_patterns, counts = np.unique(
        bitstrings,
        return_counts=True,
    )
    order = np.argsort(counts)[::-1]

    pattern_rows: list[dict[str, Any]] = []
    for rank, position in enumerate(order[:top_patterns], start=1):
        pattern = str(unique_patterns[position])
        count = int(counts[position])
        pattern_mask = bitstrings == pattern
        pattern_rows.append(
            {
                "fold": fold,
                "split": split,
                "rank": rank,
                "bit_pattern": pattern,
                "count": count,
                "fraction": float(count / len(bitstrings)),
                "fire_fraction_within_pattern": float(
                    report_targets[pattern_mask].mean()
                ),
            }
        )

    all_zero = np.all(~bits_valid, axis=1)
    all_one = np.all(bits_valid, axis=1)

    diversity = within_region_diversity(
        bits_by_example,
        valid_mask,
    )

    summary = {
        "fold": fold,
        "split": split,
        "n_examples": int(len(example_indices)),
        "n_valid_reports": int(len(bits_valid)),
        "positive_example_fraction": float(targets_by_example.mean()),
        "positive_report_fraction": float(report_targets.mean()),
        "dimension": int(dimension),
        "mean_transmitted_one_fraction": float(np.mean(p_one)),
        "min_transmitted_one_fraction": float(np.min(p_one)),
        "max_transmitted_one_fraction": float(np.max(p_one)),
        "mean_bit_entropy_bits": float(np.mean(entropy)),
        "min_bit_entropy_bits": float(np.min(entropy)),
        "number_near_constant_bits": int(
            np.sum(
                (p_one <= near_constant_threshold)
                | (p_one >= 1.0 - near_constant_threshold)
            )
        ),
        "fraction_all_zero_reports": float(np.mean(all_zero)),
        "fraction_all_one_reports": float(np.mean(all_one)),
        "number_unique_report_patterns": int(len(unique_patterns)),
        "effective_number_report_patterns": effective_pattern_count(counts),
        "most_common_pattern_fraction": float(
            np.max(counts) / len(bitstrings)
        ),
        "mean_absolute_class_gap": float(
            np.mean(np.abs(p_one_fire - p_one_no_fire))
        ),
        "max_absolute_class_gap": float(
            np.max(np.abs(p_one_fire - p_one_no_fire))
        ),
        "expected_fire_vs_no_fire_hamming_bits": (
            expected_cross_class_hamming(p_one_fire, p_one_no_fire)
        ),
        "expected_fire_within_class_hamming_bits": (
            expected_within_class_hamming(p_one_fire)
        ),
        "expected_no_fire_within_class_hamming_bits": (
            expected_within_class_hamming(p_one_no_fire)
        ),
        **diversity,
    }

    return summary, component_rows, pattern_rows


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    if not (0.0 <= args.near_constant_threshold < 0.5):
        raise ValueError(
            "--near-constant-threshold must be in [0, 0.5)."
        )
    if args.top_patterns <= 0:
        raise ValueError("--top-patterns must be positive.")

    dataset_root = args.dataset_root.resolve()
    receiver_root = args.receiver_root.resolve()
    output_root = args.output_root.resolve()

    print("Handcrafted 1-bit information diagnostic")
    print("=" * 88)
    print(f"Dataset root:       {dataset_root}")
    print(f"Receiver root:      {receiver_root}")
    print(f"Output root:        {output_root}")
    print(f"Split:              {args.split}")
    print(f"Selected stats:     {args.stats}")
    print(
        "1-bit rule:         bit_j = 1[(x_j - mu_j)/sigma_j >= 0] "
        "= 1[x_j >= mu_j]"
    )
    print(
        "Mid-rise levels:    {-2,+2} for common clipping interval [-4,4]"
    )

    fold_summaries: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    pattern_rows: list[dict[str, Any]] = []

    for fold in range(3):
        npz_path = (
            dataset_root
            / f"fold_{fold}"
            / f"region_dataset_fold{fold}.npz"
        )
        manifest_path = (
            dataset_root
            / f"fold_{fold}"
            / f"region_manifest_fold{fold}.csv"
        )
        arrays = load_region_arrays(
            npz_path,
            manifest_path,
        )

        feature_names = read_feature_names(
            dataset_root,
            fold,
        )
        selected_indices = select_handcrafted_indices(
            feature_names,
            args.stats,
        )

        summary, fold_components, fold_patterns = diagnose_fold(
            fold=fold,
            arrays=arrays,
            feature_names=feature_names,
            selected_indices=selected_indices,
            receiver_root=receiver_root,
            split=args.split,
            near_constant_threshold=args.near_constant_threshold,
            top_patterns=args.top_patterns,
        )

        fold_summaries.append(summary)
        component_rows.extend(fold_components)
        pattern_rows.extend(fold_patterns)

        print(f"\n=== Fold {fold} ===")
        print(
            f"valid reports:                {summary['n_valid_reports']}"
        )
        print(
            f"mean P(bit=1):                "
            f"{summary['mean_transmitted_one_fraction']:.4f}"
        )
        print(
            f"P(bit=1) range:               "
            f"[{summary['min_transmitted_one_fraction']:.4f}, "
            f"{summary['max_transmitted_one_fraction']:.4f}]"
        )
        print(
            f"mean bit entropy:             "
            f"{summary['mean_bit_entropy_bits']:.4f} bits"
        )
        print(
            f"near-constant bits:           "
            f"{summary['number_near_constant_bits']}/{summary['dimension']}"
        )
        print(
            f"all-zero reports:             "
            f"{100*summary['fraction_all_zero_reports']:.4f}%"
        )
        print(
            f"all-one reports:              "
            f"{100*summary['fraction_all_one_reports']:.4f}%"
        )
        print(
            f"unique 30-bit patterns:       "
            f"{summary['number_unique_report_patterns']}"
        )
        print(
            f"effective pattern count:      "
            f"{summary['effective_number_report_patterns']:.2f}"
        )
        print(
            f"most common pattern fraction: "
            f"{100*summary['most_common_pattern_fraction']:.2f}%"
        )
        print(
            f"mean |P1(fire)-P1(no-fire)|:  "
            f"{summary['mean_absolute_class_gap']:.4f}"
        )
        print(
            f"expected fire/no-fire Hamming:"
            f" {summary['expected_fire_vs_no_fire_hamming_bits']:.2f}/"
            f"{summary['dimension']} bits"
        )
        print(
            f"mean unique patterns/region:  "
            f"{summary['mean_unique_patterns_per_region']:.2f}"
        )
        print(
            f"single-pattern regions:       "
            f"{100*summary['fraction_regions_single_pattern']:.2f}%"
        )

    output_root.mkdir(parents=True, exist_ok=True)

    fold_table = pd.DataFrame(fold_summaries)
    component_table = pd.DataFrame(component_rows)
    pattern_table = pd.DataFrame(pattern_rows)

    fold_path = output_root / "handcrafted_1bit_fold_summary.csv"
    component_path = (
        output_root / "handcrafted_1bit_component_stats.csv"
    )
    pattern_path = (
        output_root / "handcrafted_1bit_top_patterns.csv"
    )
    json_path = output_root / "handcrafted_1bit_summary.json"

    fold_table.to_csv(fold_path, index=False)
    component_table.to_csv(component_path, index=False)
    pattern_table.to_csv(pattern_path, index=False)

    aggregate = {
        "split": args.split,
        "selected_stats": list(args.stats),
        "one_bit_interpretation": (
            "After receiver-standardization, symmetric 1-bit mid-rise "
            "quantization transmits one bit per element indicating whether "
            "the raw handcrafted statistic is below or above its development-"
            "set mean. The reconstructed standardized values are -2 and +2."
        ),
        "folds": fold_summaries,
        "aggregate_across_folds": {
            column: float(fold_table[column].mean())
            for column in [
                "mean_transmitted_one_fraction",
                "mean_bit_entropy_bits",
                "fraction_all_zero_reports",
                "fraction_all_one_reports",
                "number_unique_report_patterns",
                "effective_number_report_patterns",
                "most_common_pattern_fraction",
                "mean_absolute_class_gap",
                "expected_fire_vs_no_fire_hamming_bits",
                "mean_unique_patterns_per_region",
                "fraction_regions_single_pattern",
                "mean_within_region_normalized_hamming",
            ]
        },
    }
    save_json(aggregate, json_path)

    print("\n=== Most class-discriminative bits by fold ===")
    display_columns = [
        "fold",
        "feature_name",
        "raw_positive_fraction",
        "transmitted_one_fraction",
        "p_one_given_fire",
        "p_one_given_no_fire",
        "absolute_class_gap",
        "bit_entropy_bits",
    ]
    for fold in range(3):
        top = (
            component_table[component_table["fold"] == fold]
            .sort_values("absolute_class_gap", ascending=False)
            .head(10)
        )
        print(f"\nFold {fold}")
        print(top[display_columns].to_string(index=False))

    print("\nSaved:")
    print(f"  {fold_path}")
    print(f"  {component_path}")
    print(f"  {pattern_path}")
    print(f"  {json_path}")

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
