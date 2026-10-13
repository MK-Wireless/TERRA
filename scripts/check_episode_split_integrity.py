#!/usr/bin/env python
"""Validate episode-level outer and inner split manifests from a sensor-CV run."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any


def infer_fold_number(path: Path) -> int:
    match = re.search(r"fold[_-]?(\d+)", path.parent.name.lower())
    if match is None:
        raise ValueError(f"Could not infer fold number from: {path}")
    return int(match.group(1))


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object.")
    return payload


def read_id_set(payload: dict[str, Any], key: str, path: Path) -> set[str]:
    values = payload.get(key)
    if not isinstance(values, list):
        raise ValueError(f"{path}: expected '{key}' to be a list.")

    ids = [str(value) for value in values]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}: '{key}' contains duplicate episode IDs.")

    return set(ids)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Check outer development/test and inner train/validation "
            "episode assignments in a completed sensor-CV run."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        required=True,
        help="Completed sensor-CV run directory.",
    )
    parser.add_argument(
        "--skip-inner",
        action="store_true",
        help="Check only outer development/test assignments.",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    errors: list[str] = []

    outer_paths = sorted(
        args.run_root.glob("fold_*/outer_manifest.json"),
        key=infer_fold_number,
    )

    if not outer_paths:
        raise FileNotFoundError(
            f"No fold_*/outer_manifest.json files were found under "
            f"{args.run_root.resolve()}"
        )

    records: list[dict[str, Any]] = []

    print("\nEpisode-level split integrity check")
    print("=" * 76)

    for outer_path in outer_paths:
        fold = infer_fold_number(outer_path)
        outer = load_json(outer_path)

        development = read_id_set(outer, "development", outer_path)
        test = read_id_set(outer, "test", outer_path)
        overlap = development & test
        universe = development | test

        if overlap:
            errors.append(
                f"Fold {fold}: development/test overlap contains "
                f"{len(overlap)} episode(s): {sorted(overlap)[:10]}"
            )

        records.append(
            {
                "fold": fold,
                "development": development,
                "test": test,
                "universe": universe,
            }
        )

        print(
            f"Fold {fold}: development={len(development)}, "
            f"test={len(test)}, total={len(universe)}, "
            f"outer overlap={len(overlap)}"
        )

        if args.skip_inner:
            continue

        for direction in (0, 1):
            inner_path = (
                args.run_root
                / f"fold_{fold}"
                / f"inner_{direction}_manifest.json"
            )

            if not inner_path.exists():
                errors.append(f"Missing inner manifest: {inner_path}")
                continue

            inner = load_json(inner_path)
            train = read_id_set(inner, "train", inner_path)
            val = read_id_set(inner, "val", inner_path)

            inner_overlap = train & val
            outside = (train | val) - development
            omitted = development - (train | val)

            if inner_overlap:
                errors.append(
                    f"Fold {fold}, inner {direction}: train/validation overlap "
                    f"contains {len(inner_overlap)} episode(s): "
                    f"{sorted(inner_overlap)[:10]}"
                )

            if outside:
                errors.append(
                    f"Fold {fold}, inner {direction}: {len(outside)} episode(s) "
                    f"fall outside the outer development set: "
                    f"{sorted(outside)[:10]}"
                )

            if omitted:
                errors.append(
                    f"Fold {fold}, inner {direction}: {len(omitted)} outer "
                    f"development episode(s) are omitted: "
                    f"{sorted(omitted)[:10]}"
                )

            print(
                f"  inner {direction}: train={len(train)}, val={len(val)}, "
                f"overlap={len(inner_overlap)}, outside={len(outside)}, "
                f"omitted={len(omitted)}"
            )

    reference_universe = records[0]["universe"]

    for record in records[1:]:
        missing = reference_universe - record["universe"]
        extra = record["universe"] - reference_universe

        if missing or extra:
            errors.append(
                f"Fold {record['fold']} has a different episode universe. "
                f"Missing={sorted(missing)[:10]}, "
                f"extra={sorted(extra)[:10]}"
            )

    test_counts: Counter[str] = Counter()
    development_counts: Counter[str] = Counter()

    for record in records:
        test_counts.update(record["test"])
        development_counts.update(record["development"])

    number_of_folds = len(records)

    bad_test_counts = {
        episode: test_counts.get(episode, 0)
        for episode in reference_universe
        if test_counts.get(episode, 0) != 1
    }

    expected_development_count = number_of_folds - 1
    bad_development_counts = {
        episode: development_counts.get(episode, 0)
        for episode in reference_universe
        if development_counts.get(episode, 0)
        != expected_development_count
    }

    if bad_test_counts:
        errors.append(
            f"{len(bad_test_counts)} episode(s) do not appear in exactly one "
            f"outer test fold. Examples: "
            f"{list(sorted(bad_test_counts.items()))[:10]}"
        )

    if bad_development_counts:
        errors.append(
            f"{len(bad_development_counts)} episode(s) do not appear in "
            f"development exactly {expected_development_count} time(s). "
            f"Examples: "
            f"{list(sorted(bad_development_counts.items()))[:10]}"
        )

    print("-" * 76)
    print(f"Run root:               {args.run_root.resolve()}")
    print(f"Outer folds checked:    {number_of_folds}")
    print(f"Unique episodes:        {len(reference_universe)}")
    print(f"Bad test counts:        {len(bad_test_counts)}")
    print(f"Bad development counts: {len(bad_development_counts)}")

    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"  - {error}")
        return_code = 1
    else:
        print(
            "\nPASSED: no outer development/test or inner "
            "train/validation episode leakage was found."
        )
        return_code = 0

    elapsed = time.perf_counter() - started
    print(f"Total runtime: {elapsed:.2f} seconds")
    sys.exit(return_code)


if __name__ == "__main__":
    main()
