#!/usr/bin/env python
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fire_ura.data import (
    derive_episode_ids,
    load_sensor_table,
    make_episode_table,
)
from fire_ura.utils import load_yaml, save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe episode-level train/validation/test splits."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Overrides split.output in the YAML config.",
    )
    return parser.parse_args()


def _allocate_stratum(
    episode_ids: list[str],
    rng: np.random.Generator,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[str]]:
    shuffled = list(episode_ids)
    rng.shuffle(shuffled)
    n = len(shuffled)

    if n == 1:
        return {"train": shuffled, "val": [], "test": []}

    n_test = int(round(n * test_fraction))
    n_val = int(round(n * val_fraction))

    if test_fraction > 0 and n >= 2:
        n_test = max(1, n_test)
    if val_fraction > 0 and n >= 3:
        n_val = max(1, n_val)

    while n_test + n_val > n - 1:
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
        else:
            break

    test_ids = shuffled[:n_test]
    val_ids = shuffled[n_test:n_test + n_val]
    train_ids = shuffled[n_test + n_val:]
    return {"train": train_ids, "val": val_ids, "test": test_ids}


def grouped_random_split(
    episodes: pd.DataFrame,
    *,
    seed: int,
    val_fraction: float,
    test_fraction: float,
) -> dict[str, list[str]]:
    if val_fraction < 0 or test_fraction < 0 or val_fraction + test_fraction >= 1:
        raise ValueError("Require val_fraction >= 0, test_fraction >= 0, and sum < 1.")

    rng = np.random.default_rng(seed)
    split_ids = {"train": [], "val": [], "test": []}

    for _, group in episodes.groupby("stratum", sort=True):
        allocation = _allocate_stratum(
            group["episode_id"].astype(str).tolist(),
            rng,
            val_fraction,
            test_fraction,
        )
        for split_name in split_ids:
            split_ids[split_name].extend(allocation[split_name])

    for split_name in split_ids:
        split_ids[split_name] = sorted(split_ids[split_name])
    return split_ids


def leave_one_repetition_out_split(
    episodes: pd.DataFrame,
    *,
    test_repetition: str,
    val_repetition: str | None,
) -> dict[str, list[str]]:
    repetition = episodes["repetition_id"].astype(str)
    test_mask = repetition == str(test_repetition)
    val_mask = (
        repetition == str(val_repetition)
        if val_repetition is not None
        else pd.Series(False, index=episodes.index)
    )

    if not test_mask.any():
        raise ValueError(f"No episodes have repetition_id={test_repetition!r}.")
    if val_repetition is not None and not val_mask.any():
        raise ValueError(f"No episodes have repetition_id={val_repetition!r}.")
    if (test_mask & val_mask).any():
        raise ValueError("Test and validation repetition IDs must differ.")

    train_mask = ~(test_mask | val_mask)
    return {
        "train": sorted(episodes.loc[train_mask, "episode_id"].astype(str).tolist()),
        "val": sorted(episodes.loc[val_mask, "episode_id"].astype(str).tolist()),
        "test": sorted(episodes.loc[test_mask, "episode_id"].astype(str).tolist()),
    }


def summarise_split(
    episodes: pd.DataFrame,
    episode_ids: list[str],
) -> dict[str, Any]:
    selected = episodes[episodes["episode_id"].astype(str).isin(episode_ids)]
    return {
        "n_episodes": int(len(selected)),
        "n_fire_episodes": int(selected["fire_label"].sum()),
        "n_nonfire_episodes": int((selected["fire_label"] == 0).sum()),
        "scenario_counts": {
            str(key): int(value)
            for key, value in Counter(selected["scenario_label"].astype(str)).items()
        },
        "n_rows": int(selected["n_rows"].sum()),
    }


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)

    data_config = config["data"]
    split_config = config["split"]

    frame, inferred_schema = load_sensor_table(
        data_config["paths"],
        channels=data_config.get(
            "channels", ["temperature", "humidity", "co", "co2"]
        ),
        column_overrides=data_config.get("column_overrides", {}),
        fire_positive_values=data_config.get(
            "fire_positive_values",
            ["fire", "1", "true", "yes", "fire_relevant"],
        ),
        delimiter=data_config.get("delimiter", "auto"),
        encoding=data_config.get("encoding", "utf-8"),
    )

    episode_config = data_config.get("episode", {})
    frame = derive_episode_ids(
        frame,
        boundary_columns=episode_config.get(
            "boundary_columns", ["scenario_label"]
        ),
        max_gap_seconds=episode_config.get("max_gap_seconds", 30.0),
        overwrite=episode_config.get("overwrite_existing", False),
    )
    episodes = make_episode_table(frame)

    strategy = split_config.get("strategy", "grouped_random")
    if strategy == "grouped_random":
        split_ids = grouped_random_split(
            episodes,
            seed=int(split_config.get("seed", 42)),
            val_fraction=float(split_config.get("val_fraction", 0.15)),
            test_fraction=float(split_config.get("test_fraction", 0.15)),
        )
    elif strategy == "leave_one_repetition_out":
        split_ids = leave_one_repetition_out_split(
            episodes,
            test_repetition=str(split_config["test_repetition"]),
            val_repetition=(
                str(split_config["val_repetition"])
                if split_config.get("val_repetition") is not None
                else None
            ),
        )
    else:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. "
            "Use grouped_random or leave_one_repetition_out."
        )

    output_path = args.output or Path(split_config.get("output", "data/splits/fold_0.json"))
    payload = {
        "config_path": str(args.config),
        "strategy": strategy,
        "schema": inferred_schema.as_dict(),
        "episode_derivation": episode_config,
        "splits": split_ids,
        "summary": {
            name: summarise_split(episodes, ids)
            for name, ids in split_ids.items()
        },
    }
    save_json(payload, output_path)

    episode_table_path = output_path.with_name(output_path.stem + "_episodes.csv")
    episode_table_path.parent.mkdir(parents=True, exist_ok=True)
    episodes.to_csv(episode_table_path, index=False)

    print(f"Wrote split manifest: {output_path}")
    print(f"Wrote episode table:  {episode_table_path}")
    for split_name, summary in payload["summary"].items():
        print(
            f"{split_name:>5}: {summary['n_episodes']} episodes, "
            f"{summary['n_fire_episodes']} fire, "
            f"{summary['n_nonfire_episodes']} non-fire"
        )


if __name__ == "__main__":
    main()
