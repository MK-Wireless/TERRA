#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fire_ura.data import FireWindowDataset, derive_episode_ids, load_sensor_table, make_episode_table
from fire_ura.region_data import RegionArrays, save_region_arrays
from fire_ura.representation import (
    FoldSensorProjector,
    handcrafted_feature_names,
    handcrafted_temporal_features,
)
from fire_ura.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build fold-specific regional set datasets from frozen sensor encoders."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
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
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def resolve_sensor_config(run_root: Path, configured: str | Path | None) -> Path:
    if configured is not None:
        path = Path(configured)
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    candidates = sorted(run_root.glob("config_*.yaml")) + sorted(run_root.glob("config_*.yml"))
    if len(candidates) != 1:
        raise FileNotFoundError(
            "Could not uniquely infer the copied sensor config. Set sensor_config_path "
            f"in the regional YAML. Candidates: {candidates}"
        )
    return candidates[0]


def window_target_mode(window_config: Mapping[str, Any]) -> str:
    return str(window_config.get("target_mode", "last"))


def make_window_loader(
    frame: pd.DataFrame,
    *,
    episode_ids: Sequence[str],
    projector: FoldSensorProjector,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    dataset = FireWindowDataset(
        frame,
        episode_ids=episode_ids,
        channels=projector.channels,
        window_length=int(projector.window_config.get("length", 12)),
        stride=int(projector.window_config.get("stride", 12)),
        normalization=projector.normalization,
        target_mode=window_target_mode(projector.window_config),
        target_column="fire_label",
        excluded_progress_labels=(),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        drop_last=False,
    )


def mode_text(values: Sequence[str]) -> str:
    if not values:
        return ""
    counts = Counter(str(value) for value in values)
    maximum = max(counts.values())
    return sorted(value for value, count in counts.items() if count == maximum)[0]


def collect_sensor_windows(
    loader: DataLoader,
    *,
    projector: FoldSensorProjector,
    split_name: str,
    episode_repetition: Mapping[str, str],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].to(torch.float32)
        embeddings, probabilities = projector.project(x)
        handcrafted = handcrafted_temporal_features(x).cpu().numpy().astype(np.float32)
        embeddings_np = embeddings.detach().cpu().numpy().astype(np.float32)
        if probabilities is None:
            raise FileNotFoundError(
                "The fold classifier checkpoint was not found. It is required for the "
                "average-local-probability receiver baseline."
            )
        probabilities_np = probabilities.detach().cpu().numpy().astype(np.float32)
        targets = batch["label"].cpu().numpy().astype(np.int64)

        batch_size = int(targets.shape[0])
        for index in range(batch_size):
            episode_id = str(batch["episode_id"][index])
            records.append(
                {
                    "split": split_name,
                    "source_file": str(batch["source_file"][index]),
                    "episode_id": episode_id,
                    "repetition_id": str(episode_repetition.get(episode_id, "")),
                    "sensor_id": str(batch["sensor_id"][index]),
                    "scenario_label": str(batch["scenario_label"][index]),
                    "progress_label": str(batch["progress_label"][index]),
                    "start_timestamp": str(batch["start_timestamp"][index]),
                    "end_timestamp": str(batch["end_timestamp"][index]),
                    "target": int(targets[index]),
                    "embedding": embeddings_np[index],
                    "handcrafted": handcrafted[index],
                    "local_probability": float(probabilities_np[index]),
                }
            )
    if not records:
        raise RuntimeError(f"No sensor windows were generated for split {split_name!r}.")
    return records


def group_region_examples(
    records: Sequence[dict[str, Any]],
    *,
    expected_max_sensors: int | None,
) -> RegionArrays:
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        key = (
            str(record["split"]),
            str(record["episode_id"]),
            str(record["start_timestamp"]),
            str(record["end_timestamp"]),
        )
        groups[key].append(record)

    ordered_keys = sorted(groups)
    maximum_set_size = max(len(groups[key]) for key in ordered_keys)
    if expected_max_sensors is not None and maximum_set_size > expected_max_sensors:
        raise ValueError(
            f"Observed a set of {maximum_set_size} sensors, exceeding configured "
            f"expected_max_sensors={expected_max_sensors}."
        )
    padded_size = expected_max_sensors or maximum_set_size
    first_record = records[0]
    learned_dim = int(np.asarray(first_record["embedding"]).size)
    handcrafted_dim = int(np.asarray(first_record["handcrafted"]).size)
    n_examples = len(ordered_keys)

    learned = np.zeros((n_examples, padded_size, learned_dim), dtype=np.float32)
    handcrafted = np.zeros((n_examples, padded_size, handcrafted_dim), dtype=np.float32)
    local_probability = np.zeros((n_examples, padded_size), dtype=np.float32)
    mask = np.zeros((n_examples, padded_size), dtype=bool)
    target = np.zeros(n_examples, dtype=np.int64)
    split_code = np.zeros(n_examples, dtype=np.int8)
    metadata_rows: list[dict[str, Any]] = []

    for example_index, key in enumerate(ordered_keys):
        group = sorted(groups[key], key=lambda item: str(item["sensor_id"]))
        sensor_ids = [str(item["sensor_id"]) for item in group]
        if len(sensor_ids) != len(set(sensor_ids)):
            duplicates = sorted(
                sensor_id for sensor_id, count in Counter(sensor_ids).items() if count > 1
            )
            raise ValueError(f"Duplicate sensors in regional group {key}: {duplicates}")
        targets = {int(item["target"]) for item in group}
        if len(targets) != 1:
            raise ValueError(f"Inconsistent regional labels in group {key}: {targets}")
        scenarios = {str(item["scenario_label"]) for item in group}
        if len(scenarios) != 1:
            raise ValueError(f"Inconsistent scenarios in group {key}: {scenarios}")

        count = len(group)
        learned[example_index, :count] = np.stack([item["embedding"] for item in group])
        handcrafted[example_index, :count] = np.stack(
            [item["handcrafted"] for item in group]
        )
        local_probability[example_index, :count] = np.asarray(
            [item["local_probability"] for item in group], dtype=np.float32
        )
        mask[example_index, :count] = True
        target[example_index] = targets.pop()
        split_code[example_index] = 0 if key[0] == "development" else 1
        metadata_rows.append(
            {
                "example_index": example_index,
                "split": key[0],
                "source_file": str(group[0]["source_file"]),
                "episode_id": key[1],
                "repetition_id": str(group[0]["repetition_id"]),
                "scenario_label": next(iter(scenarios)),
                "progress_label": mode_text([str(item["progress_label"]) for item in group]),
                "start_timestamp": key[2],
                "end_timestamp": key[3],
                "target": int(target[example_index]),
                "set_size": count,
                "sensor_ids": "|".join(sensor_ids),
                "mean_local_probability": float(
                    local_probability[example_index, :count].mean()
                ),
            }
        )

    arrays = RegionArrays(
        learned=learned,
        handcrafted=handcrafted,
        local_probability=local_probability,
        mask=mask,
        target=target,
        split_code=split_code,
        metadata=pd.DataFrame(metadata_rows),
    )
    arrays.validate()
    return arrays


def main() -> None:
    args = parse_args()
    regional_config = load_yaml(args.config)
    run_root = Path(regional_config["sensor_cv_run_root"])
    if not run_root.is_dir():
        raise FileNotFoundError(run_root)
    sensor_config_path = resolve_sensor_config(
        run_root,
        regional_config.get("sensor_config_path"),
    )
    sensor_config = load_yaml(sensor_config_path)

    build_config = dict(regional_config.get("region_dataset", {}))
    requested_device = args.device or build_config.get("device", "auto")
    device = resolve_device(str(requested_device))
    batch_size = int(build_config.get("batch_size", 1024))
    num_workers = int(build_config.get("num_workers", 0))
    pin_memory = bool(build_config.get("pin_memory", device.type == "cuda"))
    expected_max_sensors_value = build_config.get("expected_max_sensors", 9)
    expected_max_sensors = (
        None if expected_max_sensors_value is None else int(expected_max_sensors_value)
    )

    output_root = args.output_root or Path(
        build_config.get(
            "output_root",
            f"runs/region_dataset_{datetime.now():%Y%m%d-%H%M%S}",
        )
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_root / "region_config.yaml")
    shutil.copy2(sensor_config_path, output_root / "sensor_config.yaml")

    data_config = sensor_config["data"]
    channels = list(data_config.get("channels", ["temperature", "humidity", "co", "co2"]))
    frame, schema = load_sensor_table(
        data_config["paths"],
        channels=channels,
        column_overrides=data_config.get("column_overrides", {}),
        fire_positive_values=data_config.get(
            "fire_positive_values", ["fire", "1", "true", "yes"]
        ),
        delimiter=data_config.get("delimiter", "auto"),
        encoding=data_config.get("encoding", "utf-8"),
    )
    episode_config = data_config.get("episode", {})
    frame = derive_episode_ids(
        frame,
        boundary_columns=episode_config.get("boundary_columns", ["scenario_label"]),
        max_gap_seconds=episode_config.get("max_gap_seconds", 30.0),
        overwrite=episode_config.get("overwrite_existing", False),
    )
    episodes = make_episode_table(frame)
    episode_repetition = dict(
        zip(episodes["episode_id"].astype(str), episodes["repetition_id"].astype(str))
    )

    n_folds = int(sensor_config.get("sensor_cv", {}).get("n_folds", 3))
    print(f"Sensor run root: {run_root}")
    print(f"Sensor config:   {sensor_config_path}")
    print(f"Output root:     {output_root}")
    print(f"Device:          {device}")

    fold_summaries: list[dict[str, Any]] = []
    for fold_index in range(n_folds):
        print(f"\n=== Building regional dataset for fold {fold_index}/{n_folds - 1} ===")
        fold_source = run_root / f"fold_{fold_index}"
        manifest_path = fold_source / "outer_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        final_dir = fold_source / "final"
        encoder_candidates = sorted(final_dir.glob(f"best_encoder_*_fold{fold_index}.pt"))
        if len(encoder_candidates) != 1:
            raise FileNotFoundError(
                f"Expected exactly one fold-{fold_index} encoder in {final_dir}; "
                f"found {encoder_candidates}."
            )
        projector = FoldSensorProjector(
            encoder_candidates[0],
            device=device,
        )
        if tuple(channels) != projector.channels:
            raise ValueError(
                f"Sensor config channels {channels} do not match checkpoint channels "
                f"{projector.channels}."
            )

        records: list[dict[str, Any]] = []
        for split_name, episode_ids in (
            ("development", manifest["development"]),
            ("test", manifest["test"]),
        ):
            loader = make_window_loader(
                frame,
                episode_ids=episode_ids,
                projector=projector,
                batch_size=batch_size,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
            split_records = collect_sensor_windows(
                loader,
                projector=projector,
                split_name=split_name,
                episode_repetition=episode_repetition,
            )
            records.extend(split_records)
            print(f"{split_name:>11}: {len(split_records):,} sensor windows")

        arrays = group_region_examples(
            records,
            expected_max_sensors=expected_max_sensors,
        )
        fold_output = output_root / f"fold_{fold_index}"
        npz_path = fold_output / f"region_dataset_fold{fold_index}.npz"
        metadata_path = fold_output / f"region_manifest_fold{fold_index}.csv"
        save_region_arrays(arrays, npz_path, metadata_path)

        fold_metadata = {
            "fold": fold_index,
            "encoder_checkpoint": str(projector.encoder_checkpoint_path),
            "classifier_checkpoint": str(projector.classifier_checkpoint_path),
            "channels": list(projector.channels),
            "window": projector.window_config,
            "normalization": projector.normalization.as_dict(),
            "learned_dimension": int(arrays.learned.shape[2]),
            "handcrafted_dimension": int(arrays.handcrafted.shape[2]),
            "handcrafted_feature_names": handcrafted_feature_names(projector.channels),
            "maximum_set_size": int(arrays.mask.shape[1]),
            "n_development_examples": int((arrays.split_code == 0).sum()),
            "n_test_examples": int((arrays.split_code == 1).sum()),
            "test_positive_fraction": float(
                arrays.target[arrays.split_code == 1].mean()
            ),
        }
        save_json(fold_metadata, fold_output / f"region_metadata_fold{fold_index}.json")
        fold_summaries.append(fold_metadata)
        print(
            f"Regional examples: development={fold_metadata['n_development_examples']:,}, "
            f"test={fold_metadata['n_test_examples']:,}, "
            f"max set={fold_metadata['maximum_set_size']}"
        )

    save_json(
        {
            "schema": schema.as_dict(),
            "sensor_cv_run_root": str(run_root),
            "sensor_config_path": str(sensor_config_path),
            "folds": fold_summaries,
        },
        output_root / "region_dataset_summary.json",
    )
    print(f"\nSaved regional datasets to: {output_root}")


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        main()
    finally:
        elapsed = time.perf_counter() - started
        hours, remainder = divmod(elapsed, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "\nTotal runtime: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
            f"({elapsed:.2f} seconds)"
        )
