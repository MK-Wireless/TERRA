#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
import zlib
from collections import defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Sampler

from fire_ura.data import (
    CANONICAL_CHANNELS,
    FireWindowDataset,
    NormalizationStats,
    PerSensorBackgroundNormalizationStats,
    derive_episode_ids,
    load_sensor_table,
    make_episode_table,
)
from fire_ura.sensor_model import SensorClassifier, build_sensor_model
from fire_ura.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run three-fold repetition-level cross-validation for the local "
            "sensor encoder. Each outer fold trains on two fire repetitions and "
            "tests on the remaining repetition."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional output-root override.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda", "mps"),
        default=None,
        help="Optional device override.",
    )
    return parser.parse_args()


def normalise_text(value: Any) -> str:
    import re

    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value).strip().lower(),
    ).strip("_")


def model_output_tag(model_config: dict[str, Any]) -> str:
    """Return a concise, filename-safe model label."""
    configured = model_config.get("output_tag")
    if configured:
        tag = normalise_text(configured)
    else:
        architecture = normalise_text(
            model_config.get("architecture", "cnn")
        )
        tag = {
            "flat_mlp": "mlp",
            "temporal_cnn": "cnn",
            "cnn": "cnn",
            "invariant_mlp": "invariant_mlp",
            "calibrated_mlp": "invariant_mlp",
            "relative_mlp": "invariant_mlp",
        }.get(architecture, architecture)

    if not tag:
        raise ValueError("Could not derive a non-empty model output tag.")
    return tag


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
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(convert(payload), handle, indent=2, sort_keys=True)
    temporary.replace(path)


def validate_config_structure(config: dict[str, Any]) -> None:
    split = config.get("split", {})
    misplaced = [
        key
        for key in ("sensor_model", "sensor_training", "sensor_cv")
        if isinstance(split, dict) and key in split
    ]
    if misplaced:
        names = ", ".join(misplaced)
        raise ValueError(
            f"{names} are nested under 'split:' in the YAML. Move them to the "
            "top level so they are siblings of data/window/split."
        )

    required = ("data", "window", "sensor_model", "sensor_training", "sensor_cv")
    missing = [key for key in required if key not in config]
    if missing:
        raise KeyError(
            "Missing top-level configuration sections: "
            + ", ".join(missing)
        )


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested == "mps":
        available = (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        )
        if not available:
            raise RuntimeError("MPS was requested but is unavailable.")
    return torch.device(requested)


def repetition_sort_key(value: Any) -> tuple[int, float | str]:
    text = str(value)
    try:
        return (0, float(text))
    except ValueError:
        return (1, text)


def stable_scenario_seed(seed: int, scenario: str) -> int:
    checksum = zlib.crc32(scenario.encode("utf-8"))
    return int((seed + checksum) % (2**32 - 1))


def assert_disjoint(named_ids: dict[str, Iterable[str]]) -> None:
    sets = {
        name: {str(value) for value in values}
        for name, values in named_ids.items()
    }
    names = list(sets)
    for index, first in enumerate(names):
        for second in names[index + 1 :]:
            overlap = sets[first].intersection(sets[second])
            if overlap:
                raise ValueError(
                    f"Episode leakage between {first} and {second}; "
                    f"example: {sorted(overlap)[0]}"
                )


def build_outer_folds(
    episodes: pd.DataFrame,
    *,
    fire_scenarios: Sequence[str],
    n_folds: int,
    seed: int,
) -> list[dict[str, list[str]]]:
    if n_folds < 2:
        raise ValueError("sensor_cv.n_folds must be at least 2.")

    scenario_lookup = {
        normalise_text(value): str(value)
        for value in fire_scenarios
    }
    requested = set(scenario_lookup)

    fire_rows = episodes[episodes["fire_label"].astype(int) == 1].copy()
    fire_rows["_scenario_key"] = fire_rows["scenario_label"].map(normalise_text)

    observed = set(fire_rows["_scenario_key"])
    missing_scenarios = requested.difference(observed)
    if missing_scenarios:
        raise ValueError(
            "Configured fire scenarios were not found: "
            + ", ".join(sorted(missing_scenarios))
        )

    fire_by_scenario: dict[str, list[str]] = {}
    for scenario_key in sorted(requested):
        group = fire_rows[fire_rows["_scenario_key"] == scenario_key].copy()
        group = group.sort_values(
            ["repetition_id", "start_time"],
            key=lambda column: (
                column.map(repetition_sort_key)
                if column.name == "repetition_id"
                else column
            ),
            kind="stable",
        )
        repetition_ids = sorted(
            group["repetition_id"].astype(str).unique(),
            key=repetition_sort_key,
        )
        if len(repetition_ids) != n_folds:
            raise ValueError(
                f"Scenario {scenario_lookup[scenario_key]!r} has "
                f"{len(repetition_ids)} repetitions, but n_folds={n_folds}. "
                f"Observed repetition IDs: {repetition_ids}"
            )

        ordered_ids: list[str] = []
        for repetition_id in repetition_ids:
            matching = group[
                group["repetition_id"].astype(str) == repetition_id
            ]
            if len(matching) != 1:
                raise ValueError(
                    f"Scenario {scenario_lookup[scenario_key]!r}, repetition "
                    f"{repetition_id!r} maps to {len(matching)} episodes; expected 1."
                )
            ordered_ids.append(str(matching.iloc[0]["episode_id"]))
        fire_by_scenario[scenario_key] = ordered_ids

    nonfire_rows = episodes[episodes["fire_label"].astype(int) == 0].copy()
    nonfire_rows["_scenario_key"] = nonfire_rows["scenario_label"].map(
        normalise_text
    )
    nonfire_test_by_fold: list[list[str]] = [[] for _ in range(n_folds)]

    for scenario_key, group in nonfire_rows.groupby("_scenario_key", sort=True):
        ids = sorted(group["episode_id"].astype(str).tolist())
        rng = np.random.default_rng(
            stable_scenario_seed(seed, str(scenario_key))
        )
        rng.shuffle(ids)
        for index, episode_id in enumerate(ids):
            nonfire_test_by_fold[index % n_folds].append(episode_id)

    all_nonfire = set(nonfire_rows["episode_id"].astype(str))
    folds: list[dict[str, list[str]]] = []

    for fold_index in range(n_folds):
        test_fire = [
            ids[fold_index]
            for ids in fire_by_scenario.values()
        ]
        development_fire = [
            episode_id
            for ids in fire_by_scenario.values()
            for index, episode_id in enumerate(ids)
            if index != fold_index
        ]
        test_nonfire = sorted(nonfire_test_by_fold[fold_index])
        development_nonfire = sorted(all_nonfire.difference(test_nonfire))

        fold = {
            "development": sorted(development_fire + development_nonfire),
            "test": sorted(test_fire + test_nonfire),
        }
        assert_disjoint(fold)
        folds.append(fold)

    all_episode_ids = set(episodes["episode_id"].astype(str))
    assignment_counts = {
        episode_id: sum(
            episode_id in set(fold["test"])
            for fold in folds
        )
        for episode_id in all_episode_ids
    }
    bad_assignments = {
        episode_id: count
        for episode_id, count in assignment_counts.items()
        if count != 1
    }
    if bad_assignments:
        example_id, example_count = next(iter(bad_assignments.items()))
        raise RuntimeError(
            "Every episode must appear in exactly one outer test fold. "
            f"Bad assignments={len(bad_assignments)}; "
            f"example {example_id!r} appears {example_count} times."
        )

    return folds


def split_inner_development(
    episodes: pd.DataFrame,
    development_ids: Sequence[str],
    *,
    fire_scenarios: Sequence[str],
    direction: int,
    seed: int,
) -> dict[str, list[str]]:
    if direction not in (0, 1):
        raise ValueError("direction must be 0 or 1.")

    development = episodes[
        episodes["episode_id"].astype(str).isin(
            {str(value) for value in development_ids}
        )
    ].copy()
    development["_scenario_key"] = development["scenario_label"].map(
        normalise_text
    )
    fire_keys = {normalise_text(value) for value in fire_scenarios}

    train_ids: list[str] = []
    val_ids: list[str] = []

    fire_development = development[
        development["fire_label"].astype(int) == 1
    ]
    for scenario_key in sorted(fire_keys):
        group = fire_development[
            fire_development["_scenario_key"] == scenario_key
        ].copy()
        group = group.sort_values(
            ["repetition_id", "start_time"],
            kind="stable",
        )
        ids = group["episode_id"].astype(str).tolist()
        if len(ids) != 2:
            raise ValueError(
                f"Inner split expected two development fire episodes for "
                f"{scenario_key!r}, found {len(ids)}."
            )
        train_ids.append(ids[direction])
        val_ids.append(ids[1 - direction])

    nonfire = development[
        development["fire_label"].astype(int) == 0
    ]
    for scenario_key, group in nonfire.groupby("_scenario_key", sort=True):
        ids = sorted(group["episode_id"].astype(str).tolist())
        rng = np.random.default_rng(
            stable_scenario_seed(seed + 1000, str(scenario_key))
        )
        rng.shuffle(ids)

        side_a = ids[0::2]
        side_b = ids[1::2]

        # A singleton is used in only one side per direction, never both.
        if direction == 0:
            train_ids.extend(side_a)
            val_ids.extend(side_b)
        else:
            train_ids.extend(side_b)
            val_ids.extend(side_a)

    split = {
        "train": sorted(train_ids),
        "val": sorted(val_ids),
    }
    assert_disjoint(split)

    if not split["train"] or not split["val"]:
        raise ValueError("Inner train/validation split is empty.")
    return split


def filter_training_rows(
    frame: pd.DataFrame,
    episode_ids: Sequence[str],
    excluded_progress_labels: Sequence[str],
) -> pd.DataFrame:
    selected = frame[
        frame["episode_id"].astype(str).isin(
            {str(value) for value in episode_ids}
        )
    ].copy()

    excluded = {
        normalise_text(value)
        for value in excluded_progress_labels
    }
    if excluded:
        if "progress_label" not in selected.columns:
            raise KeyError(
                "excluded_progress_labels were configured, but progress_label "
                "is unavailable."
            )
        selected = selected[
            ~selected["progress_label"].map(normalise_text).isin(excluded)
        ].copy()

    if selected.empty:
        raise ValueError(
            "No rows remain for training after progress-label filtering."
        )
    return selected


def dataset_targets(dataset: FireWindowDataset) -> np.ndarray:
    targets = np.empty(len(dataset), dtype=np.int64)
    for index, (group_index, start) in enumerate(dataset._windows):
        labels = dataset._groups[group_index]["labels"][
            start : start + dataset.window_length
        ]
        if dataset.target_mode == "last":
            targets[index] = int(labels[-1])
        elif dataset.target_mode == "any":
            targets[index] = int(labels.max())
        else:
            targets[index] = int(labels.mean() >= 0.5)
    return targets


class EpisodeBalancedSampler(Sampler[int]):
    """Choose class, episode, and window uniformly in that order."""

    def __init__(
        self,
        dataset: FireWindowDataset,
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0

        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        targets = dataset_targets(dataset)
        episode_windows: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}

        for window_index, (group_index, _) in enumerate(dataset._windows):
            label = int(targets[window_index])
            episode_id = str(dataset._groups[group_index]["episode_id"])
            episode_windows[label].setdefault(episode_id, []).append(
                window_index
            )

        for label in (0, 1):
            if not episode_windows[label]:
                raise ValueError(
                    f"Training dataset contains no windows for class {label}."
                )

        self.episode_windows = episode_windows
        self.episodes = {
            label: sorted(episode_windows[label])
            for label in (0, 1)
        }

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1

        labels = np.resize(
            np.asarray([0, 1], dtype=np.int64),
            self.num_samples,
        )
        rng.shuffle(labels)

        for value in labels:
            label = int(value)
            episodes = self.episodes[label]
            episode_id = episodes[int(rng.integers(len(episodes)))]
            candidates = self.episode_windows[label][episode_id]
            yield int(candidates[int(rng.integers(len(candidates)))])


def make_loader(
    dataset: FireWindowDataset,
    *,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    training: bool,
    samples_per_epoch: int | None,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)

    if training:
        sampler = EpisodeBalancedSampler(
            dataset,
            num_samples=int(samples_per_epoch or len(dataset)),
            seed=seed,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
        generator=generator,
        drop_last=False,
    )


def augment_windows(
    x: Tensor,
    config: dict[str, Any],
) -> Tensor:
    if not bool(config.get("enabled", False)):
        return x

    result = x.clone()
    batch, channels, length = result.shape

    gain_std = float(config.get("gain_std", 0.02))
    bias_std = float(config.get("bias_std", 0.05))
    noise_std = float(config.get("noise_std", 0.02))
    missing_probability = float(
        config.get("missing_sample_probability", 0.01)
    )
    max_shift = int(config.get("max_time_shift", 1))

    if gain_std > 0:
        gain = 1.0 + gain_std * torch.randn(
            batch, channels, 1,
            device=result.device,
            dtype=result.dtype,
        )
        result = result * gain

    if bias_std > 0:
        bias = bias_std * torch.randn(
            batch, channels, 1,
            device=result.device,
            dtype=result.dtype,
        )
        result = result + bias

    if noise_std > 0:
        result = result + noise_std * torch.randn_like(result)

    if missing_probability > 0:
        mask = torch.rand(
            batch, 1, length,
            device=result.device,
        ) < missing_probability
        result = result.masked_fill(mask, 0.0)

    if max_shift > 0:
        original = result.clone()
        shifts = torch.randint(
            -max_shift,
            max_shift + 1,
            (batch,),
            device=result.device,
        )
        for index, shift_tensor in enumerate(shifts):
            shift = int(shift_tensor.item())
            if shift > 0:
                result[index, :, shift:] = original[index, :, :-shift]
                result[index, :, :shift] = original[index, :, :1]
            elif shift < 0:
                amount = -shift
                result[index, :, :-amount] = original[index, :, amount:]
                result[index, :, -amount:] = original[index, :, -1:]

    return result


def safe_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def safe_auprc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float(targets.mean())
    return float(average_precision_score(targets, probabilities))


def metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
) -> dict[str, float | int]:
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(
        targets,
        predictions,
        labels=[0, 1],
    ).ravel()
    recall = float(recall_score(targets, predictions, zero_division=0))

    prevalence = float(targets.mean())
    auprc = safe_auprc(targets, probabilities)
    lift = float(auprc - prevalence)
    normalised_lift = (
        float(lift / (1.0 - prevalence))
        if prevalence < 1.0
        else float("nan")
    )

    return {
        "threshold": float(threshold),
        "n_windows": int(targets.size),
        "positive_fraction": prevalence,
        "no_skill_auprc": prevalence,
        "auprc_lift": lift,
        "auprc_normalised_lift": normalised_lift,
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(
            precision_score(targets, predictions, zero_division=0)
        ),
        "recall": recall,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "auroc": safe_auroc(targets, probabilities),
        "auprc": auprc,
        "brier": float(brier_score_loss(targets, probabilities)),
        "missed_detection_rate": float(1.0 - recall),
        "false_alarm_rate": (
            float(fp / (fp + tn))
            if fp + tn
            else float("nan")
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def scenario_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    scenarios: Sequence[str],
    *,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    table = pd.DataFrame(
        {
            "target": targets,
            "probability": probabilities,
            "scenario": list(scenarios),
        }
    )
    output: dict[str, dict[str, float | int]] = {}
    for scenario, group in table.groupby("scenario", sort=True):
        output[str(scenario)] = metrics(
            group["target"].to_numpy(dtype=np.int64),
            group["probability"].to_numpy(dtype=np.float64),
            threshold=threshold,
        )
    return output


def build_model(
    model_config: dict[str, Any],
    *,
    input_channels: int,
    window_length: int,
    device: torch.device,
) -> SensorClassifier:
    return build_sensor_model(
        architecture=str(model_config.get("architecture", "cnn")),
        input_channels=input_channels,
        window_length=window_length,
        base_channels=int(model_config.get("base_channels", 8)),
        hidden_channels=int(model_config.get("hidden_channels", 16)),
        embedding_dim=int(model_config.get("embedding_dim", 8)),
        kernel_size=int(model_config.get("kernel_size", 5)),
        dropout=float(model_config.get("dropout", 0.15)),
        embedding_activation=str(
            model_config.get("embedding_activation", "tanh")
        ),
        head_dropout=float(model_config.get("head_dropout", 0.0)),
        invariant_use_absolute=bool(
            model_config.get("invariant_use_absolute", True)
        ),
        invariant_use_relative=bool(
            model_config.get("invariant_use_relative", True)
        ),
        invariant_use_differences=bool(
            model_config.get("invariant_use_differences", True)
        ),
        invariant_use_shape=bool(
            model_config.get("invariant_use_shape", True)
        ),
        invariant_use_statistics=bool(
            model_config.get("invariant_use_statistics", True)
        ),
        invariant_use_level_statistics=bool(
            model_config.get(
                "invariant_use_level_statistics",
                True,
            )
        ),
        invariant_use_invariant_statistics=bool(
            model_config.get(
                "invariant_use_invariant_statistics",
                True,
            )
        ),
    ).to(device)


def fit_normalization(
    training_rows: pd.DataFrame,
    *,
    channels: Sequence[str],
    target_column: str,
    training_config: dict[str, Any],
) -> NormalizationStats | PerSensorBackgroundNormalizationStats:
    normalization_config = training_config.get("normalization", {})
    if isinstance(normalization_config, str):
        mode = normalization_config
        normalization_config = {}
    elif isinstance(normalization_config, dict):
        mode = normalization_config.get("mode", "global")
    else:
        raise TypeError(
            "sensor_training.normalization must be a string or mapping."
        )

    mode_key = normalise_text(mode)
    if mode_key in {"global", "training_global"}:
        return NormalizationStats.fit(
            training_rows,
            channels=channels,
        )
    if mode_key in {
        "per_sensor_background",
        "sensor_background",
        "background_calibrated",
    }:
        return PerSensorBackgroundNormalizationStats.fit(
            training_rows,
            channels=channels,
            target_column=str(
                normalization_config.get("target_column", target_column)
            ),
            background_value=int(
                normalization_config.get("background_value", 0)
            ),
            scenario_column=str(
                normalization_config.get("scenario_column", "scenario_label")
            ),
            scenario_values=list(
                normalization_config.get("scenario_values", [])
            ),
            minimum_samples_per_sensor=int(
                normalization_config.get("minimum_samples_per_sensor", 50)
            ),
        )
    raise ValueError(
        "sensor_training.normalization.mode must be 'global' or "
        "'per_sensor_background'. "
        f"Received {mode!r}."
    )


def make_datasets(
    frame: pd.DataFrame,
    *,
    train_ids: Sequence[str],
    eval_ids: Sequence[str],
    channels: Sequence[str],
    window_config: dict[str, Any],
    target_column: str,
    excluded_progress_labels: Sequence[str],
    training_config: dict[str, Any],
) -> tuple[
    FireWindowDataset,
    FireWindowDataset,
    NormalizationStats | PerSensorBackgroundNormalizationStats,
]:
    training_rows = filter_training_rows(
        frame,
        train_ids,
        excluded_progress_labels,
    )
    normalization = fit_normalization(
        training_rows,
        channels=channels,
        target_column=target_column,
        training_config=training_config,
    )

    common = {
        "frame": frame,
        "channels": channels,
        "window_length": int(window_config.get("length", 12)),
        "stride": int(window_config.get("stride", 12)),
        "normalization": normalization,
        "target_mode": str(window_config.get("target_mode", "last")),
        "target_column": target_column,
    }

    train_dataset = FireWindowDataset(
        episode_ids=train_ids,
        excluded_progress_labels=excluded_progress_labels,
        **common,
    )
    eval_dataset = FireWindowDataset(
        episode_ids=eval_ids,
        excluded_progress_labels=(),
        **common,
    )
    return train_dataset, eval_dataset, normalization


def make_amp_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_epoch(
    model: SensorClassifier,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    *,
    amp_enabled: bool,
    scaler: Any,
    gradient_clip_norm: float | None,
    augmentation_config: dict[str, Any],
) -> tuple[float, np.ndarray, np.ndarray]:
    model.train()
    losses = 0.0
    count = 0
    targets_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []

    for batch in loader:
        x = batch["x"].to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        targets = batch["label"].to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        x = augment_windows(x, augmentation_config)

        optimizer.zero_grad(set_to_none=True)
        autocast = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if amp_enabled
            else nullcontext()
        )
        with autocast:
            logits, _ = model(x)
            loss = loss_function(logits, targets)

        if amp_enabled:
            scaler.scale(loss).backward()
            if gradient_clip_norm is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(
                    model.parameters(),
                    gradient_clip_norm,
                )
            optimizer.step()

        batch_size = int(targets.shape[0])
        losses += float(loss.detach()) * batch_size
        count += batch_size
        targets_parts.append(targets.detach().cpu().numpy())
        probability_parts.append(
            torch.sigmoid(logits.detach()).cpu().numpy()
        )

    return (
        losses / count,
        np.concatenate(targets_parts).astype(np.int64),
        np.concatenate(probability_parts).astype(np.float64),
    )


@torch.no_grad()
def evaluate(
    model: SensorClassifier,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray, pd.DataFrame]:
    model.eval()
    losses = 0.0
    count = 0
    targets_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    metadata_parts: list[pd.DataFrame] = []

    for batch in loader:
        x = batch["x"].to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        targets = batch["label"].to(
            device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        logits, _ = model(x)
        loss = loss_function(logits, targets)

        batch_size = int(targets.shape[0])
        losses += float(loss) * batch_size
        count += batch_size
        targets_parts.append(targets.cpu().numpy())
        probability_parts.append(torch.sigmoid(logits).cpu().numpy())
        metadata_parts.append(
            pd.DataFrame(
                {
                    "source_file": [str(v) for v in batch["source_file"]],
                    "episode_id": [str(v) for v in batch["episode_id"]],
                    "sensor_id": [str(v) for v in batch["sensor_id"]],
                    "scenario_label": [str(v) for v in batch["scenario_label"]],
                    "progress_label": [str(v) for v in batch["progress_label"]],
                    "start_timestamp": [str(v) for v in batch["start_timestamp"]],
                    "end_timestamp": [str(v) for v in batch["end_timestamp"]],
                }
            )
        )

    if count == 0:
        raise RuntimeError("Evaluation loader produced no windows.")

    return (
        losses / count,
        np.concatenate(targets_parts).astype(np.int64),
        np.concatenate(probability_parts).astype(np.float64),
        pd.concat(metadata_parts, ignore_index=True),
    )


@dataclass
class TrainingSettings:
    batch_size: int
    num_workers: int
    pin_memory: bool
    samples_per_epoch: int | None
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    minimum_improvement: float
    gradient_clip_norm: float | None
    amp: bool
    augmentation: dict[str, Any]
    raw_config: dict[str, Any]


def read_training_settings(
    training_config: dict[str, Any],
    device: torch.device,
) -> TrainingSettings:
    if str(training_config.get("sampling", "episode_balanced")) != "episode_balanced":
        raise ValueError(
            "run_sensor_cv.py currently requires sampling: episode_balanced."
        )
    if str(training_config.get("class_balance", "none")) != "none":
        raise ValueError(
            "Use class_balance: none with episode-balanced sampling."
        )

    clip = training_config.get("gradient_clip_norm", 5.0)
    samples = training_config.get("samples_per_epoch")
    return TrainingSettings(
        batch_size=int(training_config.get("batch_size", 256)),
        num_workers=int(training_config.get("num_workers", 0)),
        pin_memory=bool(
            training_config.get("pin_memory", device.type == "cuda")
        ),
        samples_per_epoch=None if samples is None else int(samples),
        learning_rate=float(
            training_config.get("learning_rate", 1e-3)
        ),
        weight_decay=float(training_config.get("weight_decay", 1e-4)),
        max_epochs=int(training_config.get("epochs", 30)),
        early_stopping_patience=int(
            training_config.get("early_stopping_patience", 6)
        ),
        minimum_improvement=float(
            training_config.get("minimum_improvement", 1e-4)
        ),
        gradient_clip_norm=(
            None if clip is None else float(clip)
        ),
        amp=bool(training_config.get("amp", True)) and device.type == "cuda",
        augmentation=dict(training_config.get("augmentation", {})),
        raw_config=dict(training_config),
    )


def selection_run(
    *,
    frame: pd.DataFrame,
    train_ids: Sequence[str],
    val_ids: Sequence[str],
    channels: Sequence[str],
    window_config: dict[str, Any],
    model_config: dict[str, Any],
    target_column: str,
    excluded_progress_labels: Sequence[str],
    settings: TrainingSettings,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    """Use AUPRC only to select the epoch; do not calibrate a threshold."""
    set_seed(seed, deterministic=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_tag = model_output_tag(model_config)

    train_dataset, val_dataset, normalization = make_datasets(
        frame,
        train_ids=train_ids,
        eval_ids=val_ids,
        channels=channels,
        window_config=window_config,
        target_column=target_column,
        excluded_progress_labels=excluded_progress_labels,
        training_config=settings.raw_config,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        seed=seed,
        training=True,
        samples_per_epoch=settings.samples_per_epoch,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        seed=seed + 1,
        training=False,
        samples_per_epoch=None,
    )

    model = build_model(
        model_config,
        input_channels=len(channels),
        window_length=int(window_config.get("length", 12)),
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    scaler = make_amp_scaler(settings.amp)

    best_auprc = float("-inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, settings.max_epochs + 1):
        train_loss, train_targets, train_probabilities = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            amp_enabled=settings.amp,
            scaler=scaler,
            gradient_clip_norm=settings.gradient_clip_norm,
            augmentation_config=settings.augmentation,
        )
        val_loss, val_targets, val_probabilities, _ = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )

        train_summary = metrics(
            train_targets,
            train_probabilities,
            threshold=0.5,
        )
        val_summary = metrics(
            val_targets,
            val_probabilities,
            threshold=0.5,
        )
        current = float(val_summary["auprc"])

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auprc": train_summary["auprc"],
                "train_auprc_lift": train_summary["auprc_lift"],
                "train_auroc": train_summary["auroc"],
                "val_loss": val_loss,
                "val_auprc": val_summary["auprc"],
                "val_no_skill_auprc": val_summary["no_skill_auprc"],
                "val_auprc_lift": val_summary["auprc_lift"],
                "val_auroc": val_summary["auroc"],
            }
        )
        pd.DataFrame(history).to_csv(
            output_dir / f"history_{model_tag}.csv",
            index=False,
        )

        improved = current > best_auprc + settings.minimum_improvement
        marker = "*" if improved else " "
        print(
            f"{marker} selection epoch {epoch:02d}: "
            f"train AUPRC={train_summary['auprc']:.4f}, "
            f"val AUPRC={current:.4f} "
            f"(base={val_summary['no_skill_auprc']:.4f}, "
            f"lift={val_summary['auprc_lift']:.4f}), "
            f"val AUROC={val_summary['auroc']:.4f}"
        )

        if improved:
            best_auprc = current
            best_epoch = epoch
            stale = 0
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        else:
            stale += 1

        if stale >= settings.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Selection training produced no checkpoint.")

    model.load_state_dict(best_state)
    val_loss, val_targets, val_probabilities, val_metadata = evaluate(
        model,
        val_loader,
        loss_function,
        device,
    )
    val_summary = metrics(
        val_targets,
        val_probabilities,
        threshold=0.5,
    )
    val_summary["loss"] = val_loss

    torch.save(
        {
            "checkpoint_type": "inner_sensor_classifier",
            "epoch": best_epoch,
            "model_state_dict": best_state,
            "normalization": normalization.as_dict(),
            "target_column": target_column,
            "excluded_progress_labels": list(excluded_progress_labels),
            "diagnostic_threshold": 0.5,
        },
        output_dir / f"best_classifier_{model_tag}.pt",
    )
    save_json(
        {
            "best_epoch": best_epoch,
            "selection_metric": "validation_auprc",
            "validation_threshold_independent": {
                key: val_summary[key]
                for key in (
                    "auprc",
                    "no_skill_auprc",
                    "auprc_lift",
                    "auprc_normalised_lift",
                    "auroc",
                )
            },
            "validation_threshold_0.5": val_summary,
            "by_scenario_threshold_0.5": scenario_metrics(
                val_targets,
                val_probabilities,
                val_metadata["scenario_label"].tolist(),
                threshold=0.5,
            ),
            "train_episode_ids": list(train_ids),
            "val_episode_ids": list(val_ids),
        },
        output_dir / f"metrics_{model_tag}.json",
    )

    return {
        "best_epoch": best_epoch,
        "validation": val_summary,
    }


def final_run(
    *,
    frame: pd.DataFrame,
    development_ids: Sequence[str],
    test_ids: Sequence[str],
    channels: Sequence[str],
    window_config: dict[str, Any],
    model_config: dict[str, Any],
    target_column: str,
    excluded_progress_labels: Sequence[str],
    settings: TrainingSettings,
    device: torch.device,
    seed: int,
    epochs: int,
    fold_index: int,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed, deterministic=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    model_tag = model_output_tag(model_config)
    fold_tag = f"{model_tag}_fold{fold_index}"

    train_dataset, test_dataset, normalization = make_datasets(
        frame,
        train_ids=development_ids,
        eval_ids=test_ids,
        channels=channels,
        window_config=window_config,
        target_column=target_column,
        excluded_progress_labels=excluded_progress_labels,
        training_config=settings.raw_config,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        seed=seed,
        training=True,
        samples_per_epoch=settings.samples_per_epoch,
    )
    test_loader = make_loader(
        test_dataset,
        batch_size=settings.batch_size,
        num_workers=settings.num_workers,
        pin_memory=settings.pin_memory,
        seed=seed + 1,
        training=False,
        samples_per_epoch=None,
    )

    model = build_model(
        model_config,
        input_channels=len(channels),
        window_length=int(window_config.get("length", 12)),
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    loss_function = nn.BCEWithLogitsLoss()
    scaler = make_amp_scaler(settings.amp)
    history: list[dict[str, Any]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_targets, train_probabilities = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            amp_enabled=settings.amp,
            scaler=scaler,
            gradient_clip_norm=settings.gradient_clip_norm,
            augmentation_config=settings.augmentation,
        )
        train_summary = metrics(
            train_targets,
            train_probabilities,
            threshold=0.5,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auprc": train_summary["auprc"],
                "train_auprc_lift": train_summary["auprc_lift"],
                "train_auroc": train_summary["auroc"],
            }
        )
        print(
            f"  final epoch {epoch:02d}/{epochs:02d}: "
            f"loss={train_loss:.4f}, "
            f"AUPRC={train_summary['auprc']:.4f}"
        )

    pd.DataFrame(history).to_csv(
        output_dir / f"history_{fold_tag}.csv",
        index=False,
    )

    test_loss, test_targets, test_probabilities, test_metadata = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )
    at_half = metrics(
        test_targets,
        test_probabilities,
        threshold=0.5,
    )
    at_half["loss"] = test_loss

    episode_lookup = (
        frame[["episode_id", "repetition_id"]]
        .drop_duplicates("episode_id")
        .assign(episode_id=lambda table: table["episode_id"].astype(str))
    )
    predictions = test_metadata.copy()
    predictions["target"] = test_targets
    predictions["probability"] = test_probabilities
    predictions["prediction_threshold_0.5"] = (
        test_probabilities >= 0.5
    ).astype(np.int64)
    predictions = predictions.merge(
        episode_lookup,
        on="episode_id",
        how="left",
        validate="many_to_one",
    )
    predictions.to_csv(
        output_dir / f"test_predictions_{fold_tag}.csv",
        index=False,
    )

    grouping_sets = [
        ["scenario_label"],
        ["repetition_id"],
        ["progress_label"],
        ["scenario_label", "repetition_id"],
        ["sensor_id"],
    ]
    score_rows: list[dict[str, Any]] = []
    for grouping in grouping_sets:
        for keys, group in predictions.groupby(grouping, dropna=False, sort=True):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row: dict[str, Any] = {
                "grouping": "+".join(grouping),
                **{column: value for column, value in zip(grouping, keys)},
                "n_windows": int(len(group)),
                "positive_fraction": float(group["target"].mean()),
                "mean_probability": float(group["probability"].mean()),
                "median_probability": float(group["probability"].median()),
                "q05_probability": float(group["probability"].quantile(0.05)),
                "q95_probability": float(group["probability"].quantile(0.95)),
            }
            if group["target"].nunique() >= 2:
                row["auprc"] = safe_auprc(
                    group["target"].to_numpy(dtype=np.int64),
                    group["probability"].to_numpy(dtype=np.float64),
                )
                row["auroc"] = safe_auroc(
                    group["target"].to_numpy(dtype=np.int64),
                    group["probability"].to_numpy(dtype=np.float64),
                )
            score_rows.append(row)
    pd.DataFrame(score_rows).to_csv(
        output_dir / f"test_score_summary_{fold_tag}.csv",
        index=False,
    )

    checkpoint_metadata = {
        "epoch": epochs,
        "normalization": normalization.as_dict(),
        "channels": list(channels),
        "window": window_config,
        "encoder_config": model.encoder.config.to_dict(),
        "target_column": target_column,
        "excluded_progress_labels": list(excluded_progress_labels),
        "diagnostic_threshold": 0.5,
        "development_episode_ids": list(development_ids),
        "test_episode_ids": list(test_ids),
    }
    torch.save(
        {
            **checkpoint_metadata,
            "checkpoint_type": "sensor_classifier_cv_final",
            "model_state_dict": model.state_dict(),
        },
        output_dir / f"best_classifier_{fold_tag}.pt",
    )
    torch.save(
        {
            **checkpoint_metadata,
            "checkpoint_type": "sensor_encoder_cv_final",
            "model_state_dict": model.encoder.state_dict(),
        },
        output_dir / f"best_encoder_{fold_tag}.pt",
    )

    result = {
        "selected_epochs": epochs,
        "primary_metrics": {
            key: at_half[key]
            for key in (
                "auprc",
                "no_skill_auprc",
                "auprc_lift",
                "auprc_normalised_lift",
                "auroc",
            )
        },
        "test": {
            "threshold_0.5": at_half,
            "by_scenario_threshold_0.5": scenario_metrics(
                test_targets,
                test_probabilities,
                test_metadata["scenario_label"].tolist(),
                threshold=0.5,
            ),
        },
    }
    save_json(result, output_dir / f"metrics_{fold_tag}.json")
    return result


def summarise_episode_ids(
    episodes: pd.DataFrame,
    ids: Sequence[str],
) -> dict[str, Any]:
    selected = episodes[
        episodes["episode_id"].astype(str).isin(
            {str(value) for value in ids}
        )
    ]
    fire_selected = selected[selected["fire_label"].astype(int) == 1]
    return {
        "n_episodes": int(len(selected)),
        "n_fire": int(selected["fire_label"].astype(int).sum()),
        "n_nonfire": int(
            (selected["fire_label"].astype(int) == 0).sum()
        ),
        "scenario_counts": {
            str(key): int(value)
            for key, value in selected["scenario_label"]
            .value_counts()
            .sort_index()
            .items()
        },
        "fire_repetitions": {
            str(scenario): sorted(
                group["repetition_id"].astype(str).unique().tolist(),
                key=repetition_sort_key,
            )
            for scenario, group in fire_selected.groupby(
                "scenario_label", sort=True
            )
        },
    }


def write_fold_diagnostics(
    *,
    frame: pd.DataFrame,
    episodes: pd.DataFrame,
    development_ids: Sequence[str],
    test_ids: Sequence[str],
    channels: Sequence[str],
    target_column: str,
    output_dir: Path,
) -> None:
    """Write raw-data diagnostics for repetition and sensor distribution shifts."""
    output_dir.mkdir(parents=True, exist_ok=True)
    development_set = {str(value) for value in development_ids}
    test_set = {str(value) for value in test_ids}

    selected = frame[
        frame["episode_id"].astype(str).isin(development_set | test_set)
    ].copy()
    selected["episode_id"] = selected["episode_id"].astype(str)
    selected["split"] = np.where(
        selected["episode_id"].isin(test_set),
        "test",
        "development",
    )

    episode_manifest = episodes[
        episodes["episode_id"].astype(str).isin(development_set | test_set)
    ].copy()
    episode_manifest["episode_id"] = episode_manifest["episode_id"].astype(str)
    episode_manifest["split"] = np.where(
        episode_manifest["episode_id"].isin(test_set),
        "test",
        "development",
    )
    episode_manifest.to_csv(
        output_dir / "episode_manifest.csv",
        index=False,
    )

    progress = (
        selected.groupby(
            ["split", target_column, "progress_label"],
            dropna=False,
        )
        .size()
        .rename("n_rows")
        .reset_index()
    )
    progress["fraction_within_split_target"] = progress["n_rows"] / (
        progress.groupby(["split", target_column])["n_rows"]
        .transform("sum")
        .clip(lower=1)
    )
    progress.to_csv(
        output_dir / "progress_distribution.csv",
        index=False,
    )

    distribution_rows: list[dict[str, Any]] = []
    for (split_name, target), group in selected.groupby(
        ["split", target_column],
        dropna=False,
        sort=True,
    ):
        for channel in channels:
            values = pd.to_numeric(group[channel], errors="coerce").dropna()
            if values.empty:
                continue
            distribution_rows.append(
                {
                    "split": split_name,
                    "target": int(target),
                    "channel": channel,
                    "n_rows": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                    "q05": float(values.quantile(0.05)),
                    "median": float(values.median()),
                    "q95": float(values.quantile(0.95)),
                }
            )
    channel_distribution = pd.DataFrame(distribution_rows)
    channel_distribution.to_csv(
        output_dir / "channel_distribution.csv",
        index=False,
    )

    shift_rows: list[dict[str, Any]] = []
    for target in sorted(selected[target_column].dropna().unique()):
        for channel in channels:
            train_values = pd.to_numeric(
                selected.loc[
                    (selected["split"] == "development")
                    & (selected[target_column] == target),
                    channel,
                ],
                errors="coerce",
            ).dropna()
            test_values = pd.to_numeric(
                selected.loc[
                    (selected["split"] == "test")
                    & (selected[target_column] == target),
                    channel,
                ],
                errors="coerce",
            ).dropna()
            if train_values.empty or test_values.empty:
                continue
            train_std = float(train_values.std(ddof=1))
            mean_difference = float(test_values.mean() - train_values.mean())
            shift_rows.append(
                {
                    "target": int(target),
                    "channel": channel,
                    "development_mean": float(train_values.mean()),
                    "test_mean": float(test_values.mean()),
                    "development_std": train_std,
                    "test_std": float(test_values.std(ddof=1)),
                    "mean_difference": mean_difference,
                    "standardized_mean_shift": (
                        float(mean_difference / train_std)
                        if train_std > 0
                        else float("nan")
                    ),
                }
            )
    pd.DataFrame(shift_rows).to_csv(
        output_dir / "channel_standardized_shift.csv",
        index=False,
    )

    sensor_rows: list[dict[str, Any]] = []
    for (split_name, sensor_id, target), group in selected.groupby(
        ["split", "sensor_id", target_column],
        dropna=False,
        sort=True,
    ):
        for channel in channels:
            values = pd.to_numeric(group[channel], errors="coerce").dropna()
            if values.empty:
                continue
            sensor_rows.append(
                {
                    "split": split_name,
                    "sensor_id": str(sensor_id),
                    "target": int(target),
                    "channel": channel,
                    "n_rows": int(len(values)),
                    "mean": float(values.mean()),
                    "std": float(values.std(ddof=1)),
                }
            )
    pd.DataFrame(sensor_rows).to_csv(
        output_dir / "sensor_channel_distribution.csv",
        index=False,
    )


def aggregate_fold_metrics(
    fold_results: list[dict[str, Any]],
    *,
    model_tag: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate_names = (
        "auprc",
        "no_skill_auprc",
        "auprc_lift",
        "auprc_normalised_lift",
        "auroc",
        "f1_at_0.5",
        "missed_detection_rate_at_0.5",
        "false_alarm_rate_at_0.5",
        "accuracy_at_0.5",
        "positive_fraction",
    )

    for fold_index, result in enumerate(fold_results):
        primary = result["primary_metrics"]
        diagnostic = result["test"]["threshold_0.5"]
        row = {
            "fold": fold_index,
            "auprc": primary["auprc"],
            "no_skill_auprc": primary["no_skill_auprc"],
            "auprc_lift": primary["auprc_lift"],
            "auprc_normalised_lift": primary["auprc_normalised_lift"],
            "auroc": primary["auroc"],
            "f1_at_0.5": diagnostic["f1"],
            "missed_detection_rate_at_0.5": diagnostic[
                "missed_detection_rate"
            ],
            "false_alarm_rate_at_0.5": diagnostic["false_alarm_rate"],
            "accuracy_at_0.5": diagnostic["accuracy"],
            "positive_fraction": diagnostic["positive_fraction"],
            "selected_epochs": result["selected_epochs"],
        }
        rows.append(row)

    table = pd.DataFrame(rows)
    if model_tag is not None:
        table.insert(0, "model", model_tag)

    summary: dict[str, Any] = {}
    if model_tag is not None:
        summary["model"] = model_tag
    for name in aggregate_names:
        values = table[name].to_numpy(dtype=np.float64)
        summary[name] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1)),
            "values": values.tolist(),
        }
    summary["selected_epochs"] = table["selected_epochs"].tolist()
    summary["primary_metrics"] = [
        "auprc",
        "no_skill_auprc",
        "auprc_lift",
        "auprc_normalised_lift",
        "auroc",
    ]
    summary["threshold_note"] = (
        "F1, missed-detection rate, false-alarm rate, and accuracy are "
        "secondary diagnostics evaluated at a fixed threshold of 0.5."
    )
    return table, summary


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    validate_config_structure(config)

    data_config = config["data"]
    window_config = config["window"]
    model_config = config["sensor_model"]
    training_config = config["sensor_training"]
    cv_config = config["sensor_cv"]
    model_tag = model_output_tag(model_config)

    seed = int(training_config.get("seed", 42))
    requested_device = (
        args.device
        or training_config.get("device")
        or "auto"
    )
    device = resolve_device(str(requested_device))
    settings = read_training_settings(training_config, device)

    output_root = (
        args.output_root
        or Path(
            cv_config.get(
                "output_root",
                "runs/"
                f"sensor_cv_{model_tag}_"
                f"{datetime.now():%Y%m%d-%H%M%S}",
            )
        )
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_root / f"config_{model_tag}.yaml")

    channels = list(
        data_config.get("channels", list(CANONICAL_CHANNELS))
    )
    frame, schema = load_sensor_table(
        data_config["paths"],
        channels=channels,
        column_overrides=data_config.get("column_overrides", {}),
        fire_positive_values=data_config.get(
            "fire_positive_values",
            ["fire", "1", "true", "yes"],
        ),
        delimiter=data_config.get("delimiter", "auto"),
        encoding=data_config.get("encoding", "utf-8"),
    )

    episode_config = data_config.get("episode", {})
    frame = derive_episode_ids(
        frame,
        boundary_columns=episode_config.get(
            "boundary_columns",
            ["scenario_label"],
        ),
        max_gap_seconds=episode_config.get(
            "max_gap_seconds",
            30.0,
        ),
        overwrite=episode_config.get(
            "overwrite_existing",
            False,
        ),
    )
    episodes = make_episode_table(frame)

    target_column = str(
        training_config.get(
            "target_column",
            "fire_label_control",
        )
    )
    if target_column not in frame.columns:
        raise KeyError(
            f"Configured target_column={target_column!r} is absent. "
            "Add data.column_overrides.fire_label_control to the YAML."
        )

    excluded_progress_labels = list(
        training_config.get(
            "excluded_progress_labels",
            [],
        )
    )
    fire_scenarios = list(
        cv_config.get(
            "fire_scenarios",
            ["Cable", "Candles", "Lunts", "Wood"],
        )
    )
    n_folds = int(cv_config.get("n_folds", 3))

    folds = build_outer_folds(
        episodes,
        fire_scenarios=fire_scenarios,
        n_folds=n_folds,
        seed=seed,
    )

    print(f"Output root:          {output_root}")
    print(f"Output tag:           {model_tag}")
    print(f"Device:               {device}")
    print(f"Target column:        {target_column}")
    print(
        "Excluded in training: "
        + (", ".join(excluded_progress_labels) or "none")
    )
    normalization_config = training_config.get("normalization", {})
    normalization_mode = (
        normalization_config
        if isinstance(normalization_config, str)
        else normalization_config.get("mode", "global")
    )
    print(f"Normalization:        {normalization_mode}")
    print(
        "Resolved model:       "
        f"architecture={model_config.get('architecture', 'cnn')}, "
        f"hidden={model_config.get('hidden_channels', 16)}, "
        f"embedding={model_config.get('embedding_dim', 8)}"
    )

    save_json(
        {
            "model": model_tag,
            "schema": schema.as_dict(),
            "target_column": target_column,
            "excluded_progress_labels": excluded_progress_labels,
            "fire_scenarios": fire_scenarios,
            "n_folds": n_folds,
            "model_config": model_config,
            "training_config": training_config,
            "normalization_mode": normalization_mode,
        },
        output_root / f"run_metadata_{model_tag}.json",
    )
    episodes.to_csv(
        output_root / f"episodes_{model_tag}.csv",
        index=False,
    )

    fold_results: list[dict[str, Any]] = []

    for fold_index, outer in enumerate(folds):
        print(f"\n=== Outer fold {fold_index + 1}/{n_folds} ===")
        fold_dir = output_root / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True)

        outer_payload = {
            "fold": fold_index,
            "development": outer["development"],
            "test": outer["test"],
            "development_summary": summarise_episode_ids(
                episodes,
                outer["development"],
            ),
            "test_summary": summarise_episode_ids(
                episodes,
                outer["test"],
            ),
        }
        save_json(
            outer_payload,
            fold_dir / "outer_manifest.json",
        )
        write_fold_diagnostics(
            frame=frame,
            episodes=episodes,
            development_ids=outer["development"],
            test_ids=outer["test"],
            channels=channels,
            target_column=target_column,
            output_dir=fold_dir / "diagnostics",
        )
        print(
            "Development: "
            f"{outer_payload['development_summary']['n_fire']} fire, "
            f"{outer_payload['development_summary']['n_nonfire']} non-fire episodes"
        )
        print(
            "Test:        "
            f"{outer_payload['test_summary']['n_fire']} fire, "
            f"{outer_payload['test_summary']['n_nonfire']} non-fire episodes"
        )

        inner_results: list[dict[str, Any]] = []
        for direction in (0, 1):
            inner = split_inner_development(
                episodes,
                outer["development"],
                fire_scenarios=fire_scenarios,
                direction=direction,
                seed=seed + fold_index,
            )
            save_json(
                {
                    "direction": direction,
                    **inner,
                    "train_summary": summarise_episode_ids(
                        episodes,
                        inner["train"],
                    ),
                    "val_summary": summarise_episode_ids(
                        episodes,
                        inner["val"],
                    ),
                },
                fold_dir / f"inner_{direction}_manifest.json",
            )
            print(f"\nInner direction {direction}:")
            inner_result = selection_run(
                frame=frame,
                train_ids=inner["train"],
                val_ids=inner["val"],
                channels=channels,
                window_config=window_config,
                model_config=model_config,
                target_column=target_column,
                excluded_progress_labels=excluded_progress_labels,
                settings=settings,
                device=device,
                seed=seed + 100 * fold_index + 10 * direction,
                output_dir=fold_dir / f"inner_{direction}",
            )
            inner_results.append(inner_result)

        selected_epochs = max(
            1,
            int(
                np.rint(
                    np.mean(
                        [
                            result["best_epoch"]
                            for result in inner_results
                        ]
                    )
                )
            ),
        )
        print(
            f"\nRetraining on both development repetitions for "
            f"{selected_epochs} epoch(s)."
        )
        final_result = final_run(
            frame=frame,
            development_ids=outer["development"],
            test_ids=outer["test"],
            channels=channels,
            window_config=window_config,
            model_config=model_config,
            target_column=target_column,
            excluded_progress_labels=excluded_progress_labels,
            settings=settings,
            device=device,
            seed=seed + 1000 + fold_index,
            epochs=selected_epochs,
            fold_index=fold_index,
            output_dir=fold_dir / "final",
        )
        final_result["inner_selection"] = inner_results
        save_json(
            final_result,
            fold_dir / f"fold_result_{model_tag}_fold{fold_index}.json",
        )
        fold_results.append(final_result)

        primary = final_result["primary_metrics"]
        diagnostic = final_result["test"]["threshold_0.5"]
        print(
            f"Fold {fold_index}: AUPRC={primary['auprc']:.4f} "
            f"(baseline={primary['no_skill_auprc']:.4f}, "
            f"lift={primary['auprc_lift']:.4f}), "
            f"AUROC={primary['auroc']:.4f}; "
            f"diagnostic F1@0.5={diagnostic['f1']:.4f}, "
            f"FAR@0.5={diagnostic['false_alarm_rate']:.4f}"
        )

    table, summary = aggregate_fold_metrics(
        fold_results,
        model_tag=model_tag,
    )
    table.to_csv(
        output_root / f"fold_metrics_{model_tag}.csv",
        index=False,
    )
    save_json(
        {
            "folds": fold_results,
            "aggregate": summary,
        },
        output_root / f"summary_{model_tag}.json",
    )

    print("\n=== Three-fold aggregate: primary metrics ===")
    for name in (
        "auprc",
        "no_skill_auprc",
        "auprc_lift",
        "auprc_normalised_lift",
        "auroc",
    ):
        entry = summary[name]
        print(
            f"{name:>24}: "
            f"{entry['mean']:.4f} ± {entry['std']:.4f}"
        )
    print("\nSecondary diagnostics at fixed threshold 0.5")
    for name in (
        "f1_at_0.5",
        "missed_detection_rate_at_0.5",
        "false_alarm_rate_at_0.5",
    ):
        entry = summary[name]
        print(
            f"{name:>36}: "
            f"{entry['mean']:.4f} ± {entry['std']:.4f}"
        )
    print(f"Saved results to: {output_root}")


if __name__ == "__main__":
    start_time = time.perf_counter()
    try:
        main()
    finally:
        elapsed_seconds = time.perf_counter() - start_time
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(
            "\nTotal runtime: "
            f"{int(hours):02d}:{int(minutes):02d}:{seconds:05.2f} "
            f"({elapsed_seconds:.2f} seconds)"
        )
