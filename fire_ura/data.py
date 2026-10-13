from __future__ import annotations

import csv
import glob
import math
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CANONICAL_CHANNELS = (
    "co2",
    "co",
    "h2",
    "humidity",
    "pm05",
    "pm100",
    "pm10",
    "pm25",
    "pm40",
    "pm_typical_size",
    "pm_total",
    "temperature",
    "uv",
    "voc",
    "voc_raw",
)

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": (
        "timestamp", "time_stamp", "datetime", "date_time", "time", "date",
    ),
    "sensor_id": (
        "sensor_id", "sensorid", "node_id", "nodeid", "device_id", "deviceid",
        "position_id", "positionid",
    ),
    "temperature": (
        "temperature", "air_temperature", "airtemperature", "temp",
        "temperature_c", "temperature_celsius", "air_temp",
    ),
    "humidity": (
        "humidity", "relative_humidity", "relativehumidity", "rh", "hum",
    ),
    "co": (
        "co", "carbon_monoxide", "carbonmonoxide", "co_ppm",
    ),
    "co2": (
        "co2", "co_2", "carbon_dioxide", "carbondioxide", "eco2", "co2_ppm",
    ),
    "scenario_label": (
        "scenario_label", "scenariolabel", "scenario", "experiment",
        "experiment_label", "event_label", "event",
    ),
    "fire_label": (
        "fire_label", "firelabel", "is_fire", "isfire", "fire",
    ),
    "fire_label_control": (
        "fire_label_control", "firelabelcontrol", "local_fire_label",
        "sensor_fire_target", "control_fire_label",
    ),
    "ternary_label": (
        "ternary_label", "ternarylabel", "class_label", "classlabel",
    ),
    "anomaly_label": (
        "anomaly_label", "anomalylabel", "is_anomaly", "isanomaly", "anomaly",
    ),
    "progress_label": (
        "progress_label", "progresslabel", "stage", "phase", "intensity_label",
    ),
    "episode_id": (
        "episode_id", "episodeid", "run_id", "runid", "trial_id", "trialid",
    ),
    "repetition_id": (
        "repetition_id", "repetitionid", "repeat_id", "repeatid",
        "rep", "repetition",
    ),
}


def _normalise_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")


def _resolve_column(
    columns: Sequence[str],
    canonical_name: str,
    override: str | None = None,
    required: bool = False,
) -> str | None:
    """Resolve a canonical field against exact and normalised column aliases."""
    if override:
        if override not in columns:
            raise KeyError(
                f"Configured column '{override}' for '{canonical_name}' is absent. "
                f"Available columns: {list(columns)}"
            )
        return override

    normalised_to_original: dict[str, str] = {}
    for column in columns:
        normalised_to_original.setdefault(_normalise_name(column), column)

    candidates = (canonical_name,) + _COLUMN_ALIASES.get(canonical_name, ())
    for candidate in candidates:
        match = normalised_to_original.get(_normalise_name(candidate))
        if match is not None:
            return match

    if required:
        raise KeyError(
            f"Could not infer required column '{canonical_name}'. "
            f"Set data.column_overrides.{canonical_name} in the YAML config. "
            f"Available columns: {list(columns)}"
        )
    return None


@dataclass(frozen=True)
class DatasetSchema:
    timestamp: str
    sensor_id: str
    channels: dict[str, str]
    scenario_label: str | None = None
    fire_label: str | None = None
    fire_label_control: str | None = None
    ternary_label: str | None = None
    anomaly_label: str | None = None
    progress_label: str | None = None
    episode_id: str | None = None
    repetition_id: str | None = None

    @classmethod
    def infer(
        cls,
        frame: pd.DataFrame,
        overrides: Mapping[str, str] | None = None,
        channels: Sequence[str] = CANONICAL_CHANNELS,
    ) -> "DatasetSchema":
        overrides = dict(overrides or {})
        columns = list(frame.columns)

        resolved_channels: dict[str, str] = {}
        for channel in channels:
            if channel not in CANONICAL_CHANNELS:
                raise ValueError(
                    f"Unsupported canonical channel '{channel}'. "
                    f"Supported channels: {CANONICAL_CHANNELS}"
                )
            resolved_channels[channel] = _resolve_column(
                columns, channel, overrides.get(channel), required=True
            )

        schema = cls(
            timestamp=_resolve_column(
                columns, "timestamp", overrides.get("timestamp"), required=True
            ),
            sensor_id=_resolve_column(
                columns, "sensor_id", overrides.get("sensor_id"), required=True
            ),
            channels=resolved_channels,
            scenario_label=_resolve_column(
                columns, "scenario_label", overrides.get("scenario_label")
            ),
            fire_label=_resolve_column(
                columns, "fire_label", overrides.get("fire_label")
            ),
            fire_label_control=_resolve_column(
                columns,
                "fire_label_control",
                overrides.get("fire_label_control"),
            ),
            ternary_label=_resolve_column(
                columns, "ternary_label", overrides.get("ternary_label")
            ),
            anomaly_label=_resolve_column(
                columns, "anomaly_label", overrides.get("anomaly_label")
            ),
            progress_label=_resolve_column(
                columns, "progress_label", overrides.get("progress_label")
            ),
            episode_id=_resolve_column(
                columns, "episode_id", overrides.get("episode_id")
            ),
            repetition_id=_resolve_column(
                columns, "repetition_id", overrides.get("repetition_id")
            ),
        )

        if not any((schema.fire_label, schema.ternary_label)):
            raise KeyError(
                "Could not infer a fire-specific label. The dataset must expose "
                "'fire_label' or 'ternary_label', or you must configure an override. "
                "An anomaly label alone is insufficient because nuisance events are non-fire."
            )
        return schema

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "sensor_id": self.sensor_id,
            "channels": self.channels,
            "scenario_label": self.scenario_label,
            "fire_label": self.fire_label,
            "fire_label_control": self.fire_label_control,
            "ternary_label": self.ternary_label,
            "anomaly_label": self.anomaly_label,
            "progress_label": self.progress_label,
            "episode_id": self.episode_id,
            "repetition_id": self.repetition_id,
        }


@dataclass(frozen=True)
class NormalizationStats:
    mean: dict[str, float]
    std: dict[str, float]

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        channels: Sequence[str] = CANONICAL_CHANNELS,
    ) -> "NormalizationStats":
        if frame.empty:
            raise ValueError("Cannot fit normalization statistics on an empty frame.")

        mean: dict[str, float] = {}
        std: dict[str, float] = {}
        for channel in channels:
            values = pd.to_numeric(frame[channel], errors="coerce")
            channel_mean = float(values.mean())
            channel_std = float(values.std(ddof=0))
            if not math.isfinite(channel_mean):
                raise ValueError(f"Channel '{channel}' has no finite training values.")
            if not math.isfinite(channel_std) or channel_std < 1e-12:
                channel_std = 1.0
            mean[channel] = channel_mean
            std[channel] = channel_std
        return cls(mean=mean, std=std)

    def transform(
        self,
        values: np.ndarray,
        channels: Sequence[str] = CANONICAL_CHANNELS,
        *,
        sensor_id: str | None = None,
    ) -> np.ndarray:
        del sensor_id
        means = np.asarray([self.mean[channel] for channel in channels], dtype=np.float32)
        stds = np.asarray([self.std[channel] for channel in channels], dtype=np.float32)
        return (values.astype(np.float32) - means) / stds

    def as_dict(self) -> dict[str, Any]:
        return {"mode": "global", "mean": self.mean, "std": self.std}


@dataclass(frozen=True)
class PerSensorBackgroundNormalizationStats:
    """Training-only sensor calibration from local non-fire observations.

    Each sensor is normalised relative to its own background mean and standard
    deviation. Sensors without enough background samples use global background
    statistics fitted on the same training episodes.
    """

    global_mean: dict[str, float]
    global_std: dict[str, float]
    sensor_mean: dict[str, dict[str, float]]
    sensor_std: dict[str, dict[str, float]]
    target_column: str
    background_value: int = 0
    scenario_column: str = "scenario_label"
    scenario_values: tuple[str, ...] = ()
    minimum_samples_per_sensor: int = 50

    @classmethod
    def fit(
        cls,
        frame: pd.DataFrame,
        channels: Sequence[str] = CANONICAL_CHANNELS,
        *,
        target_column: str = "fire_label_control",
        background_value: int = 0,
        scenario_column: str = "scenario_label",
        scenario_values: Sequence[str] = (),
        minimum_samples_per_sensor: int = 50,
    ) -> "PerSensorBackgroundNormalizationStats":
        if frame.empty:
            raise ValueError("Cannot fit normalization statistics on an empty frame.")
        if "sensor_id" not in frame.columns:
            raise KeyError("Per-sensor normalization requires a sensor_id column.")
        if target_column not in frame.columns:
            raise KeyError(
                f"Per-sensor background normalization requires {target_column!r}."
            )
        if minimum_samples_per_sensor <= 0:
            raise ValueError("minimum_samples_per_sensor must be positive.")

        local_nonfire_mask = (
            pd.to_numeric(frame[target_column], errors="coerce")
            .fillna(background_value)
            .astype(int)
            .eq(int(background_value))
        )
        background = frame[local_nonfire_mask].copy()

        requested_scenarios = tuple(str(value) for value in scenario_values)
        if requested_scenarios:
            if scenario_column not in frame.columns:
                raise KeyError(
                    f"Configured background scenario column {scenario_column!r} "
                    "is absent."
                )
            requested_keys = {
                _normalise_label(value)
                for value in requested_scenarios
            }
            clean_background = frame[
                local_nonfire_mask
                & frame[scenario_column].map(_normalise_label).isin(requested_keys)
            ].copy()
            if clean_background.empty:
                warnings.warn(
                    "No rows matched the requested clean-background scenarios; "
                    "falling back to all local non-fire training rows.",
                    RuntimeWarning,
                )
            else:
                background = clean_background

        if background.empty:
            warnings.warn(
                "No local non-fire rows were available in the training split; "
                "falling back to all training rows for normalization.",
                RuntimeWarning,
            )
            background = frame.copy()

        global_stats = NormalizationStats.fit(background, channels=channels)
        sensor_mean: dict[str, dict[str, float]] = {}
        sensor_std: dict[str, dict[str, float]] = {}

        for sensor_id, group in background.groupby("sensor_id", sort=True):
            if len(group) < minimum_samples_per_sensor:
                continue
            stats = NormalizationStats.fit(group, channels=channels)
            key = str(sensor_id)
            sensor_mean[key] = stats.mean
            sensor_std[key] = stats.std

        return cls(
            global_mean=global_stats.mean,
            global_std=global_stats.std,
            sensor_mean=sensor_mean,
            sensor_std=sensor_std,
            target_column=str(target_column),
            background_value=int(background_value),
            scenario_column=str(scenario_column),
            scenario_values=requested_scenarios,
            minimum_samples_per_sensor=int(minimum_samples_per_sensor),
        )

    def transform(
        self,
        values: np.ndarray,
        channels: Sequence[str] = CANONICAL_CHANNELS,
        *,
        sensor_id: str | None = None,
    ) -> np.ndarray:
        key = "" if sensor_id is None else str(sensor_id)
        mean = self.sensor_mean.get(key, self.global_mean)
        std = self.sensor_std.get(key, self.global_std)
        means = np.asarray([mean[channel] for channel in channels], dtype=np.float32)
        stds = np.asarray([std[channel] for channel in channels], dtype=np.float32)
        return (values.astype(np.float32) - means) / stds

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": "per_sensor_background",
            "target_column": self.target_column,
            "background_value": self.background_value,
            "scenario_column": self.scenario_column,
            "scenario_values": list(self.scenario_values),
            "minimum_samples_per_sensor": self.minimum_samples_per_sensor,
            "global_mean": self.global_mean,
            "global_std": self.global_std,
            "sensor_mean": self.sensor_mean,
            "sensor_std": self.sensor_std,
        }


def expand_input_paths(paths: str | Path | Sequence[str | Path]) -> list[Path]:
    """Expand files, directories, and glob patterns into unique CSV-like files."""
    if isinstance(paths, (str, Path)):
        paths = [paths]

    expanded: list[Path] = []
    for raw_path in paths:
        path_text = str(raw_path)
        candidate = Path(path_text)

        if candidate.is_dir():
            for suffix in ("*.csv", "*.txt", "*.tsv"):
                expanded.extend(candidate.rglob(suffix))
        elif candidate.is_file():
            expanded.append(candidate)
        else:
            expanded.extend(Path(match) for match in glob.glob(path_text, recursive=True))

    unique = sorted({path.resolve() for path in expanded if path.is_file()})
    if not unique:
        raise FileNotFoundError(f"No input files matched: {[str(p) for p in paths]}")
    return unique


def _detect_delimiter(path: Path, encoding: str = "utf-8") -> str:
    with path.open("r", encoding=encoding, errors="replace") as handle:
        sample = handle.read(65536)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        return ","


def _read_table(
    path: Path,
    delimiter: str = "auto",
    encoding: str = "utf-8",
    nrows: int | None = None,
) -> pd.DataFrame:
    sep = _detect_delimiter(path, encoding) if delimiter == "auto" else delimiter
    frame = pd.read_csv(
        path,
        sep=sep,
        encoding=encoding,
        nrows=nrows,
        low_memory=False,
    )
    if frame.empty:
        raise ValueError(f"Input file contains no rows: {path}")
    frame.columns = [str(column).strip() for column in frame.columns]
    return frame


def _normalise_label(value: Any) -> str:
    if pd.isna(value):
        return ""
    return _normalise_name(str(value))



def _parse_binary_label_series(
    series: pd.Series,
    positive_values: Sequence[str],
) -> pd.Series:
    """Parse a binary label series while preserving per-row sensor targets."""
    positives = {_normalise_label(value) for value in positive_values}

    if pd.api.types.is_bool_dtype(series):
        return series.astype(np.int8)
    if pd.api.types.is_numeric_dtype(series):
        return (
            pd.to_numeric(series, errors="coerce")
            .fillna(0)
            .gt(0)
            .astype(np.int8)
        )
    return series.map(
        lambda value: int(_normalise_label(value) in positives)
    ).astype(np.int8)


def _parse_fire_labels(
    frame: pd.DataFrame,
    schema: DatasetSchema,
    fire_values: Sequence[str],
) -> pd.Series:
    if schema.fire_label:
        return _parse_binary_label_series(
            frame[schema.fire_label],
            fire_values,
        )

    if schema.ternary_label:
        return _parse_binary_label_series(
            frame[schema.ternary_label],
            fire_values,
        )

    raise RuntimeError("DatasetSchema guarantees a fire_label or ternary_label.")


def load_sensor_table(
    paths: str | Path | Sequence[str | Path],
    *,
    channels: Sequence[str] = CANONICAL_CHANNELS,
    column_overrides: Mapping[str, str] | None = None,
    fire_positive_values: Sequence[str] = ("fire", "1", "true", "yes", "fire_relevant"),
    delimiter: str = "auto",
    encoding: str = "utf-8",
    nrows_per_file: int | None = None,
) -> tuple[pd.DataFrame, DatasetSchema]:
    """
    Load one or more raw files and return a canonical table.

    Canonical output columns:
      source_file, timestamp, sensor_id, temperature, humidity, co, co2,
      scenario_label, progress_label, fire_label, [fire_label_control],
      [episode_id], [repetition_id].
    """
    input_paths = expand_input_paths(paths)
    canonical_frames: list[pd.DataFrame] = []
    inferred_schema: DatasetSchema | None = None

    for path in input_paths:
        raw = _read_table(
            path, delimiter=delimiter, encoding=encoding, nrows=nrows_per_file
        )
        schema = DatasetSchema.infer(
            raw, overrides=column_overrides, channels=channels
        )
        if inferred_schema is None:
            inferred_schema = schema

        canonical = pd.DataFrame(index=raw.index)
        canonical["source_file"] = path.name

        timestamps = raw[schema.timestamp]
        parsed_time = pd.to_datetime(timestamps, errors="coerce", utc=True)
        if parsed_time.notna().mean() >= 0.95:
            canonical["timestamp"] = parsed_time
        else:
            numeric_time = pd.to_numeric(timestamps, errors="coerce")
            if numeric_time.notna().mean() < 0.95:
                raise ValueError(
                    f"Timestamp column '{schema.timestamp}' in {path.name} could not "
                    "be parsed as datetime or numeric time."
                )
            canonical["timestamp"] = numeric_time.astype(np.float64)

        canonical["sensor_id"] = raw[schema.sensor_id].astype(str).str.strip()

        for canonical_channel, raw_column in schema.channels.items():
            canonical[canonical_channel] = pd.to_numeric(
                raw[raw_column], errors="coerce"
            )

        canonical["scenario_label"] = (
            raw[schema.scenario_label].astype(str)
            if schema.scenario_label
            else "unknown"
        )
        canonical["progress_label"] = (
            raw[schema.progress_label].astype(str)
            if schema.progress_label
            else ""
        )
        canonical["fire_label"] = _parse_fire_labels(
            raw, schema, fire_positive_values
        )
        if schema.fire_label_control:
            canonical["fire_label_control"] = _parse_binary_label_series(
                raw[schema.fire_label_control],
                fire_positive_values,
            )

        if schema.episode_id:
            canonical["episode_id"] = raw[schema.episode_id].astype(str)
        if schema.repetition_id:
            canonical["repetition_id"] = raw[schema.repetition_id].astype(str)

        before = len(canonical)
        canonical = canonical.dropna(
            subset=["timestamp", "sensor_id", *channels]
        ).copy()
        dropped = before - len(canonical)
        if dropped:
            canonical.attrs["dropped_rows"] = dropped

        canonical_frames.append(canonical)

    assert inferred_schema is not None
    combined = pd.concat(canonical_frames, ignore_index=True)
    combined = combined.sort_values(
        ["source_file", "timestamp", "sensor_id"], kind="stable"
    ).reset_index(drop=True)
    return combined, inferred_schema


def _timestamp_gap_seconds(previous: Any, current: Any) -> float:
    if isinstance(previous, pd.Timestamp) and isinstance(current, pd.Timestamp):
        return float((current - previous).total_seconds())
    return float(current) - float(previous)


def _majority_value(series: pd.Series) -> Any:
    """Return the most frequent non-missing value, with deterministic tie-breaking."""
    non_missing = series.dropna()
    if non_missing.empty:
        return ""

    counts = non_missing.astype(str).value_counts(sort=True)
    max_count = counts.max()
    tied_values = sorted(counts[counts == max_count].index.tolist())
    chosen_text = tied_values[0]

    for value in non_missing:
        if str(value) == chosen_text:
            return value
    return non_missing.iloc[0]


def derive_episode_ids(
    frame: pd.DataFrame,
    *,
    boundary_columns: Sequence[str] = ("scenario_label",),
    max_gap_seconds: float | None = 30.0,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Derive region-level episode IDs after synchronising labels across sensor nodes.

    At each timestamp, all sensor rows are assigned one canonical region-level
    boundary label and one canonical binary fire label using majority voting.
    Episodes are then formed from contiguous runs of the canonical boundary labels.

    The original sensor-specific labels are retained as ``sensor_<column>`` fields
    for diagnostics.
    """
    if "episode_id" in frame.columns and not overwrite:
        result = frame.copy()
        if "repetition_id" not in result.columns:
            result = assign_repetition_ids(result)
        return result

    required = {
        "source_file",
        "timestamp",
        "fire_label",
        *boundary_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Cannot derive episodes; missing columns: {sorted(missing)}")

    result_parts: list[pd.DataFrame] = []
    canonical_columns = [*boundary_columns, "fire_label"]

    for source_file, source_frame in frame.groupby("source_file", sort=False):
        label_rows = source_frame[["timestamp", *canonical_columns]].copy()

        disagreement = (
            label_rows.groupby("timestamp", dropna=False)[canonical_columns]
            .nunique(dropna=False)
            .gt(1)
            .any(axis=1)
        )
        n_conflicts = int(disagreement.sum())
        if n_conflicts:
            n_timestamps = int(label_rows["timestamp"].nunique(dropna=False))
            warnings.warn(
                f"{n_conflicts} of {n_timestamps} timestamps in {source_file} "
                "contain conflicting labels across sensors. Canonical region-level "
                "labels will be assigned by majority vote.",
                RuntimeWarning,
            )

        aggregation = {column: _majority_value for column in canonical_columns}
        timestamp_labels = (
            label_rows.groupby("timestamp", as_index=False, dropna=False)
            .agg(aggregation)
            .sort_values("timestamp", kind="stable")
            .reset_index(drop=True)
        )

        starts = np.zeros(len(timestamp_labels), dtype=bool)
        if len(starts):
            starts[0] = True

        for index in range(1, len(timestamp_labels)):
            previous = timestamp_labels.iloc[index - 1]
            current = timestamp_labels.iloc[index]

            label_changed = any(
                _normalise_label(previous[column])
                != _normalise_label(current[column])
                for column in boundary_columns
            )

            gap_too_large = False
            if max_gap_seconds is not None:
                gap = _timestamp_gap_seconds(
                    previous["timestamp"], current["timestamp"]
                )
                gap_too_large = gap > max_gap_seconds

            starts[index] = label_changed or gap_too_large

        segment_numbers = np.cumsum(starts) - 1
        timestamp_labels["episode_id"] = [
            f"{Path(source_file).stem}__ep{number:04d}"
            for number in segment_numbers
        ]

        source_copy = source_frame.copy()
        for column in canonical_columns:
            sensor_column = f"sensor_{column}"
            if sensor_column not in source_copy.columns:
                source_copy[sensor_column] = source_copy[column]

        rename_map = {
            column: f"canonical_{column}" for column in canonical_columns
        }
        canonical_lookup = timestamp_labels[
            ["timestamp", "episode_id", *canonical_columns]
        ].rename(columns=rename_map)

        merged = source_copy.merge(
            canonical_lookup,
            on="timestamp",
            how="left",
            validate="many_to_one",
        )

        for column in canonical_columns:
            merged[column] = merged.pop(f"canonical_{column}")

        result_parts.append(merged)

    result = pd.concat(result_parts, ignore_index=True)
    result = assign_repetition_ids(result)
    return result.sort_values(
        ["source_file", "timestamp", "sensor_id"], kind="stable"
    ).reset_index(drop=True)


def assign_repetition_ids(frame: pd.DataFrame) -> pd.DataFrame:
    """
    Assign occurrence indices to episodes sharing the same canonical scenario label.

    This is a deterministic convenience field. It should be inspected before using
    leave-one-repetition-out splitting, because source datasets may provide a more
    authoritative repetition identifier.
    """
    if "episode_id" not in frame.columns:
        raise KeyError("episode_id is required before assigning repetition IDs.")

    result = frame.copy()
    episode_table = (
        result.groupby(["source_file", "episode_id"], as_index=False)
        .agg(
            scenario_label=("scenario_label", "first"),
            start_time=("timestamp", "min"),
        )
        .sort_values(["source_file", "scenario_label", "start_time"], kind="stable")
    )
    episode_table["repetition_id"] = (
        episode_table.groupby(["source_file", "scenario_label"], dropna=False)
        .cumcount()
        .astype(str)
    )

    result = result.drop(columns=["repetition_id"], errors="ignore").merge(
        episode_table[["source_file", "episode_id", "repetition_id"]],
        on=["source_file", "episode_id"],
        how="left",
        validate="many_to_one",
    )
    return result


def make_episode_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Create one metadata row per canonical region-level episode."""
    required = {
        "source_file",
        "episode_id",
        "repetition_id",
        "scenario_label",
        "fire_label",
        "timestamp",
        "sensor_id",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing episode metadata columns: {sorted(missing)}")

    consistency = (
        frame.groupby(["source_file", "episode_id"])[
            ["scenario_label", "fire_label"]
        ]
        .nunique(dropna=False)
    )
    inconsistent = consistency.gt(1).any(axis=1)
    if inconsistent.any():
        bad_key = inconsistent[inconsistent].index[0]
        raise ValueError(
            "Canonical labels are not constant within episode "
            f"{bad_key}. Episode derivation is inconsistent."
        )

    episode_table = (
        frame.groupby(["source_file", "episode_id"], as_index=False)
        .agg(
            repetition_id=("repetition_id", "first"),
            scenario_label=("scenario_label", "first"),
            fire_label=("fire_label", "first"),
            start_time=("timestamp", "min"),
            end_time=("timestamp", "max"),
            n_rows=("timestamp", "size"),
            n_sensors=("sensor_id", "nunique"),
        )
    )
    episode_table["stratum"] = (
        episode_table["fire_label"].astype(str)
        + "__"
        + episode_table["scenario_label"].map(_normalise_label)
    )
    return episode_table.sort_values(
        ["source_file", "start_time"], kind="stable"
    ).reset_index(drop=True)


class FireWindowDataset(Dataset):
    """
    Sensor-level temporal-window dataset.

    Returned x has shape [channels, window_length], suitable for torch.nn.Conv1d.
    Windows never cross source-file, episode, or sensor boundaries.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        episode_ids: Iterable[str] | None = None,
        channels: Sequence[str] = CANONICAL_CHANNELS,
        window_length: int = 12,
        stride: int = 3,
        normalization: NormalizationStats | PerSensorBackgroundNormalizationStats | None = None,
        target_mode: str = "last",
        target_column: str = "fire_label",
        excluded_progress_labels: Sequence[str] = (),
    ) -> None:
        if window_length <= 0:
            raise ValueError("window_length must be positive.")
        if stride <= 0:
            raise ValueError("stride must be positive.")
        if target_mode not in {"last", "any", "majority"}:
            raise ValueError("target_mode must be one of: last, any, majority.")

        required = {
            "source_file",
            "episode_id",
            "sensor_id",
            "timestamp",
            target_column,
            *channels,
        }
        if excluded_progress_labels:
            required.add("progress_label")
        missing = required.difference(frame.columns)
        if missing:
            raise KeyError(f"Window dataset is missing columns: {sorted(missing)}")

        selected = frame
        if episode_ids is not None:
            episode_set = {str(value) for value in episode_ids}
            selected = frame[frame["episode_id"].astype(str).isin(episode_set)]
        selected = selected.sort_values(
            ["source_file", "episode_id", "sensor_id", "timestamp"], kind="stable"
        )

        self.channels = tuple(channels)
        self.window_length = int(window_length)
        self.stride = int(stride)
        self.normalization = normalization
        self.target_mode = target_mode
        self.target_column = str(target_column)
        self.excluded_progress_labels = {
            _normalise_label(value)
            for value in excluded_progress_labels
        }

        self._groups: list[dict[str, Any]] = []
        self._windows: list[tuple[int, int]] = []

        group_columns = ["source_file", "episode_id", "sensor_id"]
        for group_key, group in selected.groupby(group_columns, sort=False):
            values = group.loc[:, self.channels].to_numpy(dtype=np.float32)
            labels = group[self.target_column].to_numpy(dtype=np.int64)
            timestamps = group["timestamp"].to_numpy()
            progress_labels = (
                group["progress_label"]
                .map(_normalise_label)
                .to_numpy(dtype=object)
                if "progress_label" in group.columns
                else np.full(len(group), "", dtype=object)
            )

            if len(group) < self.window_length:
                continue

            group_index = len(self._groups)
            self._groups.append(
                {
                    "values": values,
                    "labels": labels,
                    "timestamps": timestamps,
                    "progress_labels": progress_labels,
                    "source_file": str(group_key[0]),
                    "episode_id": str(group_key[1]),
                    "sensor_id": str(group_key[2]),
                    "scenario_label": str(group["scenario_label"].iloc[0]),
                }
            )

            final_start = len(group) - self.window_length
            for start in range(0, final_start + 1, self.stride):
                stop = start + self.window_length
                if self.excluded_progress_labels:
                    window_progress = progress_labels[start:stop]
                    if any(
                        value in self.excluded_progress_labels
                        for value in window_progress
                    ):
                        continue
                self._windows.append((group_index, start))

        if not self._windows:
            raise ValueError(
                "No valid windows were created. Check the split, window length, "
                "episode derivation, and missing-value filtering."
            )

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        group_index, start = self._windows[index]
        group = self._groups[group_index]
        stop = start + self.window_length

        values = group["values"][start:stop]
        if self.normalization is not None:
            values = self.normalization.transform(
                values,
                self.channels,
                sensor_id=group["sensor_id"],
            )

        labels = group["labels"][start:stop]
        if self.target_mode == "last":
            label = int(labels[-1])
        elif self.target_mode == "any":
            label = int(labels.max())
        else:
            label = int(labels.mean() >= 0.5)

        x = torch.from_numpy(values.T.copy())  # [channels, time]
        return {
            "x": x,
            "label": torch.tensor(label, dtype=torch.float32),
            "source_file": group["source_file"],
            "episode_id": group["episode_id"],
            "sensor_id": group["sensor_id"],
            "scenario_label": group["scenario_label"],
            "target_column": self.target_column,
            "progress_label": str(group["progress_labels"][stop - 1]),
            "start_timestamp": str(group["timestamps"][start]),
            "end_timestamp": str(group["timestamps"][stop - 1]),
        }
