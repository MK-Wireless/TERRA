from pathlib import Path

import pandas as pd

from fire_ura.data import (
    FireWindowDataset,
    NormalizationStats,
    derive_episode_ids,
    load_sensor_table,
    make_episode_table,
)


def _write_synthetic_csv(path: Path) -> None:
    rows = []
    timestamps = pd.date_range("2026-01-01", periods=16, freq="10s", tz="UTC")
    scenarios = ["Background"] * 8 + ["Fire"] * 8

    for timestamp, scenario in zip(timestamps, scenarios):
        for sensor_id in ("S1", "S2", "S3"):
            rows.append(
                {
                    "Timestamp": timestamp.isoformat(),
                    "Sensor_ID": sensor_id,
                    "Air_Temperature": 20.0 + (scenario == "Fire") * 10.0,
                    "Humidity": 50.0 - (scenario == "Fire") * 10.0,
                    "CO": 2.0 + (scenario == "Fire") * 8.0,
                    "CO2": 420.0 + (scenario == "Fire") * 300.0,
                    "scenario_label": scenario,
                    "ternary_label": scenario,
                    "h2": 0.0,
                    "pm05": 0.0,
                    "pm10": 0.0,
                    "pm25": 0.0,
                    "pm40": 0.0,
                    "pm100": 0.0,
                    "pm_typical_size": 0.0,
                    "pm_total": 0.0,
                    "uv": 0.0,
                    "voc": 0.0,
                    "voc_raw": 0.0,
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_load_episode_and_windows(tmp_path: Path) -> None:
    csv_path = tmp_path / "synthetic.csv"
    _write_synthetic_csv(csv_path)

    frame, schema = load_sensor_table(csv_path)
    assert schema.channels["temperature"] == "Air_Temperature"
    assert set(frame["fire_label"].unique()) == {0, 1}

    frame = derive_episode_ids(frame)
    episodes = make_episode_table(frame)
    assert len(episodes) == 2

    train_ids = episodes["episode_id"].tolist()
    stats = NormalizationStats.fit(frame)
    dataset = FireWindowDataset(
        frame,
        episode_ids=train_ids,
        window_length=4,
        stride=2,
        normalization=stats,
    )
    sample = dataset[0]
    assert tuple(sample["x"].shape) == (15, 4)
    assert sample["label"].ndim == 0


def test_conflicting_sensor_labels_are_canonicalised(tmp_path: Path) -> None:
    rows = []
    timestamps = pd.date_range("2026-01-01", periods=4, freq="10s", tz="UTC")

    for index, timestamp in enumerate(timestamps):
        for sensor_id in ("S1", "S2", "S3"):
            is_fire = index >= 2 or (index == 1 and sensor_id == "S1")
            scenario = "Wood" if is_fire else "Background"
            rows.append(
                {
                    "Timestamp": timestamp.isoformat(),
                    "Sensor_ID": sensor_id,
                    "Air_Temperature": 20.0 + 10.0 * is_fire,
                    "Humidity": 50.0,
                    "CO": 2.0,
                    "CO2": 420.0,
                    "scenario_label": scenario,
                    "ternary_label": "Fire" if is_fire else "Background",
                    "h2": 0.0,
                    "pm05": 0.0,
                    "pm10": 0.0,
                    "pm25": 0.0,
                    "pm40": 0.0,
                    "pm100": 0.0,
                    "pm_typical_size": 0.0,
                    "pm_total": 0.0,
                    "uv": 0.0,
                    "voc": 0.0,
                    "voc_raw": 0.0,
                }
            )

    csv_path = tmp_path / "conflict.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    frame, _ = load_sensor_table(csv_path)
    frame = derive_episode_ids(frame)
    episodes = make_episode_table(frame)

    assert len(episodes) == 2
    assert episodes["scenario_label"].tolist() == ["Background", "Wood"]
    assert episodes["fire_label"].tolist() == [0, 1]

    per_time = frame.groupby("timestamp")[["scenario_label", "fire_label"]].nunique()
    assert not per_time.gt(1).any().any()
