from pathlib import Path
import importlib.util
import sys

import pandas as pd

from fire_ura.data import FireWindowDataset, derive_episode_ids, make_episode_table


def load_cv_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_sensor_cv.py"
    spec = importlib.util.spec_from_file_location("run_sensor_cv", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_local_target_and_progress_filter():
    rows = []
    base = pd.Timestamp("2026-01-01T00:00:00Z")
    progress = ["Ignition", "Flaming", "Ventilating", "Ventilating"]
    control = [0, 1, 1, 0]

    for index in range(4):
        rows.append(
            {
                "source_file": "x.csv",
                "episode_id": "ep0",
                "sensor_id": "S0",
                "timestamp": base + pd.Timedelta(seconds=10 * index),
                "scenario_label": "Cable",
                "progress_label": progress[index],
                "fire_label": 1,
                "fire_label_control": control[index],
                "temperature": float(index),
                "humidity": float(index),
                "co": float(index),
                "co2": float(index),
            }
        )

    frame = pd.DataFrame(rows)
    dataset = FireWindowDataset(frame, channels=("temperature", "humidity", "co", "co2"), window_length=2, stride=1, target_mode="last", target_column="fire_label_control", excluded_progress_labels=["Ventilating"])

    assert len(dataset) == 1
    assert int(dataset[0]["label"].item()) == 1
    assert dataset[0]["target_column"] == "fire_label_control"


def test_outer_fold_exactly_once_and_fire_rotation():
    cv = load_cv_module()
    rows = []
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    episode_index = 0

    for scenario in ("Cable", "Candles", "Lunts", "Wood"):
        for repetition in range(3):
            rows.append(
                {
                    "source_file": "x.csv",
                    "episode_id": f"{scenario}_{repetition}",
                    "repetition_id": str(repetition),
                    "scenario_label": scenario,
                    "fire_label": 1,
                    "start_time": start + pd.Timedelta(minutes=episode_index),
                    "end_time": start + pd.Timedelta(minutes=episode_index + 1),
                    "n_rows": 10,
                    "n_sensors": 2,
                }
            )
            episode_index += 1

    for index in range(7):
        rows.append(
            {
                "source_file": "x.csv",
                "episode_id": f"background_{index}",
                "repetition_id": str(index),
                "scenario_label": "Background",
                "fire_label": 0,
                "start_time": start + pd.Timedelta(minutes=episode_index),
                "end_time": start + pd.Timedelta(minutes=episode_index + 1),
                "n_rows": 10,
                "n_sensors": 2,
            }
        )
        episode_index += 1

    episodes = pd.DataFrame(rows)
    folds = cv.build_outer_folds(
        episodes,
        fire_scenarios=["Cable", "Candles", "Lunts", "Wood"],
        n_folds=3,
        seed=42,
    )

    all_ids = set(episodes["episode_id"])
    for episode_id in all_ids:
        assert sum(episode_id in fold["test"] for fold in folds) == 1

    for fold in folds:
        test = episodes[episodes["episode_id"].isin(fold["test"])]
        fire_counts = (
            test[test["fire_label"] == 1]["scenario_label"]
            .value_counts()
            .to_dict()
        )
        assert fire_counts == {
            "Cable": 1,
            "Candles": 1,
            "Lunts": 1,
            "Wood": 1,
        }
