from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from fire_ura.sensor_model import build_sensor_model


def load_script(name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_flat_mlp_model_shapes():
    model = build_sensor_model(
        architecture="flat_mlp",
        input_channels=4,
        window_length=12,
        hidden_channels=16,
        embedding_dim=8,
    )
    logits, embeddings = model(torch.randn(5, 4, 12))
    assert logits.shape == (5,)
    assert embeddings.shape == (5, 8)


def test_metrics_report_auprc_lift():
    cv = load_script("run_sensor_cv")
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.2, 0.8, 0.9], dtype=np.float64)
    result = cv.metrics(targets, probabilities, threshold=0.5)
    assert result["auprc"] == 1.0
    assert result["no_skill_auprc"] == 0.5
    assert result["auprc_lift"] == 0.5
    assert result["auprc_normalised_lift"] == 1.0


def test_aggregate_has_no_selected_threshold():
    cv = load_script("run_sensor_cv")
    diagnostic = {
        "auprc": 0.6,
        "no_skill_auprc": 0.2,
        "auprc_lift": 0.4,
        "auprc_normalised_lift": 0.5,
        "auroc": 0.7,
        "f1": 0.3,
        "missed_detection_rate": 0.4,
        "false_alarm_rate": 0.2,
        "accuracy": 0.8,
        "positive_fraction": 0.2,
    }
    result = {
        "selected_epochs": 2,
        "primary_metrics": {
            key: diagnostic[key]
            for key in (
                "auprc",
                "no_skill_auprc",
                "auprc_lift",
                "auprc_normalised_lift",
                "auroc",
            )
        },
        "test": {"threshold_0.5": diagnostic},
    }
    table, summary = cv.aggregate_fold_metrics([result, result, result])
    assert "selected_threshold" not in table.columns
    assert "selected_thresholds" not in summary
    assert "f1_at_0.5" in table.columns


def test_model_output_tags_and_aggregate_model_column():
    cv = load_script("run_sensor_cv")
    assert cv.model_output_tag({"architecture": "cnn"}) == "cnn"
    assert cv.model_output_tag({"architecture": "flat_mlp"}) == "mlp"
    assert cv.model_output_tag(
        {"architecture": "cnn", "output_tag": "My CNN"}
    ) == "my_cnn"

    fold_result = {
        "selected_epochs": 2,
        "primary_metrics": {
            "auprc": 0.6,
            "no_skill_auprc": 0.2,
            "auprc_lift": 0.4,
            "auprc_normalised_lift": 0.5,
            "auroc": 0.7,
        },
        "test": {
            "threshold_0.5": {
                "f1": 0.4,
                "missed_detection_rate": 0.3,
                "false_alarm_rate": 0.2,
                "accuracy": 0.8,
                "positive_fraction": 0.2,
            }
        },
    }
    table, summary = cv.aggregate_fold_metrics(
        [fold_result, fold_result],
        model_tag="cnn",
    )
    assert table["model"].tolist() == ["cnn", "cnn"]
    assert summary["model"] == "cnn"
