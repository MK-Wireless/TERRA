#!/usr/bin/env python
from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, RandomSampler

from fire_ura.region_data import (
    EpisodeBalancedRegionSampler,
    RegionArrays,
    RegionSetDataset,
    assert_disjoint_indices,
    fit_feature_standardizer,
    indices_for_episodes,
    load_region_arrays,
    masked_mean_numpy,
)
from fire_ura.region_model import build_region_model
from fire_ura.utils import load_yaml, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run three-fold regional receiver evaluation under perfect recovery."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=None)
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
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(convert(payload), indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def safe_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float("nan")
    return float(roc_auc_score(targets, probabilities))


def safe_auprc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    if np.unique(targets).size < 2:
        return float(targets.mean())
    return float(average_precision_score(targets, probabilities))

def mdr_at_far(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    target_far: float,
) -> dict[str, float]:
    """
    Evaluate the best empirical ROC operating point satisfying
    FAR <= target_far.

    This is a test-set performance metric, not a threshold-selection
    procedure for deployment.
    """
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    n_positive = int(np.sum(targets == 1))
    n_negative = int(np.sum(targets == 0))

    if n_positive == 0 or n_negative == 0:
        return {
            "target_far": float(target_far),
            "achieved_far": float("nan"),
            "mdr": float("nan"),
            "tpr": float("nan"),
        }

    fpr, tpr, _ = roc_curve(
        targets,
        probabilities,
        drop_intermediate=False,
    )

    feasible = np.flatnonzero(
        fpr <= float(target_far) + 1e-12
    )

    if feasible.size == 0:
        # In practice the ROC origin should always be feasible,
        # but keep this guard for robustness.
        return {
            "target_far": float(target_far),
            "achieved_far": float("nan"),
            "mdr": float("nan"),
            "tpr": float("nan"),
        }

    # Among all operating points satisfying the FAR constraint,
    # choose the one with maximum detection probability.
    feasible_tpr = tpr[feasible]
    best_local = int(np.argmax(feasible_tpr))
    best_index = int(feasible[best_local])

    achieved_far = float(fpr[best_index])
    achieved_tpr = float(tpr[best_index])

    return {
        "target_far": float(target_far),
        "achieved_far": achieved_far,
        "mdr": float(1.0 - achieved_tpr),
        "tpr": achieved_tpr,
    }

def metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    targets = np.asarray(targets, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    predictions = (probabilities >= threshold).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(targets, predictions, labels=[0, 1]).ravel()
    prevalence = float(targets.mean())
    auprc = safe_auprc(targets, probabilities)
    lift = float(auprc - prevalence)
    normalised_lift = (
        float(lift / (1.0 - prevalence)) if prevalence < 1.0 else float("nan")
    )
    recall = float(recall_score(targets, predictions, zero_division=0))
    op_far_1pct = mdr_at_far(
        targets,
        probabilities,
        target_far=0.01,
    )

    op_far_5pct = mdr_at_far(
        targets,
        probabilities,
        target_far=0.05,
    )
    return {
        "threshold": float(threshold),
        "n_examples": int(targets.size),
        "positive_fraction": prevalence,
        "no_skill_auprc": prevalence,
        "auprc": auprc,
        "auprc_lift": lift,
        "auprc_normalised_lift": normalised_lift,
        "auroc": safe_auroc(targets, probabilities),
        "accuracy": float(accuracy_score(targets, predictions)),
        "precision": float(precision_score(targets, predictions, zero_division=0)),
        "recall": recall,
        "f1": float(f1_score(targets, predictions, zero_division=0)),
        "brier": float(brier_score_loss(targets, probabilities)),
        "missed_detection_rate": (
            float(fn / (fn + tp))
            if fn + tp
            else float("nan")
        ),
        "false_alarm_rate": (
            float(fp / (fp + tn))
            if fp + tn
            else float("nan")
        ),
        "mdr_at_far_1pct": op_far_1pct["mdr"],
        "achieved_far_at_1pct": op_far_1pct["achieved_far"],
        "mdr_at_far_5pct": op_far_5pct["mdr"],
        "achieved_far_at_5pct": op_far_5pct["achieved_far"],
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def by_scenario_metrics(
    metadata: pd.DataFrame,
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int]]:
    table = metadata.copy()
    table["target_for_metric"] = targets
    table["probability_for_metric"] = probabilities
    output: dict[str, dict[str, float | int]] = {}
    for scenario, group in table.groupby("scenario_label", sort=True):
        output[str(scenario)] = metrics(
            group["target_for_metric"].to_numpy(dtype=np.int64),
            group["probability_for_metric"].to_numpy(dtype=np.float64),
        )
    return output


def episode_balanced_weights(targets: np.ndarray, episode_ids: Sequence[str]) -> np.ndarray:
    table = pd.DataFrame(
        {"target": np.asarray(targets, dtype=np.int64), "episode_id": list(map(str, episode_ids))}
    )
    weights = np.zeros(len(table), dtype=np.float64)
    classes = sorted(table["target"].unique().tolist())
    if classes != [0, 1]:
        raise ValueError(f"Expected binary development targets, found {classes}.")
    for target in classes:
        class_table = table[table["target"] == target]
        episodes = sorted(class_table["episode_id"].unique().tolist())
        for episode_id in episodes:
            indices = class_table.index[class_table["episode_id"] == episode_id].to_numpy()
            weights[indices] = 1.0 / (len(classes) * len(episodes) * len(indices))
    weights *= len(weights) / weights.sum()
    return weights


@dataclass(frozen=True)
class NeuralSettings:
    batch_size: int
    num_workers: int
    samples_per_epoch: int
    sampling_mode: str
    learning_rate: float
    weight_decay: float
    max_epochs: int
    early_stopping_patience: int
    minimum_improvement: float
    gradient_clip_norm: float | None
    hidden_dim: int
    phi_hidden_dim: int
    phi_output_dim: int
    rho_hidden_dim: int
    dropout: float


def read_neural_settings(config: Mapping[str, Any]) -> NeuralSettings:
    clip = config.get("gradient_clip_norm", 5.0)
    return NeuralSettings(
        batch_size=int(config.get("batch_size", 256)),
        num_workers=int(config.get("num_workers", 0)),
        samples_per_epoch=int(config.get("samples_per_epoch", 6000)),
        sampling_mode=str(
            config.get("sampling_mode", "episode_class_balanced")
        ).strip().lower(),
        learning_rate=float(config.get("learning_rate", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4)),
        max_epochs=int(config.get("epochs", 30)),
        early_stopping_patience=int(config.get("early_stopping_patience", 6)),
        minimum_improvement=float(config.get("minimum_improvement", 1e-4)),
        gradient_clip_norm=None if clip is None else float(clip),
        hidden_dim=int(config.get("hidden_dim", 32)),
        phi_hidden_dim=int(config.get("phi_hidden_dim", 32)),
        phi_output_dim=int(config.get("phi_output_dim", 16)),
        rho_hidden_dim=int(config.get("rho_hidden_dim", 16)),
        dropout=float(config.get("dropout", 0.15)),
    )


def make_loader(
    dataset: RegionSetDataset,
    *,
    targets: np.ndarray,
    episode_ids: Sequence[str],
    settings: NeuralSettings,
    training: bool,
    seed: int,
) -> DataLoader:
    sampler = None
    shuffle = False
    generator = None

    if training:
        mode = settings.sampling_mode

        if mode == "episode_class_balanced":
            sampler = EpisodeBalancedRegionSampler(
                targets,
                episode_ids,
                num_samples=settings.samples_per_epoch,
                seed=seed,
            )

            print(
                "Confirmed episode-class-balanced loader: "
                f"num_samples={settings.samples_per_epoch}, "
                f"dataset_size={len(dataset)}, "
                f"positive_fraction={float(np.mean(targets)):.6f}"
            )

        elif mode == "empirical":
            # One shuffled pass through all available training examples.
            shuffle = True
            generator = torch.Generator()
            generator.manual_seed(seed)

            print(
                "Confirmed empirical one-pass loader: "
                f"dataset_size={len(dataset)}, "
                f"positive_fraction={float(np.mean(targets)):.6f}"
            )

        elif mode == "empirical_fixed_budget":
            # Uniform example sampling with replacement preserves the
            # empirical class prior in expectation while fixing the number
            # of training samples per epoch.
            generator = torch.Generator()
            generator.manual_seed(seed)

            sampler = RandomSampler(
                dataset,
                replacement=True,
                num_samples=settings.samples_per_epoch,
                generator=generator,
            )

            print(
                "Confirmed empirical fixed-budget loader: "
                f"num_samples={settings.samples_per_epoch}, "
                f"dataset_size={len(dataset)}, "
                f"positive_fraction={float(np.mean(targets)):.6f}"
            )

        else:
            raise ValueError(
                "sampling_mode must be one of "
                "'episode_class_balanced', 'empirical', or "
                f"'empirical_fixed_budget'; received {mode!r}."
            )

    return DataLoader(
        dataset,
        batch_size=settings.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        generator=generator,
        num_workers=settings.num_workers,
        persistent_workers=settings.num_workers > 0,
        drop_last=False,
    )


def build_model(
    architecture: str,
    *,
    input_dim: int,
    settings: NeuralSettings,
    device: torch.device,
) -> nn.Module:
    return build_region_model(
        architecture,
        input_dim=input_dim,
        hidden_dim=settings.hidden_dim,
        phi_hidden_dim=settings.phi_hidden_dim,
        phi_output_dim=settings.phi_output_dim,
        rho_hidden_dim=settings.rho_hidden_dim,
        dropout=settings.dropout,
    ).to(device)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_function: nn.Module,
    device: torch.device,
    gradient_clip_norm: float | None,
) -> float:
    model.train()
    total_loss = 0.0
    count = 0
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32)
        mask = batch["mask"].to(device, dtype=torch.bool)
        target = batch["target"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, mask)
        loss = loss_function(logits, target)
        loss.backward()
        if gradient_clip_norm is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        batch_size = int(target.shape[0])
        total_loss += float(loss.detach()) * batch_size
        count += batch_size
    return total_loss / max(count, 1)


@torch.no_grad()
def evaluate_neural(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    target_parts: list[np.ndarray] = []
    probability_parts: list[np.ndarray] = []
    index_parts: list[np.ndarray] = []
    for batch in loader:
        x = batch["x"].to(device, dtype=torch.float32)
        mask = batch["mask"].to(device, dtype=torch.bool)
        logits = model(x, mask)
        target_parts.append(batch["target"].numpy().astype(np.int64))
        probability_parts.append(torch.sigmoid(logits).cpu().numpy().astype(np.float64))
        index_parts.append(np.asarray(batch["index"], dtype=np.int64))
    return (
        np.concatenate(target_parts),
        np.concatenate(probability_parts),
        np.concatenate(index_parts),
    )


def neural_selection_run(
    arrays: RegionArrays,
    *,
    representation: str,
    architecture: str,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    settings: NeuralSettings,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> dict[str, Any]:
    set_seed(seed, deterministic=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    assert_disjoint_indices(train_indices, val_indices)
    values = getattr(arrays, representation)
    feature_mean, feature_std = fit_feature_standardizer(values, arrays.mask, train_indices)
    train_dataset = RegionSetDataset(
        arrays,
        representation=representation,
        indices=train_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    val_dataset = RegionSetDataset(
        arrays,
        representation=representation,
        indices=val_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    train_metadata = arrays.metadata.iloc[train_indices]
    train_loader = make_loader(
        train_dataset,
        targets=arrays.target[train_indices],
        episode_ids=train_metadata["episode_id"].astype(str).tolist(),
        settings=settings,
        training=True,
        seed=seed,
    )
    val_loader = make_loader(
        val_dataset,
        targets=arrays.target[val_indices],
        episode_ids=arrays.metadata.iloc[val_indices]["episode_id"].astype(str).tolist(),
        settings=settings,
        training=False,
        seed=seed + 1,
    )
    model = build_model(
        architecture,
        input_dim=values.shape[2],
        settings=settings,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    # best_auprc = float("-inf")
    best_brier = float("inf")
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, settings.max_epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            device,
            settings.gradient_clip_norm,
        )
        val_targets, val_probabilities, _ = evaluate_neural(model, val_loader, device)
        val_summary = metrics(val_targets, val_probabilities)

        current_brier = float(val_summary["brier"])
        current_auprc = float(val_summary["auprc"])

        improved = (
                current_brier
                < best_brier - settings.minimum_improvement
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_brier": current_brier,
                "val_auprc": current_auprc,
                "val_auprc_lift": val_summary["auprc_lift"],
                "val_auroc": val_summary["auroc"],
                "val_f1_at_0.5": val_summary["f1"],
                "val_far_at_0.5": val_summary["false_alarm_rate"],
                "val_mdr_at_0.5": val_summary[
                    "missed_detection_rate"
                ],
            }
        )

        pd.DataFrame(history).to_csv(
            output_dir / "history.csv",
            index=False,
        )

        print(
            f"    {'*' if improved else ' '} epoch {epoch:02d}: "
            f"loss={train_loss:.4f}, "
            f"val Brier={current_brier:.6f}, "
            f"val AUPRC={current_auprc:.4f}, "
            f"val AUROC={val_summary['auroc']:.4f}"
        )

        if improved:
            best_brier = current_brier
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1

        if stale >= settings.early_stopping_patience:
            break

    if best_state is None:
        raise RuntimeError("Regional model selection produced no checkpoint.")
    model.load_state_dict(best_state)
    val_targets, val_probabilities, val_order = evaluate_neural(model, val_loader, device)
    val_summary = metrics(val_targets, val_probabilities)
    torch.save(
        {
            "architecture": architecture,
            "representation": representation,
            "best_epoch": best_epoch,
            "selection_metric": "validation_brier",
            "best_validation_brier": best_brier,
            "model_state_dict": best_state,
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "best_model.pt",
    )
    pd.DataFrame(
        {
            "example_index": val_order,
            "target": val_targets,
            "probability": val_probabilities,
        }
    ).to_csv(output_dir / "validation_predictions.csv", index=False)
    save_json(
        {
            "best_epoch": best_epoch,
            "selection_metric": "validation_brier",
            "best_validation_brier": best_brier,
            "validation": val_summary,
        },
        output_dir / "metrics.json",
    )

    return {
        "best_epoch": best_epoch,
        "selection_metric": "validation_brier",
        "best_validation_brier": best_brier,
        "validation": val_summary,
    }


def neural_final_run(
    arrays: RegionArrays,
    *,
    representation: str,
    architecture: str,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    epochs: int,
    settings: NeuralSettings,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed, deterministic=True)
    output_dir.mkdir(parents=True, exist_ok=False)
    values = getattr(arrays, representation)
    feature_mean, feature_std = fit_feature_standardizer(
        values, arrays.mask, development_indices
    )
    development_dataset = RegionSetDataset(
        arrays,
        representation=representation,
        indices=development_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    test_dataset = RegionSetDataset(
        arrays,
        representation=representation,
        indices=test_indices,
        feature_mean=feature_mean,
        feature_std=feature_std,
    )
    development_metadata = arrays.metadata.iloc[development_indices]
    development_loader = make_loader(
        development_dataset,
        targets=arrays.target[development_indices],
        episode_ids=development_metadata["episode_id"].astype(str).tolist(),
        settings=settings,
        training=True,
        seed=seed,
    )
    test_loader = make_loader(
        test_dataset,
        targets=arrays.target[test_indices],
        episode_ids=arrays.metadata.iloc[test_indices]["episode_id"].astype(str).tolist(),
        settings=settings,
        training=False,
        seed=seed + 1,
    )
    model = build_model(
        architecture,
        input_dim=values.shape[2],
        settings=settings,
        device=device,
    )
    optimizer = AdamW(model.parameters(), lr=settings.learning_rate, weight_decay=settings.weight_decay)
    loss_function = nn.BCEWithLogitsLoss()
    history: list[dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(
            model,
            development_loader,
            optimizer,
            loss_function,
            device,
            settings.gradient_clip_norm,
        )
        history.append({"epoch": epoch, "train_loss": train_loss})
        print(f"    final epoch {epoch:02d}/{epochs:02d}: loss={train_loss:.4f}")
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    test_targets, test_probabilities, test_order = evaluate_neural(model, test_loader, device)
    expected_order = np.asarray(test_indices, dtype=np.int64)
    if not np.array_equal(test_order, expected_order):
        reorder = {int(index): position for position, index in enumerate(test_order)}
        test_probabilities = np.asarray(
            [test_probabilities[reorder[int(index)]] for index in expected_order],
            dtype=np.float64,
        )
        test_targets = arrays.target[expected_order]
    summary = metrics(test_targets, test_probabilities)
    torch.save(
        {
            "architecture": architecture,
            "representation": representation,
            "epochs": epochs,
            "model_state_dict": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
        },
        output_dir / "final_model.pt",
    )
    save_json(summary, output_dir / "metrics.json")
    return test_probabilities, summary


def fit_mean_logreg(
    arrays: RegionArrays,
    *,
    representation: str,
    development_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, Pipeline]:
    pooled = masked_mean_numpy(getattr(arrays, representation), arrays.mask)
    development_metadata = arrays.metadata.iloc[development_indices]
    weights = episode_balanced_weights(
        arrays.target[development_indices],
        development_metadata["episode_id"].astype(str).tolist(),
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=seed,
                ),
            ),
        ]
    )
    model.fit(
        pooled[development_indices],
        arrays.target[development_indices],
        classifier__sample_weight=weights,
    )
    probabilities = model.predict_proba(pooled[test_indices])[:, 1]
    return probabilities.astype(np.float64), model


def load_inner_manifest(sensor_run_root: Path, fold_index: int, direction: int) -> dict[str, Any]:
    path = sensor_run_root / f"fold_{fold_index}" / f"inner_{direction}_manifest.json"
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_results(fold_table: pd.DataFrame) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    metric_names = ("auprc", "no_skill_auprc", "auprc_lift", "auroc", "f1", "false_alarm_rate", "mdr_at_far_1pct", "mdr_at_far_5pct")
    higher_is_better = {
        "auprc",
        "no_skill_auprc",
        "auprc_lift",
        "auroc",
        "f1",
    }

    lower_is_better = {
        "false_alarm_rate",
        "mdr_at_far_1pct",
        "mdr_at_far_5pct",
    }
    for (representation, method), group in fold_table.groupby(
        ["representation", "method"], sort=True
    ):
        row: dict[str, Any] = {
            "representation": representation,
            "method": method,
            "n_folds": int(len(group)),
        }
        for name in metric_names:
            values = group[name].to_numpy(dtype=np.float64)

            row[f"{name}_mean"] = float(np.nanmean(values))
            row[f"{name}_std"] = float(
                np.nanstd(values, ddof=1)
            )

            if name in higher_is_better:
                row[f"{name}_worst"] = float(np.nanmin(values))
            elif name in lower_is_better:
                row[f"{name}_worst"] = float(np.nanmax(values))
            else:
                raise ValueError(
                    f"Unknown metric direction: {name}"
                )

            row[f"{name}_values"] = values.tolist()
        outputs.append(row)
    return outputs


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset_root = args.dataset_root or Path(config["region_dataset_root"])
    dataset_root = Path(dataset_root)
    sensor_run_root = Path(config["sensor_cv_run_root"])
    training_config = dict(config.get("region_training", {}))
    settings = read_neural_settings(training_config)
    seed = int(training_config.get("seed", 42))
    requested_device = args.device or training_config.get("device", "auto")
    device = resolve_device(str(requested_device))
    representations = list(training_config.get("representations", ["learned", "handcrafted"]))
    neural_methods = list(training_config.get("neural_methods", ["mean_mlp", "deepsets"]))
    include_logreg = bool(training_config.get("include_mean_logreg", True))
    include_local_average = bool(training_config.get("include_average_local_probability", True))
    n_folds = int(training_config.get("n_folds", 3))

    output_root = args.output_root or Path(
        training_config.get(
            "output_root",
            f"runs/region_cv_{datetime.now():%Y%m%d-%H%M%S}",
        )
    )
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    shutil.copy2(args.config, output_root / "config.yaml")

    print(f"Dataset root:   {dataset_root}")
    print(f"Sensor run:     {sensor_run_root}")
    print(f"Output root:    {output_root}")
    print(f"Device:         {device}")

    fold_rows: list[dict[str, Any]] = []
    complete_fold_results: list[dict[str, Any]] = []

    for fold_index in range(n_folds):
        print(f"\n=== Regional outer fold {fold_index + 1}/{n_folds} ===")
        fold_dataset_dir = dataset_root / f"fold_{fold_index}"
        arrays = load_region_arrays(
            fold_dataset_dir / f"region_dataset_fold{fold_index}.npz",
            fold_dataset_dir / f"region_manifest_fold{fold_index}.csv",
        )
        development_indices = np.flatnonzero(arrays.split_code == 0)
        test_indices = np.flatnonzero(arrays.split_code == 1)
        assert_disjoint_indices(development_indices, test_indices)
        test_metadata = arrays.metadata.iloc[test_indices].reset_index(drop=True)
        test_targets = arrays.target[test_indices]
        fold_output = output_root / f"fold_{fold_index}"
        fold_output.mkdir()
        prediction_table = test_metadata.copy()
        fold_result: dict[str, Any] = {"fold": fold_index, "models": []}

        if include_local_average:
            numerator = (arrays.local_probability[test_indices] * arrays.mask[test_indices]).sum(axis=1)
            denominator = arrays.mask[test_indices].sum(axis=1).clip(min=1)
            probabilities = numerator / denominator
            summary = metrics(test_targets, probabilities)
            prediction_table["probability__local_probability__average"] = probabilities
            result = {
                "representation": "local_probability",
                "method": "average",
                "metrics": summary,
                "by_scenario": by_scenario_metrics(test_metadata, test_targets, probabilities),
            }
            fold_result["models"].append(result)
            fold_rows.append(
                {"fold": fold_index, "representation": "local_probability", "method": "average", **summary}
            )
            print(
                f"  local probability average: AUPRC={summary['auprc']:.4f}, "
                f"AUROC={summary['auroc']:.4f}"
            )

        for representation in representations:
            if representation not in {"learned", "handcrafted"}:
                raise ValueError(f"Unknown representation: {representation!r}")

            if include_logreg:
                print(f"\n  {representation} + mean logistic regression")
                probabilities, model = fit_mean_logreg(
                    arrays,
                    representation=representation,
                    development_indices=development_indices,
                    test_indices=test_indices,
                    seed=seed + 100 * fold_index,
                )
                summary = metrics(test_targets, probabilities)
                import joblib

                model_dir = fold_output / representation / "mean_logreg"
                model_dir.mkdir(parents=True)
                joblib.dump(model, model_dir / "model.joblib")
                save_json(summary, model_dir / "metrics.json")
                prediction_table[f"probability__{representation}__mean_logreg"] = probabilities
                result = {
                    "representation": representation,
                    "method": "mean_logreg",
                    "metrics": summary,
                    "by_scenario": by_scenario_metrics(test_metadata, test_targets, probabilities),
                }
                fold_result["models"].append(result)
                fold_rows.append(
                    {"fold": fold_index, "representation": representation, "method": "mean_logreg", **summary}
                )
                print(f"    AUPRC={summary['auprc']:.4f}, AUROC={summary['auroc']:.4f}")

            for architecture in neural_methods:
                print(f"\n  {representation} + {architecture}")
                inner_results: list[dict[str, Any]] = []
                for direction in (0, 1):
                    manifest = load_inner_manifest(sensor_run_root, fold_index, direction)
                    train_indices = indices_for_episodes(arrays.metadata, manifest["train"])
                    val_indices = indices_for_episodes(arrays.metadata, manifest["val"])
                    train_indices = np.intersect1d(train_indices, development_indices)
                    val_indices = np.intersect1d(val_indices, development_indices)
                    print(f"    inner direction {direction}")
                    inner_result = neural_selection_run(
                        arrays,
                        representation=representation,
                        architecture=architecture,
                        train_indices=train_indices,
                        val_indices=val_indices,
                        settings=settings,
                        device=device,
                        seed=seed + 1000 * fold_index + 100 * direction,
                        output_dir=fold_output / representation / architecture / f"inner_{direction}",
                    )
                    inner_results.append(inner_result)
                selected_epochs = max(
                    1,
                    int(np.rint(np.mean([result["best_epoch"] for result in inner_results]))),
                )
                print(f"    final training for {selected_epochs} epoch(s)")
                probabilities, summary = neural_final_run(
                    arrays,
                    representation=representation,
                    architecture=architecture,
                    development_indices=development_indices,
                    test_indices=test_indices,
                    epochs=selected_epochs,
                    settings=settings,
                    device=device,
                    seed=seed + 10000 + 100 * fold_index,
                    output_dir=fold_output / representation / architecture / "final",
                )
                prediction_table[f"probability__{representation}__{architecture}"] = probabilities
                result = {
                    "representation": representation,
                    "method": architecture,
                    "selected_epochs": selected_epochs,
                    "inner_selection": inner_results,
                    "metrics": summary,
                    "by_scenario": by_scenario_metrics(test_metadata, test_targets, probabilities),
                }
                fold_result["models"].append(result)
                fold_rows.append(
                    {"fold": fold_index, "representation": representation, "method": architecture, **summary}
                )
                print(f"    AUPRC={summary['auprc']:.4f}, AUROC={summary['auroc']:.4f}")

        prediction_table.to_csv(
            fold_output / f"region_predictions_fold{fold_index}.csv",
            index=False,
        )
        save_json(fold_result, fold_output / f"region_results_fold{fold_index}.json")
        complete_fold_results.append(fold_result)

    fold_table = pd.DataFrame(fold_rows)
    fold_table.to_csv(output_root / "region_fold_metrics.csv", index=False)
    aggregate = aggregate_results(fold_table)
    pd.DataFrame(aggregate).to_csv(output_root / "region_aggregate_metrics.csv", index=False)
    save_json(
        {
            "folds": complete_fold_results,
            "aggregate": aggregate,
            "primary_metrics": ["auprc", "no_skill_auprc", "auprc_lift", "auroc"],
            "threshold_note": "F1 and error rates are secondary diagnostics at threshold 0.5.",
        },
        output_root / "region_summary.json",
    )

    print("\n=== Regional aggregate ===")
    aggregate_table = pd.DataFrame(aggregate)
    display_columns = [
        "representation",
        "method",
        "auprc_mean",
        "auprc_lift_mean",
        "auroc_mean",
        "auroc_worst",
    ]
    print(aggregate_table[display_columns].to_string(index=False))
    print(f"\nSaved regional receiver results to: {output_root}")


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
