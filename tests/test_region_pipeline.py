from __future__ import annotations

import time

import numpy as np
import pandas as pd
import torch

from fire_ura.region_data import RegionArrays, load_region_arrays, save_region_arrays
from fire_ura.region_model import DeepSetsClassifier, DeepSetsConfig, masked_mean
from fire_ura.representation import handcrafted_temporal_features


def test_handcrafted_feature_dimension() -> None:
    x = torch.randn(5, 4, 12)
    features = handcrafted_temporal_features(x)
    assert features.shape == (5, 28)


def test_masked_mean_ignores_padding() -> None:
    x = torch.tensor([[[1.0], [3.0], [100.0]]])
    mask = torch.tensor([[True, True, False]])
    pooled = masked_mean(x, mask)
    assert torch.allclose(pooled, torch.tensor([[2.0]]))


def test_deepsets_is_permutation_invariant() -> None:
    torch.manual_seed(1)
    model = DeepSetsClassifier(DeepSetsConfig(input_dim=3, dropout=0.0))
    model.eval()
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, True, False], [True, True, False, False]])
    permutation = torch.tensor([2, 0, 3, 1])
    original = model(x, mask)
    permuted = model(x[:, permutation], mask[:, permutation])
    assert torch.allclose(original, permuted, atol=1e-6)


def test_region_arrays_round_trip(tmp_path) -> None:
    learned = np.zeros((3, 2, 4), dtype=np.float32)
    handcrafted = np.zeros((3, 2, 28), dtype=np.float32)
    probability = np.full((3, 2), 0.5, dtype=np.float32)
    mask = np.asarray([[1, 1], [1, 0], [1, 1]], dtype=bool)
    target = np.asarray([0, 1, 0], dtype=np.int64)
    split_code = np.asarray([0, 0, 1], dtype=np.int8)
    metadata = pd.DataFrame(
        {
            "episode_id": ["a", "b", "c"],
            "scenario_label": ["Background", "Cable", "Background"],
        }
    )
    arrays = RegionArrays(
        learned=learned,
        handcrafted=handcrafted,
        local_probability=probability,
        mask=mask,
        target=target,
        split_code=split_code,
        metadata=metadata,
    )
    npz_path = tmp_path / "data.npz"
    csv_path = tmp_path / "data.csv"
    save_region_arrays(arrays, npz_path, csv_path)
    loaded = load_region_arrays(npz_path, csv_path)
    assert loaded.learned.shape == (3, 2, 4)
    assert loaded.handcrafted.shape == (3, 2, 28)
    assert loaded.mask.sum(axis=1).tolist() == [2, 1, 2]


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        raise SystemExit("Run this file with: python -m pytest tests/test_region_pipeline.py -q")
    finally:
        print(f"Total runtime: {time.perf_counter() - started:.2f} seconds")
