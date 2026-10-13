from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler


@dataclass(frozen=True)
class RegionArrays:
    learned: np.ndarray
    handcrafted: np.ndarray
    local_probability: np.ndarray
    mask: np.ndarray
    target: np.ndarray
    split_code: np.ndarray
    metadata: pd.DataFrame

    def validate(self) -> None:
        n_examples = int(self.target.shape[0])
        if self.learned.ndim != 3 or self.handcrafted.ndim != 3:
            raise ValueError("Learned and handcrafted arrays must be [N, K, D].")
        if self.local_probability.ndim != 2 or self.mask.ndim != 2:
            raise ValueError("Local probability and mask arrays must be [N, K].")
        if self.learned.shape[:2] != self.mask.shape:
            raise ValueError("Learned representation and mask shapes disagree.")
        if self.handcrafted.shape[:2] != self.mask.shape:
            raise ValueError("Handcrafted representation and mask shapes disagree.")
        if self.local_probability.shape != self.mask.shape:
            raise ValueError("Local probabilities and mask shapes disagree.")
        for array_name, array in (
            ("learned", self.learned),
            ("handcrafted", self.handcrafted),
            ("local_probability", self.local_probability),
            ("mask", self.mask),
            ("split_code", self.split_code),
        ):
            if array.shape[0] != n_examples:
                raise ValueError(f"{array_name} has the wrong number of examples.")
        if len(self.metadata) != n_examples:
            raise ValueError("Metadata row count does not match array count.")
        set_sizes = self.mask.sum(axis=1)
        if np.any(set_sizes < 1):
            raise ValueError("Every regional example must contain at least one report.")
        if not np.all(np.isfinite(self.learned[self.mask])):
            raise ValueError("Learned representations contain non-finite values.")
        if not np.all(np.isfinite(self.handcrafted[self.mask])):
            raise ValueError("Handcrafted representations contain non-finite values.")
        if not np.all(np.isfinite(self.local_probability[self.mask])):
            raise ValueError("Local probabilities contain non-finite values.")
        if not set(np.unique(self.target)).issubset({0, 1}):
            raise ValueError("Regional targets must be binary.")
        if not set(np.unique(self.split_code)).issubset({0, 1}):
            raise ValueError("split_code must use 0=development and 1=test.")


def save_region_arrays(arrays: RegionArrays, npz_path: Path, metadata_path: Path) -> None:
    arrays.validate()
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        learned=arrays.learned.astype(np.float32),
        handcrafted=arrays.handcrafted.astype(np.float32),
        local_probability=arrays.local_probability.astype(np.float32),
        mask=arrays.mask.astype(np.bool_),
        target=arrays.target.astype(np.int64),
        split_code=arrays.split_code.astype(np.int8),
    )
    arrays.metadata.to_csv(metadata_path, index=False)


def load_region_arrays(npz_path: Path, metadata_path: Path) -> RegionArrays:
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = RegionArrays(
            learned=payload["learned"],
            handcrafted=payload["handcrafted"],
            local_probability=payload["local_probability"],
            mask=payload["mask"].astype(bool),
            target=payload["target"].astype(np.int64),
            split_code=payload["split_code"].astype(np.int8),
            metadata=pd.read_csv(metadata_path),
        )
    arrays.validate()
    return arrays


def masked_mean_numpy(values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if values.ndim != 3 or mask.ndim != 2:
        raise ValueError("Expected values [N,K,D] and mask [N,K].")
    weights = mask.astype(values.dtype)[..., None]
    denominator = weights.sum(axis=1).clip(min=1.0)
    return (values * weights).sum(axis=1) / denominator


class RegionSetDataset(Dataset):
    """A dense padded set dataset with one mask per regional time window."""

    def __init__(
        self,
        arrays: RegionArrays,
        *,
        representation: str,
        indices: Sequence[int] | None = None,
        feature_mean: np.ndarray | None = None,
        feature_std: np.ndarray | None = None,
    ) -> None:
        arrays.validate()
        if representation not in {"learned", "handcrafted"}:
            raise ValueError("representation must be 'learned' or 'handcrafted'.")
        self.arrays = arrays
        self.representation = representation
        self.indices = np.asarray(
            np.arange(len(arrays.target)) if indices is None else indices,
            dtype=np.int64,
        )
        self.feature_mean = None if feature_mean is None else np.asarray(feature_mean)
        self.feature_std = None if feature_std is None else np.asarray(feature_std)
        if (self.feature_mean is None) != (self.feature_std is None):
            raise ValueError("feature_mean and feature_std must be supplied together.")

    def __len__(self) -> int:
        return int(self.indices.size)

    def __getitem__(self, item: int) -> dict[str, Any]:
        index = int(self.indices[item])
        values = getattr(self.arrays, self.representation)[index].astype(np.float32).copy()
        mask = self.arrays.mask[index].astype(bool).copy()
        if self.feature_mean is not None and self.feature_std is not None:
            values[mask] = (values[mask] - self.feature_mean) / self.feature_std
            values[~mask] = 0.0
        metadata = self.arrays.metadata.iloc[index]
        return {
            "x": torch.from_numpy(values),
            "mask": torch.from_numpy(mask),
            "target": torch.tensor(float(self.arrays.target[index]), dtype=torch.float32),
            "index": index,
            "episode_id": str(metadata["episode_id"]),
        }


class EpisodeBalancedRegionSampler(Sampler[int]):
    """Sample class, episode, and regional window uniformly in that order."""

    def __init__(
        self,
        targets: Sequence[int],
        episode_ids: Sequence[str],
        *,
        num_samples: int,
        seed: int,
    ) -> None:
        self.targets = np.asarray(targets, dtype=np.int64)
        self.episode_ids = np.asarray([str(value) for value in episode_ids], dtype=object)
        self.num_samples = int(num_samples)
        self.seed = int(seed)
        self.epoch = 0
        if self.targets.size != self.episode_ids.size:
            raise ValueError("targets and episode_ids must have the same length.")
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        self.windows: dict[int, dict[str, list[int]]] = {0: {}, 1: {}}
        for index, (target, episode_id) in enumerate(zip(self.targets, self.episode_ids)):
            self.windows[int(target)].setdefault(str(episode_id), []).append(index)
        for target in (0, 1):
            if not self.windows[target]:
                raise ValueError(f"No regional examples found for class {target}.")
        self.episodes = {
            target: sorted(self.windows[target])
            for target in (0, 1)
        }

    def __len__(self) -> int:
        return self.num_samples

    def __iter__(self) -> Iterable[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        labels = np.resize(np.asarray([0, 1], dtype=np.int64), self.num_samples)
        rng.shuffle(labels)
        for label_value in labels:
            label = int(label_value)
            episode_id = self.episodes[label][int(rng.integers(len(self.episodes[label])))]
            candidates = self.windows[label][episode_id]
            yield int(candidates[int(rng.integers(len(candidates)))])


def fit_feature_standardizer(
    values: np.ndarray,
    mask: np.ndarray,
    indices: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    selected_values = values[np.asarray(indices, dtype=np.int64)]
    selected_mask = mask[np.asarray(indices, dtype=np.int64)]
    valid = selected_values[selected_mask]
    if valid.size == 0:
        raise ValueError("No valid sensor reports were available for standardization.")
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0, ddof=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    return mean, std


def indices_for_episodes(
    metadata: pd.DataFrame,
    episode_ids: Sequence[str],
) -> np.ndarray:
    requested = {str(value) for value in episode_ids}
    return np.flatnonzero(metadata["episode_id"].astype(str).isin(requested).to_numpy())


def assert_disjoint_indices(first: Sequence[int], second: Sequence[int]) -> None:
    overlap = set(map(int, first)).intersection(map(int, second))
    if overlap:
        raise ValueError(f"Dataset index leakage detected; example index {min(overlap)}.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate one regional dataset file.")
    parser.add_argument("npz", type=Path)
    parser.add_argument("metadata", type=Path)
    return parser.parse_args()


def _main() -> None:
    args = _parse_args()
    arrays = load_region_arrays(args.npz, args.metadata)
    print(f"Examples:          {len(arrays.target)}")
    print(f"Maximum set size:  {arrays.mask.shape[1]}")
    print(f"Learned dimension: {arrays.learned.shape[2]}")
    print(f"Handcrafted dim:   {arrays.handcrafted.shape[2]}")
    print(f"Development:       {(arrays.split_code == 0).sum()}")
    print(f"Test:              {(arrays.split_code == 1).sum()}")
    counts = arrays.metadata["set_size"].value_counts().sort_index() if "set_size" in arrays.metadata else None
    if counts is not None:
        print("Set-size counts:")
        for set_size, count in counts.items():
            print(f"  {int(set_size)} sensors: {int(count):,} examples")


if __name__ == "__main__":
    _started = time.perf_counter()
    try:
        _main()
    finally:
        _elapsed = time.perf_counter() - _started
        print(f"Total runtime: {_elapsed:.2f} seconds")
