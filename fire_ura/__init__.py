"""Fire-URA: task-oriented finite-bit sensing for region-wise fire detection."""

from .data import (
    DatasetSchema,
    FireWindowDataset,
    NormalizationStats,
    derive_episode_ids,
    load_sensor_table,
)

__all__ = [
    "DatasetSchema",
    "FireWindowDataset",
    "NormalizationStats",
    "derive_episode_ids",
    "load_sensor_table",
]

__version__ = "0.1.0"
