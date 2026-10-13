from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np

from fire_ura.quantization import (
    build_uniform_per_component_quantizer,
    quantization_diagnostics,
    quantize_probability,
)


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "run_quantization_study.py"
)
SPEC = importlib.util.spec_from_file_location(
    "run_quantization_study",
    SCRIPT_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Could not import {SCRIPT_PATH}")
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_uniform_vector_payload_is_dimension_times_bpc() -> None:
    quantizer = build_uniform_per_component_quantizer(
        dimension=4,
        bits_per_component=3,
        lower=-4.0,
        upper=4.0,
    )
    assert quantizer.dimension == 4
    assert quantizer.bits_per_component == 3
    assert quantizer.total_bits == 12


def test_uniform_vector_quantizer_uses_common_bounds() -> None:
    values = np.asarray(
        [
            [-10.0, -4.0, 0.0, 10.0],
            [1.0, 2.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    quantizer = build_uniform_per_component_quantizer(
        dimension=4,
        bits_per_component=4,
        lower=-4.0,
        upper=4.0,
    )
    output = quantizer.transform(values)
    assert output.shape == values.shape
    assert np.all(output >= -4.0)
    assert np.all(output <= 4.0)


def test_more_uniform_bits_reduce_mse_on_same_values() -> None:
    rng = np.random.default_rng(2)
    values = rng.normal(size=(1000, 4)).astype(np.float32)
    low = build_uniform_per_component_quantizer(
        dimension=4,
        bits_per_component=2,
    )
    high = build_uniform_per_component_quantizer(
        dimension=4,
        bits_per_component=6,
    )
    low_mse = float(np.mean((values - low.transform(values)) ** 2))
    high_mse = float(np.mean((values - high.transform(values)) ** 2))
    assert high_mse <= low_mse + 1e-12


def test_diagnostics_report_fixed_payload_and_clip_fraction() -> None:
    values = np.asarray(
        [
            [-5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 5.0],
        ],
        dtype=np.float32,
    )
    quantizer = build_uniform_per_component_quantizer(
        dimension=4,
        bits_per_component=3,
        lower=-4.0,
        upper=4.0,
    )
    diag = quantization_diagnostics(values, quantizer)
    assert diag["task_bits"] == 12
    assert diag["dimension"] == 4
    assert np.isclose(diag["clip_fraction_elements"], 2.0 / 8.0)


def test_probability_quantization_is_bounded_and_monotone() -> None:
    values = np.linspace(-0.2, 1.2, 100, dtype=np.float32)
    quantized = quantize_probability(values, payload_bits=4)
    assert np.all(quantized >= 0.0)
    assert np.all(quantized <= 1.0)
    assert np.all(np.diff(quantized) >= 0.0)
    assert np.unique(quantized).size <= 16


def test_exact_k_mask_is_reproducible_and_selects_k() -> None:
    base = np.ones((20, 9), dtype=bool)
    first = MODULE.exact_k_mask_numpy(base, 3, 123)
    second = MODULE.exact_k_mask_numpy(base, 3, 123)
    third = MODULE.exact_k_mask_numpy(base, 3, 124)
    assert np.array_equal(first, second)
    assert np.all(first.sum(axis=1) == 3)
    assert not np.array_equal(first, third)


def test_message_accounting_includes_tag_nonce_and_k() -> None:
    settings = MODULE.Settings(
        learned_bits_per_component=(1, 2),
        handcrafted_bits_per_component=(1, 2),
        score_bits=(1, 2),
        k_values=(1, 5),
        test_subset_seeds=2,
        clip_lower=-4.0,
        clip_upper=4.0,
        region_tag_bits=6,
        nonce_bits=8,
        batch_size=32,
        save_predictions=False,
        hidden_dim=32,
        phi_hidden_dim=32,
        phi_output_dim=16,
        rho_hidden_dim=16,
        dropout=0.15,
        handcrafted_stats=("mean", "std"),
    )
    result = MODULE.message_accounting(
        task_bits=16,
        k=5,
        settings=settings,
    )
    assert result["message_bits"] == 30
    assert result["regional_task_bits"] == 80
    assert result["regional_message_bits"] == 150


def test_handcrafted_selection_preserves_source_order() -> None:
    names = [
        "temperature__mean",
        "co__mean",
        "temperature__std",
        "co__std",
        "temperature__slope",
        "co__slope",
    ]
    indices = MODULE.handcrafted_indices(
        names,
        ("mean", "std"),
    )
    assert indices.tolist() == [0, 1, 2, 3]


if __name__ == "__main__":
    started = time.perf_counter()
    try:
        test_uniform_vector_payload_is_dimension_times_bpc()
        test_uniform_vector_quantizer_uses_common_bounds()
        test_more_uniform_bits_reduce_mse_on_same_values()
        test_diagnostics_report_fixed_payload_and_clip_fraction()
        test_probability_quantization_is_bounded_and_monotone()
        test_exact_k_mask_is_reproducible_and_selects_k()
        test_message_accounting_includes_tag_nonce_and_k()
        test_handcrafted_selection_preserves_source_order()
        print("All TERRA quantization-study tests passed.")
    finally:
        print(
            f"Total runtime: {time.perf_counter() - started:.2f} seconds"
        )
