#!/usr/bin/env python
from __future__ import annotations

import time

import numpy as np

from fire_ura.ura_impairments import (
    decode_uniform_codes,
    sample_false_count_ratio_matched,
    sample_recovery_plan,
    sample_uniform_task_payload,
)


def test_midrise_one_bit_levels() -> None:
    values = decode_uniform_codes(
        np.asarray([0, 1]),
        bits=1,
        lower=-4.0,
        upper=4.0,
        reconstruction_mode="midrise",
    )
    np.testing.assert_allclose(values, [-2.0, 2.0])


def test_endpoint_score_levels() -> None:
    values = decode_uniform_codes(
        np.asarray([0, 63]),
        bits=6,
        lower=0.0,
        upper=1.0,
        reconstruction_mode="endpoints",
    )
    np.testing.assert_allclose(values, [0.0, 1.0])


def test_zero_impairment_keeps_target_and_adds_no_fa() -> None:
    rng = np.random.default_rng(1)
    plan = sample_recovery_plan(
        target_true_count=9,
        active_users_per_other_region=9,
        num_regions=8,
        p_md=0.0,
        p_fa=0.0,
        rng=rng,
    )
    assert np.all(plan.target_keep_mask)
    assert plan.global_true_recovered_count == 72
    assert plan.global_false_count == 0
    assert plan.target_fa_count == 0
    assert plan.realized_global_md == 0.0
    assert plan.realized_global_fa == 0.0


def test_ratio_matched_fa_fraction() -> None:
    rng = np.random.default_rng(2)
    requested = 0.20
    realized = []
    for _ in range(50000):
        f = sample_false_count_ratio_matched(
            true_recovered_count=30,
            p_fa=requested,
            rng=rng,
        )
        realized.append(f / (30 + f))
    assert abs(float(np.mean(realized)) - requested) < 0.002


def test_random_fa_tags_thin_to_target_region() -> None:
    rng = np.random.default_rng(3)
    counts = []
    global_counts = []
    for _ in range(20000):
        plan = sample_recovery_plan(
            target_true_count=9,
            active_users_per_other_region=9,
            num_regions=8,
            p_md=0.1,
            p_fa=0.3,
            rng=rng,
        )
        counts.append(plan.target_fa_count)
        global_counts.append(plan.global_false_count)

    mean_target = float(np.mean(counts))
    mean_global = float(np.mean(global_counts))
    assert mean_global > 0.0
    assert abs(mean_target / mean_global - 1.0 / 8.0) < 0.01


def test_false_full_message_ids_are_disjoint_synthetic_namespace() -> None:
    rng = np.random.default_rng(4)
    for _ in range(1000):
        plan = sample_recovery_plan(
            target_true_count=9,
            active_users_per_other_region=9,
            num_regions=4,
            p_md=0.2,
            p_fa=0.4,
            rng=rng,
        )
        ids = plan.target_fa_synthetic_nonce_ids
        assert np.all(ids < 0)
        assert len(np.unique(ids)) == len(ids)


def test_uniform_task_payload_shape_and_alphabet() -> None:
    rng = np.random.default_rng(5)
    values = sample_uniform_task_payload(
        n_messages=5000,
        dimension=30,
        bits_per_component=1,
        lower=-4.0,
        upper=4.0,
        reconstruction_mode="midrise",
        rng=rng,
    )
    assert values.shape == (5000, 30)
    observed = set(np.unique(values).tolist())
    assert observed == {-2.0, 2.0}


def _run_as_script() -> None:
    started = time.perf_counter()
    test_midrise_one_bit_levels()
    test_endpoint_score_levels()
    test_zero_impairment_keeps_target_and_adds_no_fa()
    test_ratio_matched_fa_fraction()
    test_random_fa_tags_thin_to_target_region()
    test_false_full_message_ids_are_disjoint_synthetic_namespace()
    test_uniform_task_payload_shape_and_alphabet()
    print("All URA impairment tests passed.")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    _run_as_script()
