#!/usr/bin/env python
from __future__ import annotations

"""
Synthetic unsourced-random-access (URA) list-recovery impairments.

This module implements the abstraction used by the TERRA application study:

    transmitted global message set W
        -> URA list recovery characterized by (P_MD, P_FA)
        -> recovered global list W_hat
        -> region-tag grouping
        -> regional task inference

Definitions
-----------
For distinct transmitted/recovered full messages,

    P_MD = E[ |W \\ W_hat| / |W| ]

    P_FA = E[ |W_hat \\ W| / |W_hat| ]

The synthetic layer deliberately models only list-level missed detections (MD)
and false alarms (FA). It does NOT introduce a separate payload-bit-error
process. A wrong decoded version of a genuine transmitted message is already
represented at the list level by one MD plus one FA.

False alarms
------------
A false alarm is a new full decoded message that was not transmitted.

* Its region tag is drawn uniformly from the valid region-tag alphabet.
* Its task payload is drawn uniformly from the valid quantized task-payload
  alphabet for the scheme being evaluated.
* The task payload MAY equal a genuine task payload. This is intentional:
  full-message identity includes the nonce/message identity, so a false
  message can carry the same task payload as a genuine message.
* Synthetic negative nonce/message IDs are used only as bookkeeping to
  guarantee that every FA full message is disjoint from the true full-message
  set. Their bit length does not affect the imposed synthetic MD/FA rates.

Because only one regional example is materialized by the dataset, other
regions' true messages are represented only through their activity count.
They affect the GLOBAL number of recovered true messages and hence the global
FA count. False messages are then uniformly tagged, so only those whose random
tag matches the target region are injected into its downstream receiver.

The default FA-count construction uses "ratio-matched randomized rounding".
For a given number T of correctly recovered true global messages, it draws an
integer F from the two adjacent counts whose decoded-list false fractions
bracket the requested P_FA. The mixing probability is chosen so that

    E[ F / (T + F) | T ] = requested P_FA.

Thus the simulation parameter matches the list-level FA definition directly,
rather than treating P_FA as a Bernoulli probability over nontransmitted
messages.
"""

from dataclasses import dataclass
import math
import time
from typing import Literal

import numpy as np


ReconstructionMode = Literal["endpoints", "midrise"]


@dataclass(frozen=True)
class RecoveryPlan:
    """One synthetic global URA recovery realization for one time slot."""

    target_keep_mask: np.ndarray
    target_fa_count: int
    target_fa_synthetic_nonce_ids: np.ndarray

    global_true_count: int
    global_true_recovered_count: int
    global_false_count: int

    realized_global_md: float
    realized_global_fa: float
    realized_target_md: float

    @property
    def target_true_recovered_count(self) -> int:
        return int(np.sum(self.target_keep_mask))

    @property
    def target_decoded_count(self) -> int:
        return int(self.target_true_recovered_count + self.target_fa_count)


def _validate_probability(value: float, name: str, *, allow_one: bool = True) -> float:
    value = float(value)
    upper_ok = value <= 1.0 if allow_one else value < 1.0
    if not math.isfinite(value) or value < 0.0 or not upper_ok:
        comparator = "[0,1]" if allow_one else "[0,1)"
        raise ValueError(f"{name} must lie in {comparator}, found {value}.")
    return value


def sample_false_count_ratio_matched(
    true_recovered_count: int,
    p_fa: float,
    rng: np.random.Generator,
) -> int:
    """Sample integer FA count with conditional mean FA fraction equal to p_fa.

    For T > 0, let r(F)=F/(T+F). We choose F from floor(F*) or ceil(F*),
    where F*=p_fa*T/(1-p_fa), and mix the two values so that E[r(F)|T]=p_fa.

    For T=0, the realized false fraction is defined as 0 for an empty decoded
    list and 1 for a nonempty all-false decoded list. A Bernoulli(p_fa)
    construction therefore matches the desired expectation exactly.
    """
    t = int(true_recovered_count)
    if t < 0:
        raise ValueError("true_recovered_count must be nonnegative.")
    p = _validate_probability(p_fa, "p_fa", allow_one=False)

    if p == 0.0:
        return 0

    if t == 0:
        return int(rng.random() < p)

    f_star = p * t / (1.0 - p)
    f_low = int(math.floor(f_star))
    f_high = int(math.ceil(f_star))

    if f_low == f_high:
        return f_low

    r_low = f_low / (t + f_low)
    r_high = f_high / (t + f_high)

    if not (r_low <= p <= r_high):
        raise RuntimeError(
            f"Internal FA rounding error: r_low={r_low}, p={p}, r_high={r_high}."
        )

    prob_high = (p - r_low) / (r_high - r_low)
    return f_high if rng.random() < prob_high else f_low


def sample_recovery_plan(
    *,
    target_true_count: int,
    active_users_per_other_region: int,
    num_regions: int,
    p_md: float,
    p_fa: float,
    rng: np.random.Generator,
) -> RecoveryPlan:
    """Draw one paired MD/FA recovery plan for the target regional example.

    Assumptions:
      * the target region contains ``target_true_count`` transmitted messages;
      * each of the other valid regions contains
        ``active_users_per_other_region`` transmitted messages;
      * every true full message is independently missed with probability p_md;
      * P_FA is defined globally at the URA decoder output;
      * FA region tags are uniform over the ``num_regions`` valid region tags.

    Only the target-region true-message keep mask is materialized. Other
    regions contribute through a binomial recovered-true count, which is
    sufficient to determine the global FA count.
    """
    k_target = int(target_true_count)
    k_other = int(active_users_per_other_region)
    r = int(num_regions)
    p_md = _validate_probability(p_md, "p_md", allow_one=True)
    p_fa = _validate_probability(p_fa, "p_fa", allow_one=False)

    if k_target < 0:
        raise ValueError("target_true_count must be nonnegative.")
    if k_other < 0:
        raise ValueError("active_users_per_other_region must be nonnegative.")
    if r <= 0:
        raise ValueError("num_regions must be positive.")

    target_keep_mask = rng.random(k_target) >= p_md
    target_recovered = int(np.sum(target_keep_mask))

    other_true_count = (r - 1) * k_other
    other_recovered = int(
        rng.binomial(other_true_count, 1.0 - p_md)
    ) if other_true_count > 0 else 0

    global_true_count = k_target + other_true_count
    global_true_recovered = target_recovered + other_recovered

    global_false_count = sample_false_count_ratio_matched(
        global_true_recovered,
        p_fa,
        rng,
    )

    # Equivalent to explicitly drawing every FA tag uniformly from the valid
    # region-tag alphabet and counting how many land in the target region.
    target_fa_count = int(
        rng.binomial(global_false_count, 1.0 / r)
    ) if global_false_count > 0 else 0

    # Negative IDs are an emulator-only namespace disjoint from genuine
    # message IDs. They guarantee full-message uniqueness without making the
    # synthetic error rates depend on any physical nonce length.
    target_fa_nonce_ids = -np.arange(
        1,
        target_fa_count + 1,
        dtype=np.int64,
    )

    if global_true_count > 0:
        realized_global_md = 1.0 - (
            global_true_recovered / global_true_count
        )
    else:
        realized_global_md = 0.0

    decoded_global_count = global_true_recovered + global_false_count
    realized_global_fa = (
        global_false_count / decoded_global_count
        if decoded_global_count > 0
        else 0.0
    )

    realized_target_md = (
        1.0 - target_recovered / k_target
        if k_target > 0
        else 0.0
    )

    return RecoveryPlan(
        target_keep_mask=target_keep_mask.astype(bool),
        target_fa_count=target_fa_count,
        target_fa_synthetic_nonce_ids=target_fa_nonce_ids,
        global_true_count=global_true_count,
        global_true_recovered_count=global_true_recovered,
        global_false_count=global_false_count,
        realized_global_md=float(realized_global_md),
        realized_global_fa=float(realized_global_fa),
        realized_target_md=float(realized_target_md),
    )


def decode_uniform_codes(
    codes: np.ndarray,
    *,
    bits: int,
    lower: float,
    upper: float,
    reconstruction_mode: ReconstructionMode,
) -> np.ndarray:
    """Decode integer scalar-quantizer codes into reconstruction values."""
    bits = int(bits)
    lower = float(lower)
    upper = float(upper)

    if bits <= 0:
        raise ValueError("bits must be positive.")
    if not math.isfinite(lower) or not math.isfinite(upper) or upper <= lower:
        raise ValueError("Expected finite quantizer bounds with upper > lower.")
    if reconstruction_mode not in {"endpoints", "midrise"}:
        raise ValueError(
            "reconstruction_mode must be 'endpoints' or 'midrise'."
        )

    n_levels = 1 << bits
    codes = np.asarray(codes, dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= n_levels):
        raise ValueError("A quantizer code lies outside the valid alphabet.")

    if reconstruction_mode == "endpoints":
        if n_levels == 1:
            return np.full(codes.shape, (lower + upper) / 2.0, dtype=np.float64)
        step = (upper - lower) / (n_levels - 1)
        return lower + codes.astype(np.float64) * step

    step = (upper - lower) / n_levels
    return lower + (codes.astype(np.float64) + 0.5) * step


def sample_uniform_task_payload(
    *,
    n_messages: int,
    dimension: int,
    bits_per_component: int,
    lower: float,
    upper: float,
    reconstruction_mode: ReconstructionMode,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw uniformly random valid task words and return decoded values.

    Drawing every component code uniformly is equivalent to drawing a uniform
    bit string over the complete D*b-bit task-payload alphabet.

    The returned task word is NOT forced to differ from genuine task payloads.
    Full-message uniqueness is supplied by the synthetic nonce/message ID, as
    it should be for an unsourced list-decoding abstraction.
    """
    n = int(n_messages)
    d = int(dimension)
    b = int(bits_per_component)
    if n < 0:
        raise ValueError("n_messages must be nonnegative.")
    if d <= 0:
        raise ValueError("dimension must be positive.")
    if b <= 0:
        raise ValueError("bits_per_component must be positive.")

    if n == 0:
        return np.empty((0, d), dtype=np.float32)

    n_levels = 1 << b
    codes = rng.integers(
        0,
        n_levels,
        size=(n, d),
        dtype=np.int64,
    )
    decoded = decode_uniform_codes(
        codes,
        bits=b,
        lower=lower,
        upper=upper,
        reconstruction_mode=reconstruction_mode,
    )
    return decoded.astype(np.float32)


def _self_check() -> None:
    started = time.perf_counter()
    rng = np.random.default_rng(7)

    # Verify the 1-bit mid-rise levels used by the handcrafted operating point.
    payload = sample_uniform_task_payload(
        n_messages=1000,
        dimension=1,
        bits_per_component=1,
        lower=-4.0,
        upper=4.0,
        reconstruction_mode="midrise",
        rng=rng,
    )
    observed = sorted(set(float(v) for v in payload[:, 0]))
    assert observed == [-2.0, 2.0], observed

    # Verify that ratio-matched rounding reproduces the desired list-level FA
    # fraction in expectation.
    target_pfa = 0.2
    realized = []
    for _ in range(20000):
        f = sample_false_count_ratio_matched(30, target_pfa, rng)
        realized.append(f / (30 + f))
    assert abs(float(np.mean(realized)) - target_pfa) < 0.003

    print("URA impairment module self-check passed.")
    print(f"Total runtime: {time.perf_counter() - started:.3f} seconds")


if __name__ == "__main__":
    _self_check()
