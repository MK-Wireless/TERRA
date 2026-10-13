#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCHEMES = ("TERRA-Score", "TERRA-Feature", "Handcrafted-Stats")
SELECTED_BITS = {
    "TERRA-Score": 6,
    "TERRA-Feature": 24,
    "Handcrafted-Stats": 30,
}
MARKERS = {
    "TERRA-Score": "o",
    "TERRA-Feature": "s",
    "Handcrafted-Stats": "^",
}
LINESTYLES = {
    "TERRA-Score": "-",
    "TERRA-Feature": "--",
    "Handcrafted-Stats": "-.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final two-panel TERRA Figure 2 from existing quantization results."
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path("paper_results/quantization/quantization_aggregate_metrics.csv"),
        help="Final mid-rise quantization_aggregate_metrics.csv",
    )
    parser.add_argument(
        "--float-reference",
        type=Path,
        default=Path("paper_results/quantization/quantization_float_reference_metrics.csv"),
        help="Final quantization_float_reference_metrics.csv",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/payload_and_reports"),
        help="Output path without extension",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def load_table(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    table = pd.read_csv(path)
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(missing)}")
    return table


def plot() -> None:
    args = parse_args()
    started = time.perf_counter()

    aggregate = load_table(
        args.aggregate,
        {"scheme", "k", "task_bits", "auprc_mean", "auprc_std"},
    )
    float_ref = load_table(
        args.float_reference,
        {"scheme", "k", "auprc_mean", "auprc_std"},
    )

    # Sanity checks: this figure is specifically for the final three pipelines.
    missing_schemes = [s for s in SCHEMES if s not in set(aggregate["scheme"])]
    if missing_schemes:
        raise ValueError(f"Missing schemes in aggregate table: {missing_schemes}")

    selected_rows = []
    for scheme, bits in SELECTED_BITS.items():
        rows = aggregate.loc[
            (aggregate["scheme"] == scheme)
            & (aggregate["task_bits"] == bits)
        ].copy()
        if rows.empty:
            raise ValueError(f"No rows for {scheme} at selected task_bits={bits}")
        selected_rows.append(rows)
    selected = pd.concat(selected_rows, ignore_index=True)

    k_values = sorted(selected["k"].astype(int).unique().tolist())
    expected_k = [1, 2, 3, 5, 7, 9]
    if k_values != expected_k:
        raise ValueError(f"Expected K values {expected_k}, found {k_values}")

    fig, axes = plt.subplots(1, 2, figsize=(7.25, 3.05))
    ax_rate, ax_k = axes

    # Panel (a): quantization-rate sweep at full report availability K=9.
    rate_rows = aggregate.loc[aggregate["k"].astype(int) == 9].copy()
    for scheme in SCHEMES:
        group = rate_rows.loc[rate_rows["scheme"] == scheme].sort_values("task_bits")
        x = group["task_bits"].to_numpy(dtype=float)
        y = group["auprc_mean"].to_numpy(dtype=float)
        std = group["auprc_std"].to_numpy(dtype=float)
        (line,) = ax_rate.plot(
            x,
            y,
            marker=MARKERS[scheme],
            linestyle=LINESTYLES[scheme],
            linewidth=1.45,
            markersize=4.5,
            label=scheme,
        )
        ax_rate.fill_between(x, y - std, y + std, color=line.get_color(), alpha=0.10, linewidth=0)

        # Highlight the selected task-payload operating point.
        bits = SELECTED_BITS[scheme]
        chosen = group.loc[group["task_bits"] == bits]
        if len(chosen) != 1:
            raise ValueError(f"Expected exactly one K=9 row for {scheme}, task_bits={bits}")
        ax_rate.scatter(
            [bits],
            [float(chosen.iloc[0]["auprc_mean"])],
            s=58,
            facecolors="none",
            edgecolors=line.get_color(),
            linewidths=1.2,
            zorder=5,
        )

        # Float reference: short dashed horizontal segment spanning the scheme's rate range.
        frow = float_ref.loc[
            (float_ref["scheme"] == scheme) & (float_ref["k"].astype(int) == 9)
        ]
        if len(frow) != 1:
            raise ValueError(f"Expected exactly one float reference for {scheme}, K=9")
        fval = float(frow.iloc[0]["auprc_mean"])
        xmin, xmax = float(np.min(x)), float(np.max(x))
        ax_rate.hlines(
            fval,
            xmin=xmin,
            xmax=xmax,
            colors=line.get_color(),
            linestyles=":",
            linewidth=0.9,
            alpha=0.75,
        )

    ax_rate.set_xscale("log", base=2)
    ax_rate.set_xlabel("Task payload bits/report")
    ax_rate.set_ylabel("AUPRC")
    ax_rate.set_ylim(0.90, 1.002)
    ax_rate.grid(True, alpha=0.22)
    ax_rate.text(0.02, 0.97, "(a)", transform=ax_rate.transAxes, ha="left", va="top", fontweight="bold")

    # Panel (b): report availability at the locked selected payloads.
    for scheme in SCHEMES:
        group = selected.loc[selected["scheme"] == scheme].sort_values("k")
        x = group["k"].to_numpy(dtype=int)
        y = group["auprc_mean"].to_numpy(dtype=float)
        std = group["auprc_std"].to_numpy(dtype=float)
        (line,) = ax_k.plot(
            x,
            y,
            marker=MARKERS[scheme],
            linestyle=LINESTYLES[scheme],
            linewidth=1.45,
            markersize=4.5,
            # label=f"{scheme} ({SELECTED_BITS[scheme]} bit)",
            label=scheme,
        )
        ax_k.fill_between(x, y - std, y + std, color=line.get_color(), alpha=0.10, linewidth=0)

    ax_k.set_xlabel(r"Number of available reports, $K$")
    ax_k.set_ylabel("AUPRC")
    ax_k.set_xticks(expected_k)
    ax_k.set_ylim(0.90, 1.002)
    ax_k.grid(True, alpha=0.22)
    ax_k.text(0.02, 0.97, "(b)", transform=ax_k.transAxes, ha="left", va="top", fontweight="bold")

    # One shared legend above both panels; no figure title or embedded caption.
    # handles, labels = ax_k.get_legend_handles_labels()
    # fig.legend(
    #     handles,
    #     labels,
    #     loc="upper center",
    #     bbox_to_anchor=(0.5, 1.035),
    #     ncol=3,
    #     frameon=False,
    #     fontsize=8,
    #     handlelength=2.6,
    #     columnspacing=1.2,
    # )

    ax_rate.legend(
        loc="lower left",
        frameon=False,
        fontsize=7,
        handlelength=2.0,
    )

    ax_k.legend(
        loc="lower right",
        frameon=False,
        fontsize=7,
        handlelength=2.0,
    )

    # fig.subplots_adjust(left=0.09, right=0.99, bottom=0.17, top=0.83, wspace=0.28)
    fig.subplots_adjust(
        left=0.09,
        right=0.99,
        bottom=0.17,
        top=0.96,
        wspace=0.28,
    )

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats:
        out = args.output_stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = args.dpi
        fig.savefig(out, **kwargs)
        print(f"Saved: {out}")

    plt.close(fig)
    elapsed = time.perf_counter() - started
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    plot()
