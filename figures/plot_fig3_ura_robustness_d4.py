#!/usr/bin/env python
from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.transforms import ScaledTranslation
import numpy as np
import pandas as pd


SCHEMES = (
    ("TERRA-Score", "(a) TERRA-Score"),
    ("TERRA-Feature", "(b) TERRA-Feature"),
    ("Handcrafted-Stats", "(c) Handcrafted-Stats"),
)

CONTOUR_LEVELS = (0.80, 0.90, 0.95, 0.98)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the final three-panel TERRA Figure 3 from existing URA impairment results."
    )
    parser.add_argument(
        "--aggregate",
        type=Path,
        default=Path(
            "paper_results/ura_impairment/"
            "ura_impairment_aggregate_metrics.csv"
        ),
        help="Final extended URA aggregate metrics CSV.",
    )
    parser.add_argument(
        "--output-stem",
        type=Path,
        default=Path("figures/ura_robustness"),
        help="Output path without extension.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("pdf", "png", "svg"),
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--vmin",
        type=float,
        default=0.40,
        help="Common lower AUPRC limit for all panels.",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        default=1.00,
        help="Common upper AUPRC limit for all panels.",
    )
    return parser.parse_args()


def load_aggregate(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)

    table = pd.read_csv(path)
    required = {"scheme", "p_md", "p_fa", "auprc_mean"}
    missing = sorted(required.difference(table.columns))
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {', '.join(missing)}"
        )

    return table


def main() -> None:
    args = parse_args()
    started = time.perf_counter()

    aggregate = load_aggregate(args.aggregate)

    present = set(aggregate["scheme"].astype(str))
    missing_schemes = [scheme for scheme, _ in SCHEMES if scheme not in present]
    if missing_schemes:
        raise ValueError(f"Missing schemes: {missing_schemes}")

    # Infer the common evaluated grid.
    p_md_values = np.array(
        sorted(aggregate["p_md"].astype(float).unique()),
        dtype=float,
    )
    p_fa_values = np.array(
        sorted(aggregate["p_fa"].astype(float).unique()),
        dtype=float,
    )

    expected_rows = len(p_md_values) * len(p_fa_values)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.25, 2.65),
        sharex=True,
        sharey=True,
    )

    image = None

    for ax, (scheme, panel_title) in zip(axes, SCHEMES):
        scheme_df = aggregate.loc[
            aggregate["scheme"].astype(str) == scheme
        ].copy()

        pivot = (
            scheme_df.pivot(
                index="p_md",
                columns="p_fa",
                values="auprc_mean",
            )
            .reindex(index=p_md_values, columns=p_fa_values)
        )

        if pivot.size != expected_rows or pivot.isna().any().any():
            raise ValueError(
                f"{scheme}: incomplete P_MD/P_FA grid in aggregate table."
            )

        z = pivot.to_numpy(dtype=float)

        # Use actual P_FA and P_MD coordinates so the nonuniform P_FA grid is
        # displayed with its true spacing.
        X, Y = np.meshgrid(p_fa_values, p_md_values)

        image = ax.pcolormesh(
            X,
            Y,
            z,
            shading="nearest",
            vmin=args.vmin,
            vmax=args.vmax,
        )

        # Only draw contour levels that actually intersect this panel.
        valid_levels = [
            level
            for level in CONTOUR_LEVELS
            if np.nanmin(z) <= level <= np.nanmax(z)
        ]
        if valid_levels:
            contours = ax.contour(
                X,
                Y,
                z,
                levels=valid_levels,
                colors="black",
                linewidths=0.9,
            )
            ax.clabel(
                contours,
                inline=True,
                fontsize=6.5,
                fmt=lambda value: f"{value:.2f}",
                colors="black",
            )

        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(r"$P_{\mathrm{FA}}$")
        ax.set_xticks(p_fa_values)
        ax.tick_params(axis="x", labelrotation=90, labelsize=7)
        xlabels = ax.get_xticklabels()
        for label in xlabels:
            label.set_horizontalalignment("center")
            label.set_verticalalignment("top")

        # P_FA=0 and 0.01 are genuinely very close on the numeric axis.
        # Nudge only the "0" label a few points to the left for readability,
        # without changing the actual tick/data location.
        if xlabels:
            xlabels[0].set_transform(
                xlabels[0].get_transform()
                + ScaledTranslation(-4 / 72, 0, fig.dpi_scale_trans)
            )
        ax.tick_params(axis="y", labelsize=7)

    axes[0].set_ylabel(r"$P_{\mathrm{MD}}$")
    axes[0].set_yticks(p_md_values)

    if image is None:
        raise RuntimeError("No panels were drawn.")

    fig.subplots_adjust(
        left=0.075,
        right=0.885,
        bottom=0.25,
        top=0.90,
        wspace=0.10,
    )

    # Dedicated colorbar axis keeps the scale clear of panel (c).
    cax = fig.add_axes([0.91, 0.25, 0.015, 0.65])
    cbar = fig.colorbar(image, cax=cax)
    cbar.set_label("AUPRC")
    cbar.ax.tick_params(labelsize=7)

    args.output_stem.parent.mkdir(parents=True, exist_ok=True)

    for fmt in args.formats:
        output_path = args.output_stem.with_suffix(f".{fmt}")
        kwargs = {"bbox_inches": "tight"}
        if fmt == "png":
            kwargs["dpi"] = args.dpi
        fig.savefig(output_path, **kwargs)
        print(f"Saved: {output_path}")

    plt.close(fig)

    elapsed = time.perf_counter() - started
    print(f"Total runtime: {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
