#!/usr/bin/env python3
"""Burrows's Delta with colour-coded hierarchical clustering dendrogram.

Updated version: no NLTK data dependency, deterministic colour mapping,
command-line options, and dependency checks.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    from stylometry_utils import ensure_requirements
except ModuleNotFoundError as exc:
    if exc.name == "stylometry_utils":
        raise SystemExit(
            "Could not find stylometry_utils.py. Put stylometry_utils.py in the same folder "
            "as this script, then run the command again."
        ) from None
    raise

REQUIRED_IMPORTS = {
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "scipy": "scipy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Burrows's Delta and plot a dendrogram.")
    parser.add_argument("--corpus", default="corpus", help="Folder containing .txt files. Default: corpus")
    parser.add_argument("--mfw", type=int, default=100, help="Number of most frequent words. Default: 100")
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Use raw counts instead of relative frequencies per 1,000 tokens.",
    )
    parser.add_argument("--matrix-out", default="burrows_delta_matrix.csv", help="CSV output path for the distance matrix.")
    parser.add_argument("--plot-out", default="dendrogram_visualisation_coloured.png", help="PNG output path for the dendrogram.")
    parser.add_argument(
        "--linkage-method",
        default="average",
        choices=["single", "complete", "average", "weighted", "centroid", "median", "ward"],
        help="Hierarchical clustering linkage method. Default: average",
    )
    parser.add_argument("--cmap", default="tab10", help="Matplotlib colour map name. Default: tab10")
    parser.add_argument("--legend", action="store_true", help="Add a group colour legend to the dendrogram.")
    parser.add_argument("--no-show", action="store_true", help="Save the plot but do not open a plot window.")
    parser.add_argument("--no-auto-install", action="store_true", help="Check dependencies but do not install missing packages.")
    return parser.parse_args()


def plot_coloured_dendrogram(
    delta_matrix,
    groups,
    output_path: str | Path,
    linkage_method: str,
    cmap_name: str,
    show: bool,
    legend: bool,
) -> Path:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform

    from stylometry_utils import ensure_output_parent, make_group_colours

    if delta_matrix.shape[0] < 2:
        raise ValueError("At least two texts are required for a dendrogram.")

    condensed = squareform(delta_matrix.to_numpy(dtype=float), checks=False)
    linkage_matrix = linkage(condensed, method=linkage_method)
    colours = make_group_colours(groups, cmap_name=cmap_name)

    fig, ax = plt.subplots(figsize=(12, 10))
    dendrogram(
        linkage_matrix,
        labels=list(delta_matrix.columns),
        leaf_rotation=90,
        leaf_font_size=10,
        color_threshold=0,
        above_threshold_color="black",
        ax=ax,
    )

    for label in ax.get_xmajorticklabels():
        group = label.get_text().split("_", 1)[0]
        if group in colours:
            label.set_color(colours[group])

    ax.set_title("Burrows's Delta")
    ax.set_xlabel("Texts")
    ax.set_ylabel("Distance")

    if legend:
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=colour, label=group, markersize=8)
            for group, colour in colours.items()
        ]
        ax.legend(handles=handles, title="Groups", loc="best")

    fig.tight_layout()
    path = ensure_output_parent(output_path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Dendrogram saved as '{path}'.")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    ensure_requirements(REQUIRED_IMPORTS, auto_install=not args.no_auto_install)

    from stylometry_utils import (
        calculate_z_scores,
        compute_burrows_delta,
        compute_frequency_matrix,
        extract_groups,
        load_texts_from_folder,
        save_dataframe_csv,
        tokenise_corpus,
    )

    texts = load_texts_from_folder(args.corpus)
    if len(texts) < 2:
        raise ValueError("At least two .txt files are required for a dendrogram.")

    tokenised_texts = tokenise_corpus(texts)
    frequency_matrix = compute_frequency_matrix(tokenised_texts, mfw=args.mfw, relative=not args.raw_counts)
    z_scores = calculate_z_scores(frequency_matrix)
    delta_matrix = compute_burrows_delta(z_scores)

    matrix_path = save_dataframe_csv(delta_matrix, args.matrix_out)
    print(f"Delta matrix saved as '{matrix_path}'.")

    groups = extract_groups(delta_matrix.columns)
    plot_coloured_dendrogram(
        delta_matrix,
        groups,
        output_path=args.plot_out,
        linkage_method=args.linkage_method,
        cmap_name=args.cmap,
        show=not args.no_show,
        legend=args.legend,
    )


if __name__ == "__main__":
    main()
