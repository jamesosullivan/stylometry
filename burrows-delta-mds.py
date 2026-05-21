#!/usr/bin/env python3
"""Burrows's Delta with MDS visualisation.

Updated version: no NLTK data dependency, no deprecated Matplotlib get_cmap API,
explicit scikit-learn MDS defaults, command-line options, and dependency checks.
"""

from __future__ import annotations

import argparse
import inspect
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
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Burrows's Delta and plot an MDS map.")
    parser.add_argument("--corpus", default="corpus", help="Folder containing .txt files. Default: corpus")
    parser.add_argument("--mfw", type=int, default=100, help="Number of most frequent words. Default: 100")
    parser.add_argument(
        "--raw-counts",
        action="store_true",
        help="Use raw counts instead of relative frequencies per 1,000 tokens.",
    )
    parser.add_argument("--matrix-out", default="burrows_delta_matrix.csv", help="CSV output path for the distance matrix.")
    parser.add_argument("--plot-out", default="mds_visualisation_coloured.png", help="PNG output path for the MDS plot.")
    parser.add_argument("--cmap", default="tab10", help="Matplotlib colour map name. Default: tab10")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for MDS. Default: 42")
    parser.add_argument("--n-init", type=int, default=4, help="MDS initialisations. Explicit to avoid future default changes. Default: 4")
    parser.add_argument("--no-show", action="store_true", help="Save the plot but do not open a plot window.")
    parser.add_argument("--no-auto-install", action="store_true", help="Check dependencies but do not install missing packages.")
    return parser.parse_args()


def build_mds_model(random_state: int, n_init: int):
    """Create an MDS model while staying compatible across scikit-learn versions."""
    from sklearn.manifold import MDS

    signature = inspect.signature(MDS)
    kwargs = {
        "n_components": 2,
        "random_state": random_state,
        "n_init": n_init,
    }

    # scikit-learn 1.8 renamed `dissimilarity="precomputed"` to
    # `metric="precomputed"`. Older versions used `metric` as a bool for
    # metric/non-metric MDS, so inspect the default before choosing the API.
    metric_param = signature.parameters.get("metric")
    if metric_param is not None and isinstance(metric_param.default, str):
        kwargs["metric"] = "precomputed"
    else:
        kwargs["dissimilarity"] = "precomputed"

    if "normalized_stress" in signature.parameters:
        kwargs["normalized_stress"] = "auto"
    if "init" in signature.parameters:
        # Preserve the historical behaviour rather than relying on a changing default.
        kwargs["init"] = "random"
    return MDS(**kwargs)


def plot_mds(delta_matrix, groups, output_path: str | Path, show: bool, cmap_name: str, random_state: int, n_init: int) -> Path:
    import matplotlib.pyplot as plt

    from stylometry_utils import ensure_output_parent, make_group_colours

    model = build_mds_model(random_state=random_state, n_init=n_init)
    coords = model.fit_transform(delta_matrix.to_numpy(dtype=float))

    colours = make_group_colours(groups, cmap_name=cmap_name)
    fig, ax = plt.subplots(figsize=(10, 8))
    labelled_groups: set[str] = set()

    for i, text_name in enumerate(delta_matrix.columns):
        group = groups[i]
        label = group if group not in labelled_groups else None
        ax.scatter(coords[i, 0], coords[i, 1], color=colours[group], s=100, label=label)
        ax.annotate(str(text_name), (coords[i, 0], coords[i, 1]), fontsize=9, ha="right", va="bottom")
        labelled_groups.add(group)

    ax.set_title("MDS Visualisation of Burrows's Delta")
    ax.set_xlabel("MDS Dimension 1")
    ax.set_ylabel("MDS Dimension 2")
    ax.grid(True, alpha=0.3)
    ax.legend(title="Groups", loc="best")
    fig.tight_layout()

    path = ensure_output_parent(output_path)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Plot saved as '{path}'.")
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
        raise ValueError("At least two .txt files are required for MDS.")

    tokenised_texts = tokenise_corpus(texts)
    frequency_matrix = compute_frequency_matrix(tokenised_texts, mfw=args.mfw, relative=not args.raw_counts)
    z_scores = calculate_z_scores(frequency_matrix)
    delta_matrix = compute_burrows_delta(z_scores)

    matrix_path = save_dataframe_csv(delta_matrix, args.matrix_out)
    print(f"Delta matrix saved as '{matrix_path}'.")

    groups = extract_groups(delta_matrix.columns)
    plot_mds(
        delta_matrix,
        groups,
        output_path=args.plot_out,
        show=not args.no_show,
        cmap_name=args.cmap,
        random_state=args.random_state,
        n_init=args.n_init,
    )


if __name__ == "__main__":
    main()
