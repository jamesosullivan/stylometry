#!/usr/bin/env python3
"""Document-level transformer embeddings for stylometric exploration.

Updated version: dependency checks, command-line options, chunked document
embedding, saved distance/PCA outputs, and robust plotting.
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
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "sentence_transformers": "sentence-transformers",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create document embeddings and PCA plots with SentenceTransformers.")
    parser.add_argument(
        "--corpus",
        default="~/Desktop/lit-families",
        help="Folder containing .txt files. Default: ~/Desktop/lit-families",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-roberta-large-v1",
        help="SentenceTransformer model name or local path. Default: sentence-transformers/all-roberta-large-v1",
    )
    parser.add_argument("--chunk-words", type=int, default=250, help="Approximate words per document chunk. Default: 250")
    parser.add_argument("--batch-size", type=int, default=16, help="Embedding batch size. Default: 16")
    parser.add_argument("--device", default=None, help="Optional device, e.g. cpu, cuda, mps. Default: auto")
    parser.add_argument("--output-prefix", default="roberta", help="Prefix for CSV and PNG outputs. Default: roberta")
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Do not L2-normalize chunk and document embeddings before cosine distances.",
    )
    parser.add_argument("--cmap", default="tab10", help="Matplotlib colour map name. Default: tab10")
    parser.add_argument("--no-show", action="store_true", help="Save plots but do not open plot windows.")
    parser.add_argument("--no-auto-install", action="store_true", help="Check dependencies but do not install missing packages.")
    return parser.parse_args()


def label_from_filename(filename: str) -> str:
    """Return surname_initial from surname_firstinitial_title.txt, or Unknown."""
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 2:
        return "Unknown"
    return f"{parts[0]}_{parts[1]}"


def normalise_rows(matrix):
    import numpy as np

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def encode_documents(texts_by_name, model_name: str, chunk_words: int, batch_size: int, device: str | None, normalize: bool):
    import numpy as np
    from sentence_transformers import SentenceTransformer

    from stylometry_utils import chunk_text_by_words

    model = SentenceTransformer(model_name, device=device) if device else SentenceTransformer(model_name)

    document_names: list[str] = []
    chunks: list[str] = []
    chunk_to_document: list[int] = []

    for doc_index, (filename, text) in enumerate(texts_by_name.items()):
        doc_chunks = chunk_text_by_words(text, chunk_words=chunk_words)
        if not doc_chunks:
            print(f"Skipping empty text: {filename}")
            continue
        document_names.append(filename)
        for chunk in doc_chunks:
            chunks.append(chunk)
            chunk_to_document.append(len(document_names) - 1)

    if len(document_names) < 2:
        raise ValueError("At least two non-empty .txt files are required for embedding analysis.")

    encode_kwargs = {
        "convert_to_numpy": True,
        "show_progress_bar": True,
        "batch_size": batch_size,
    }
    if normalize:
        encode_kwargs["normalize_embeddings"] = True

    try:
        chunk_embeddings = model.encode(chunks, **encode_kwargs)
    except TypeError:
        # Older sentence-transformers versions may not support normalize_embeddings.
        encode_kwargs.pop("normalize_embeddings", None)
        chunk_embeddings = model.encode(chunks, **encode_kwargs)
        if normalize:
            chunk_embeddings = normalise_rows(chunk_embeddings)

    doc_embeddings = np.zeros((len(document_names), chunk_embeddings.shape[1]), dtype=float)
    doc_counts = np.zeros(len(document_names), dtype=int)
    for embedding, doc_index in zip(chunk_embeddings, chunk_to_document):
        doc_embeddings[doc_index] += embedding
        doc_counts[doc_index] += 1
    doc_embeddings = doc_embeddings / doc_counts[:, None]
    if normalize:
        doc_embeddings = normalise_rows(doc_embeddings)

    return document_names, doc_embeddings


def save_distance_matrix(document_names, embeddings, output_prefix: str) -> Path:
    import pandas as pd
    from sklearn.metrics.pairwise import cosine_distances

    from stylometry_utils import ensure_output_parent

    distances = cosine_distances(embeddings)
    path = ensure_output_parent(f"{output_prefix}_cosine_distance_matrix.csv")
    pd.DataFrame(distances, index=document_names, columns=document_names).to_csv(path)
    print(f"Cosine distance matrix saved as '{path}'.")
    return path


def reduce_with_pca(document_names, embeddings, output_prefix: str):
    import pandas as pd
    from sklearn.decomposition import PCA

    from stylometry_utils import ensure_output_parent

    if embeddings.shape[0] < 2:
        raise ValueError("At least two documents are required for PCA visualisation.")

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(embeddings)
    labels = [label_from_filename(name) for name in document_names]
    dataframe = pd.DataFrame(
        {
            "filename": document_names,
            "label": labels,
            "PC1": coords[:, 0],
            "PC2": coords[:, 1],
            "explained_variance_PC1": pca.explained_variance_ratio_[0],
            "explained_variance_PC2": pca.explained_variance_ratio_[1],
        }
    )
    path = ensure_output_parent(f"{output_prefix}_pca_coordinates.csv")
    dataframe.to_csv(path, index=False)
    print(f"PCA coordinates saved as '{path}'.")
    return dataframe


def plot_colour_pca(pca_df, output_prefix: str, cmap_name: str, show: bool) -> Path:
    import matplotlib.pyplot as plt

    from stylometry_utils import ensure_output_parent, make_group_colours

    labels = list(pca_df["label"])
    colours = make_group_colours(labels, cmap_name=cmap_name)
    fig, ax = plt.subplots(figsize=(10, 7))

    for label in sorted(set(labels)):
        subset = pca_df[pca_df["label"] == label]
        ax.scatter(subset["PC1"], subset["PC2"], label=label, color=colours[label], s=80)

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA of RoBERTa Embeddings")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    path = ensure_output_parent(f"{output_prefix}_pca_by_author.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Colour PCA plot saved as '{path}'.")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def plot_labelled_pca(pca_df, output_prefix: str, show: bool) -> Path:
    import matplotlib.pyplot as plt

    from stylometry_utils import ensure_output_parent

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(pca_df["PC1"], pca_df["PC2"], edgecolors="black", facecolors="none", marker="o", s=80)

    for _, row in pca_df.iterrows():
        ax.annotate(row["label"], (row["PC1"], row["PC2"]), fontsize=8, ha="right", va="center")

    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("2D PCA of RoBERTa Embeddings")
    fig.tight_layout()

    path = ensure_output_parent(f"{output_prefix}_pca_labelled.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Labelled PCA plot saved as '{path}'.")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return path


def main() -> None:
    args = parse_args()
    ensure_requirements(REQUIRED_IMPORTS, auto_install=not args.no_auto_install)

    from stylometry_utils import load_texts_from_folder

    texts = load_texts_from_folder(args.corpus)
    document_names, embeddings = encode_documents(
        texts,
        model_name=args.model,
        chunk_words=args.chunk_words,
        batch_size=args.batch_size,
        device=args.device,
        normalize=not args.no_normalize,
    )
    save_distance_matrix(document_names, embeddings, output_prefix=args.output_prefix)
    pca_df = reduce_with_pca(document_names, embeddings, output_prefix=args.output_prefix)
    plot_colour_pca(pca_df, output_prefix=args.output_prefix, cmap_name=args.cmap, show=not args.no_show)
    plot_labelled_pca(pca_df, output_prefix=args.output_prefix, show=not args.no_show)


if __name__ == "__main__":
    main()
