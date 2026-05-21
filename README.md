# Stylometry

This repository contains updated Python scripts for stylometric analysis using Burrows's Delta and transformer embeddings.

## Files

- `burrows-delta-mds.py`: Computes Burrows's Delta and visualises the results using Multidimensional Scaling (MDS).
- `burrows-delta-dendrogram.py`: Computes Burrows's Delta and visualises the results using a hierarchical clustering dendrogram.
- `roberta-embeddings.py`: Computes document-level SentenceTransformer embeddings, cosine distances, and PCA visualisations.
- `RoBERTa-embeddings.py`: Compatibility wrapper for the old mixed-case filename.
- `stylometry_utils.py`: Shared helper functions used by the scripts.
- `requirements.txt`: Optional dependency list for manual installation.

## Recommended setup

Use a virtual environment so automatic installation does not modify your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Manual installation is optional because the scripts can install missing dependencies themselves. To disable auto-installation and only check dependencies, add `--no-auto-install`.

## Corpus layout

The Burrows's Delta scripts expect a folder of `.txt` files. By default, the folder is called `corpus`:

```text
corpus/
├── group1_text1.txt
├── group1_text2.txt
├── group2_text1.txt
├── group2_text2.txt
```

Group labels are taken from the filename text before the first underscore. For example, `Joyce_Ulysses.txt` is assigned to group `Joyce`.

The RoBERTa script defaults to `~/Desktop/lit-families`, but you can use any folder with `--corpus`:

```text
lit-families/
├── Joyce_J_Ulysses.txt
├── Woolf_V_ToTheLighthouse.txt
```

For the RoBERTa script, plot labels are taken from the first two underscore-separated filename parts, for example `Joyce_J`.

## Usage

### Burrows's Delta MDS

```bash
python burrows-delta-mds.py --corpus corpus --mfw 100
```

Outputs:

- `burrows_delta_matrix.csv`
- `mds_visualisation_coloured.png`

Useful options:

```bash
python burrows-delta-mds.py --corpus corpus --mfw 200 --no-show
python burrows-delta-mds.py --corpus corpus --raw-counts
python burrows-delta-mds.py --corpus corpus --matrix-out outputs/delta.csv --plot-out outputs/mds.png
```

### Burrows's Delta dendrogram

```bash
python burrows-delta-dendrogram.py --corpus corpus --mfw 100
```

Outputs:

- `burrows_delta_matrix.csv`
- `dendrogram_visualisation_coloured.png`

Useful options:

```bash
python burrows-delta-dendrogram.py --corpus corpus --mfw 200 --legend --no-show
python burrows-delta-dendrogram.py --corpus corpus --linkage-method complete
python burrows-delta-dendrogram.py --corpus corpus --matrix-out outputs/delta.csv --plot-out outputs/dendrogram.png
```

### RoBERTa / SentenceTransformer embeddings

```bash
python roberta-embeddings.py --corpus ~/Desktop/lit-families
```

Outputs:

- `roberta_cosine_distance_matrix.csv`
- `roberta_pca_coordinates.csv`
- `roberta_pca_by_author.png`
- `roberta_pca_labelled.png`

Useful options:

```bash
python roberta-embeddings.py --corpus corpus --model sentence-transformers/all-roberta-large-v1 --no-show
python roberta-embeddings.py --corpus corpus --chunk-words 300 --batch-size 8
python roberta-embeddings.py --corpus corpus --device mps
python roberta-embeddings.py --corpus corpus --output-prefix outputs/roberta_run1
```

The first run of the RoBERTa script may download the selected model from Hugging Face. The default model is `sentence-transformers/all-roberta-large-v1`.

## Notes on interpretation

Burrows's Delta is based on standardised most-frequent-word frequencies. The scripts now use relative frequency per 1,000 tokens by default to reduce text-length effects. If you need strict continuity with the older scripts, use `--raw-counts`.

The embedding script is exploratory. SentenceTransformer embeddings capture semantic and stylistic signals together, so PCA clusters should not be treated as pure style attribution without further validation.

## Troubleshooting

### A plot does not open

Use `--no-show` to save the plot without opening a GUI window:

```bash
python burrows-delta-mds.py --no-show
```

### A package will not install automatically

Install dependencies manually:

```bash
python -m pip install -r requirements.txt
```

### The RoBERTa model is slow or too large

Use a smaller SentenceTransformer model:

```bash
python roberta-embeddings.py --model sentence-transformers/all-MiniLM-L6-v2
```
