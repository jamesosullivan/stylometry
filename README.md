# Stylometry

This repository contains updated Python scripts for stylometric analysis using Burrows's Delta and transformer embeddings.

## Files

- `burrows-delta-mds.py`: Computes Burrows's Delta and visualises the results using Multidimensional Scaling (MDS).
- `burrows-delta-dendrogram.py`: Computes Burrows's Delta and visualises the results using a hierarchical clustering dendrogram.
- `roberta-embeddings.py`: Computes document-level SentenceTransformer embeddings, cosine distances, and PCA visualisations.
- `RoBERTa-embeddings.py`: Compatibility wrapper for the old mixed-case filename.
- `stylometry_utils.py`: Shared helper functions used by the scripts.
- `requirements.txt`: Dependency list used by the scripts for checking/auto-installation and by users for manual installation.

## How to cite

Please cite this software as created by:

> O'Sullivan, James, University College Cork. URL.

Suggested BibTeX:

```bibtex
@software{osullivan_stylometry,
  author = {O'Sullivan, James},
  title = {Stylometry: Python scripts for Burrows's Delta and transformer embeddings},
  institution = {University College Cork},
  type = {Software}
}
```

If you archive or release this repository through Zenodo, GitHub Releases, or another repository, add the version number, release year, and DOI to the citation.

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

## License

This project is licensed under the MIT License.

Copyright (c) 2024 James O'Sullivan.

See the `LICENSE` file for the full license text. In summary, the MIT License permits use, copying, modification, merging, publication, distribution, sublicensing, and sale of copies of the software, provided that the copyright notice and permission notice are included in copies or substantial portions of the software. The software is provided without warranty.

## Troubleshooting


### `ModuleNotFoundError: No module named 'stylometry_utils'`

Keep `stylometry_utils.py` in the same folder as the scripts. For example:

```text
commonwealth/
├── burrows-delta-mds.py
├── burrows-delta-dendrogram.py
├── roberta-embeddings.py
├── stylometry_utils.py
├── requirements.txt
└── corpus/
```

Do not copy only `burrows-delta-mds.py` into a new folder unless you also copy `stylometry_utils.py` and `requirements.txt`.

### Dependencies install but are still reported as missing

The dependency checker now distinguishes between import names, such as `sklearn`, and package names, such as `scikit-learn>=1.4`. If pip installs packages into the macOS user-site directory during a running script, the script will restart once so the new packages are visible.

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
