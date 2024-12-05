# Stylometry

This repository contains two Python scripts for performing stylometric analysis using Burrows's Delta. The scripts support two different methods for visualising the results:

1. MDS (Multidimensional Scaling) for 2D scatter plots.
2. Hierarchical Clustering for dendrogram visualisation, with colour-coded labels.

## Scripts Overview

### 1. `burrows_delta_mds.py`
- Purpose Computes Burrows's Delta and visualises the results using Multidimensional Scaling (MDS).
- Output: A scatter plot where points represent texts, and their proximity reflects stylistic similarity.

#### Features:
- Most Frequent Words (MFW): Set to 100 by default.
- Saves the Burrows's Delta matrix as `burrows_delta_matrix.csv`.
- Saves the MDS visualisation as `mds_visualisation.png`.

#### Usage:
Run the script in your terminal:
```bash
python3 burrows_delta_mds.py
```

#### Dependencies:
- `nltk`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

### 2. `burrows_delta_dendrogram.py`
- Purpose: Computes Burrows's Delta and visualises the results using a dendrogram based on hierarchical clustering.
- Output: A dendrogram where labels are colour-coded by group, extracted from the text before the first `_` in each filename.

#### Features:
- Linkage Method: Uses average linkage (default in stylometry for balanced clustering).
- Colour-Coded Labels: Groups are derived from the filenames (e.g., `group_filename.txt`).
- Saves the Burrows's Delta matrix as `burrows_delta_matrix.csv`.
- Saves the dendrogram visualisation as `dendrogram_visualisation_coloured.png`.

#### Usage:
Run the script in your terminal:
```bash
python3 burrows_delta_dendrogram.py
```

#### Dependencies:
- `nltk`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`

## Input Requirements

### Corpus
- Both scripts expect a folder named `corpus`, containing `.txt` files.
- Each file should represent a single text.
- For `burrows_delta_dendrogram.py`, filenames should follow the format:  
  `<group>_rest_of_filename.txt`.

### Example Folder Structure
```
corpus/
├── group1_text1.txt
├── group1_text2.txt
├── group2_text1.txt
├── group2_text2.txt
```

## Outputs

### Common Outputs
- Delta Matrix: Saved as `burrows_delta_matrix.csv`.
  - A symmetric matrix of stylistic distances between texts.

### Script-Specific Outputs
1. MDS Script (`burrows_delta_mds.py`):
   - Scatter Plot: Saved as `mds_visualisation.png`.

2. Dendrogram Script (`burrows_delta_dendrogram.py`):
   - Dendrogram Plot: Saved as `dendrogram_visualisation_coloured.png`.

---

## Customisation

### Adjusting Most Frequent Words (MFW)
Both scripts use the top 100 Most Frequent Words (MFW) by default. To change this, modify the `mfw` parameter in the `compute_frequencies` function:
```python
frequency_matrix = compute_frequencies(preprocessed_texts, mfw=200)  # Example: Use 200 MFW
```

### Changing Linkage Method in Dendrogram
The dendrogram script uses average linkage by default. To change the method, update the `linkage` function:
```python
linkage_matrix = linkage(condensed_matrix, method='complete')  # Use complete linkage
```
Available methods: `'single'`, `'complete'`, `'average'`, `'ward'`.

## Dependencies

Install required Python packages using pip:
```bash
pip install nltk pandas numpy matplotlib scipy scikit-learn
```
