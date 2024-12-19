# Stylometry

This repository contains Python scripts for performing stylometric analysis using various techniques:

1. **`burrows-delta-mds.py`**: Computes Burrows's Delta and visualises the results using Multidimensional Scaling (MDS).
2. **`burrows-delta-dendrogram.py`**: Computes Burrows's Delta and visualises the results using a dendrogram based on hierarchical clustering.
3. **`roberta-embeddings.py`**: Analyses stylistic similarities among literary texts using the RoBERTa model for embedding generation, dimensionality reduction via PCA, and visualisation through scatter plots.

## Scripts Overview

### 1. `burrows-delta-mds.py`

**Purpose**: Computes Burrows's Delta and visualises the results using Multidimensional Scaling (MDS).

**Output**: A scatter plot where points represent texts, and their proximity reflects stylistic similarity.

#### Features:
- Most Frequent Words (MFW): Set to 100 by default.
- Saves the Burrows's Delta matrix as `burrows_delta_matrix.csv`.
- Saves the MDS visualisation as `mds_visualisation.png`.

#### Usage:
Run the script in your terminal:
```bash
python3 burrows-delta-mds.py
```

#### Dependencies:
- `nltk`
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

### 2. `burrows-delta-dendrogram.py`

**Purpose**: Computes Burrows's Delta and visualises the results using a dendrogram based on hierarchical clustering.

**Output**: A dendrogram where labels are colour-coded by group, extracted from the text before the first `_` in each filename.

#### Features:
- Linkage Method: Uses average linkage (default in stylometry for balanced clustering).
- Colour-Coded Labels: Groups are derived from the filenames (e.g., `group_filename.txt`).
- Saves the Burrows's Delta matrix as `burrows_delta_matrix.csv`.
- Saves the dendrogram visualisation as `dendrogram_visualisation_coloured.png`.

#### Usage:
Run the script in your terminal:
```bash
python3 burrows-delta-dendrogram.py
```

#### Dependencies:
- `nltk`
- `pandas`
- `numpy`
- `matplotlib`
- `scipy`

### 3. `roberta-embeddings.py`

**Purpose**: Analyses stylistic similarities among literary texts using the RoBERTa model for embedding generation, dimensionality reduction via PCA, and visualisation through scatter plots.

## Input Requirements

### Corpus

Both `burrows-delta` scripts expect a folder named `corpus` containing `.txt` files. Each file should represent a single text. 

For `burrows-delta-dendrogram.py`, filenames should follow the format:
```
<group>_rest_of_filename.txt
```

For `roberta-embeddings.py`, the folder should be named `lit-families` on your Desktop. Filenames should follow the structure:
```
surname_firstinitial_title.txt
```

### Example Folder Structure
```
corpus/
├── group1_text1.txt
├── group1_text2.txt
├── group2_text1.txt
├── group2_text2.txt
```
```
lit-families/
├── Joyce_J_Ulysses.txt
├── Woolf_V_ToTheLighthouse.txt
```

## Outputs

### Common Outputs

- Delta Matrix: Saved as `burrows_delta_matrix.csv`. A symmetric matrix of stylistic distances between texts.

### Script-Specific Outputs

1. **MDS Script (`burrows-delta-mds.py`)**:
   - Scatter Plot: Saved as `mds_visualisation.png`.

2. **Dendrogram Script (`burrows-delta-dendrogram.py`)**:
   - Dendrogram Plot: Saved as `dendrogram_visualisation_coloured.png`.

3. **RoBERTa Script (`roberta-embeddings.py`)**:
   - Scatter plots are displayed in separate windows but are not saved automatically.

## Customisation

### Adjusting Most Frequent Words (MFW)

Both `burrows-delta` scripts use the top 100 Most Frequent Words (MFW) by default. To change this, modify the `mfw` parameter in the `compute_frequencies` function:
```python
frequency_matrix = compute_frequencies(preprocessed_texts, mfw=200)  # Example: Use 200 MFW
```

### Changing Linkage Method in Dendrogram

The dendrogram script uses average linkage by default. To change the method, update the `linkage` function:
```python
linkage_matrix = linkage(condensed_matrix, method='complete')  # Use complete linkage
```
Available methods: `'single'`, `'complete'`, `'average'`, `'ward'`.

### RoBERTa Visualisation Saving

To save the visualisations from `roberta-embeddings.py`, modify the `plt.show()` lines to include saving functionality, such as:
```python
plt.savefig('plot1.png')
```

## Dependencies

Install required Python packages using pip:
```bash
pip install nltk pandas numpy matplotlib scipy scikit-learn sentence-transformers
```

## How to Run Scripts

1. Place the respective script in any directory on your machine.
2. Ensure the required folder (`corpus` or `lit-families`) exists and contains `.txt` files.
3. Run the script using:
```bash
python <script_name>.py
```
4. View the generated plots or outputs.

## Notes

- **Model Choice**: `roberta-embeddings.py` uses `all-roberta-large-v1`, a RoBERTa-based transformer model trained for sentence embeddings. 
- **Customisation**: You can replace the model with any `SentenceTransformer` model. Update the following line:
```python
model = SentenceTransformer('all-roberta-large-v1')
```
- **Text Preprocessing**: Minimal preprocessing is applied. Add additional steps if needed for your corpus.

## Troubleshooting

1. **No Plots Displayed**:
   - Ensure `matplotlib` is installed.
   - Check for errors during PCA or embedding generation.

2. **Incorrect Labels**:
   - Verify the file naming structure. Non-conforming files are labelled as "Unknown."

