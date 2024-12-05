import os
import nltk
from collections import Counter
import pandas as pd
import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt
import matplotlib

# Force Matplotlib to use a compatible backend
matplotlib.use('TkAgg')

# Load the punkt resource
from nltk.tokenize import word_tokenize
from matplotlib.cm import get_cmap

# Path to the folder containing text files
CORPUS_FOLDER = "corpus"

# 1. Load texts from the folder
def load_texts_from_folder(folder_path):
    texts = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                texts[filename] = file.read()
    return texts

# 2. Preprocess texts
def preprocess(text):
    tokens = word_tokenize(text.lower())  # Tokenise and lowercase
    filtered_tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    return filtered_tokens

# 3. Compute word frequencies  # Most frequent words (MFW) set to 100 by default
def compute_frequencies(tokenised_texts, mfw=100):
    all_tokens = []
    for tokens in tokenised_texts.values():
        all_tokens.extend(tokens)
    most_common_words = [word for word, _ in Counter(all_tokens).most_common(mfw)] 
    
    frequencies = {}
    for name, tokens in tokenised_texts.items():
        word_counts = Counter(tokens)
        frequencies[name] = {word: word_counts[word] for word in most_common_words}
    return pd.DataFrame(frequencies).fillna(0)

# 4. Calculate z-scores
def calculate_z_scores(frequency_matrix):
    return frequency_matrix.apply(lambda col: (col - col.mean()) / col.std(), axis=1)

# 5. Compute Burrows's Delta
def compute_delta(z_matrix):
    delta_matrix = pd.DataFrame(index=z_matrix.columns, columns=z_matrix.columns)
    for text1 in z_matrix.columns:
        for text2 in z_matrix.columns:
            delta = np.mean(np.abs(z_matrix[text1] - z_matrix[text2]))
            delta_matrix.loc[text1, text2] = delta
    # Symmetrise the matrix
    delta_matrix = delta_matrix.fillna(0)  # Replace NaNs
    delta_matrix = (delta_matrix + delta_matrix.T) / 2  # Ensure symmetry
    np.fill_diagonal(delta_matrix.values, 0)  # Diagonal must be 0
    return delta_matrix

# 6. Extract Groups for Colour Coding
def extract_groups(filenames):
    """
    Extract groups from filenames based on the text before the first `_`.

    Args:
        filenames (list): List of filenames.

    Returns:
        list: Groups for each filename.
    """
    return [filename.split('_')[0] for filename in filenames]

# 7. Visualise Delta Matrix with Colour-Coded MDS
def plot_mds(delta_matrix, groups, save_as=None):
    """
    Visualise the Burrows's Delta matrix using Multidimensional Scaling (MDS)
    with colour-coded dots based on groups.

    Args:
        delta_matrix (pd.DataFrame): Pairwise distances between texts.
        groups (list): Groups for colour coding.
        save_as (str, optional): File path to save the plot. Defaults to None.
    """
    # Perform MDS
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
    mds_coords = mds.fit_transform(delta_matrix.values)

    # Map groups to colours
    unique_groups = list(set(groups))
    cmap = get_cmap("tab10")
    colours = {group: cmap(i / len(unique_groups)) for i, group in enumerate(unique_groups)}

    # Create scatter plot
    plt.figure(figsize=(10, 8))
    for i, text in enumerate(delta_matrix.columns):
        group = groups[i]
        plt.scatter(mds_coords[i, 0], mds_coords[i, 1], c=[colours[group]], s=100, label=group if group not in plt.gca().get_legend_handles_labels()[1] else None)
        plt.text(mds_coords[i, 0], mds_coords[i, 1], text, fontsize=9, ha='right', va='bottom')

    # Add titles, labels, and legend
    plt.title("MDS Visualisation of Burrows's Delta")
    plt.xlabel("MDS Dimension 1")
    plt.ylabel("MDS Dimension 2")
    plt.legend(title="Groups", loc="best")
    plt.grid(True)
    plt.tight_layout()

    # Save or show plot
    if save_as:
        plt.savefig(save_as)
        print(f"Plot saved as '{save_as}'.")
    plt.show()

# Main script
if __name__ == "__main__":
    # Ensure the corpus folder exists
    if not os.path.exists(CORPUS_FOLDER):
        raise FileNotFoundError(f"The folder '{CORPUS_FOLDER}' does not exist.")

    # Load, preprocess, and analyse texts
    texts = load_texts_from_folder(CORPUS_FOLDER)
    preprocessed_texts = {key: preprocess(value) for key, value in texts.items()}
    frequency_matrix = compute_frequencies(preprocessed_texts, mfw=100)  # MFW set to 100
    z_scores = calculate_z_scores(frequency_matrix)
    delta_matrix = compute_delta(z_scores)

    # Extract groups for colour coding
    groups = extract_groups(delta_matrix.columns)

    # Save Delta Matrix to a CSV file
    delta_matrix.to_csv("burrows_delta_matrix.csv")
    print("\nDelta matrix saved as 'burrows_delta_matrix.csv'.")
    
    # Plot MDS with colour coding
    plot_mds(delta_matrix, groups, save_as="mds_visualisation_coloured.png")
