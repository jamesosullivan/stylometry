import os
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Path to your folder containing the text files
folder_path = os.path.expanduser("~/Desktop/lit-families")

# Load the RoBERTa-based model
model = SentenceTransformer('all-roberta-large-v1')

texts = []
labels = []
file_paths = []

# Iterate over all text files in the directory
for fname in os.listdir(folder_path):
    if fname.endswith(".txt"):
        fpath = os.path.join(folder_path, fname)
        with open(fpath, 'r', encoding='utf-8') as f:
            text = f.read().strip()

        # Extract the base name without the extension
        base_name = os.path.splitext(fname)[0]
        # Expecting something like: "surname_firstinitial_title"
        parts = base_name.split("_")

        # If the filename structure is correct, parts[0] = surname, parts[1] = firstinitial(s)
        if len(parts) < 2:
            # If the file doesn't follow the expected pattern, label as Unknown
            author_label = "Unknown"
        else:
            # Combine surname and first initial(s) into a single label
            author_label = f"{parts[0]}_{parts[1]}"

        texts.append(text)
        labels.append(author_label)
        file_paths.append(fname)

# Encode the texts into embeddings
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

# Compute stylistic distances
dist_matrix = cosine_distances(embeddings)

# Reduce dimensions for visualisation
pca = PCA(n_components=2, random_state=42)
reduced = pca.fit_transform(embeddings)

#####################################################
# Plot 1: Colour-coded by surname_initial
#####################################################


plt.figure(figsize=(10,7))
unique_labels = set(labels)
for label in unique_labels:
    idxs = [i for i, l in enumerate(labels) if l == label]
    plt.scatter(reduced[idxs, 0], reduced[idxs, 1], label=label)

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("2D PCA of RoBERTa Embeddings")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

#####################################################
# Plot 2: Black and white shapes with text labels
#####################################################


plt.figure(figsize=(10,7))
# Plot all points as black hollow circles
plt.scatter(reduced[:, 0], reduced[:, 1], edgecolors='black', facecolors='none', marker='o')

# Add text labels next to each point
for i, label in enumerate(labels):
    plt.text(reduced[i, 0], reduced[i, 1], label, fontsize=8, ha='right', va='center')

plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("2D PCA of RoBERTa embeddings")
plt.tight_layout()
plt.show()
