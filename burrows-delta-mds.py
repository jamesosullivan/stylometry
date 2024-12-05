import os
import nltk
from collections import Counter
import pandas as pd
import numpy as np

# Append NLTK data path
nltk.data.path.append('/Users/jamesosullivan/nltk_data')

# Load the punkt resource
from nltk.tokenize import word_tokenize

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

# 3. Compute word frequencies
def compute_frequencies(tokenised_texts):
    all_tokens = []
    for tokens in tokenised_texts.values():
        all_tokens.extend(tokens)
    most_common_words = [word for word, _ in Counter(all_tokens).most_common(20)]  # Most frequent words (MFW)
    
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
    return delta_matrix

# Main script
if __name__ == "__main__":
    # Ensure the corpus folder exists
    if not os.path.exists(CORPUS_FOLDER):
        raise FileNotFoundError(f"The folder '{CORPUS_FOLDER}' does not exist.")

    # Load, preprocess, and analyse texts
    texts = load_texts_from_folder(CORPUS_FOLDER)
    preprocessed_texts = {key: preprocess(value) for key, value in texts.items()}
    frequency_matrix = compute_frequencies(preprocessed_texts)
    z_scores = calculate_z_scores(frequency_matrix)
    delta_matrix = compute_delta(z_scores)

    # Display results and save Delta Matrix to a CSV file
    print("\nBurrows's Delta Matrix:\n", delta_matrix)
    delta_matrix.to_csv("burrows_delta_matrix.csv")
    print("\nDelta matrix saved as 'burrows_delta_matrix.csv'.")
