from typing import List
import numpy as np
from sentence_transformers import  SentenceTransformer, util

# Specify the model
model = SentenceTransformer('all-MiniLM-L6-v2')

def are_intents_similar(phrases1: List[str], phrases2: List[str]) -> bool:
    """
    Calculate similarity between two sets of phrases using a SentenceTransformer model.

    Parameters:
        phrases1 (list of str): First set of phrases.
        phrases2 (list of str): Second set of phrases.
        model (SentenceTransformer): Preloaded SentenceTransformer model.

    Returns:
        str: "Similar" if the sets of phrases are similar, "Not Similar" otherwise.
    """

    # Compute embeddings for both lists
    embeddings1 = model.encode(phrases1, convert_to_tensor=True)
    embeddings2 = model.encode(phrases2, convert_to_tensor=True)

    # Compute cosine similarities
    similarities = util.pytorch_cos_sim(embeddings1, embeddings2)

    # Flatten the similarities tensor and exclude NaNs (if any)
    similarities_list = similarities.cpu().numpy().flatten()

    # Calculate statistics
    mean = np.mean(similarities_list)
    median = np.median(similarities_list)
    max = np.max(similarities_list)

    # Check similarity conditions
    return mean > 0.5 or median > 0.5 or max > 0.9
