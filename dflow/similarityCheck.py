from typing import Any, List, Union
from sentence_transformers import SentenceTransformer, util
import numpy as np
import pandas as pd
from textx import get_children_of_type
from itertools import product

from dflow.language import get_metamodel

def similarityCheck(models: Union[List[Any], pd.DataFrame], sections: List[str] = None, model: SentenceTransformer = None, column_name: str = None):
    """
    Compare sections across multiple models for similarity based on cosine similarity,
    or compare sentences within a single column of a DataFrame, excluding exact self-comparisons.


    Args:
        models: List of models as strings, or a DataFrame containing text data.
        sections: List of sections to compare in the models (if models is a list).
        model: The sentence-transformer model for embedding comparison.
        column_name: The name of the column to analyze (if models is a DataFrame).
       
    Returns:
        dict: A dictionary with similarity results, including mean, median, max scores, and a similarity flag.
    """
    similarity_results = {}

    #left untouched
    if isinstance(models, pd.DataFrame) and column_name:
        # If models is a DataFrame, compare sentences in the specified column, excluding self-comparisons
        sentences = models[column_name].astype(str).tolist()
       
        # Compute embeddings for the sentences in the column
        embeddings = model.encode(sentences, convert_to_tensor=True)
       
        # Compute cosine similarity matrix
        similarities = util.pytorch_cos_sim(embeddings, embeddings)
       
        # Exclude self-comparisons by setting the diagonal to NaN
        similarities_np = similarities.cpu().numpy()
        np.fill_diagonal(similarities_np, np.nan)


        # Flatten the matrix and exclude NaNs
        similarities_list = similarities_np[~np.isnan(similarities_np)]
       
        # Calculate statistics
        mean_score = np.mean(similarities_list)
        median_score = np.median(similarities_list)
        max_score = np.max(similarities_list)
       
        similarity_results[column_name] = {
            'mean': mean_score,
            'median': median_score,
            'max': max_score,
            'similar': mean_score > 0.5 or median_score > 0.5 or max_score > 0.9
        }
   
    elif isinstance(models, list) and sections:
        # If models is a list of strings, compare specified sections across models
        mm = get_metamodel()
        parsed_models = [mm.model_from_str(model) for model in models]
        for section in sections:
            sentences1 = [phrase.phrases[0] for phrase in get_children_of_type(section, parsed_models[0])[0].phrases]
            sentences2 = [phrase.phrases[0] for phrase in get_children_of_type(section, parsed_models[1])[0].phrases]
            print(sentences1)
            print(sentences2)

            similarity_results[section] = checkSimilaritiesBetweenLists(sentences1, sentences2, model)
   
    else:
        raise ValueError("Invalid input: provide either a list of models with sections or a DataFrame with a column name.")
   
    return similarity_results

def checkSimilaritiesBetweenLists(list1: List, list2: List, model: SentenceTransformer):
    all_similarities = []
    #Combination of all items of list1 with all items of list2
    for item1, item2 in product(list1, list2):
        # Compute embeddings for each section
        embedding1 = model.encode(list1, convert_to_tensor=True)
        embedding2 = model.encode(list2, convert_to_tensor=True)

        # Cosine similarities between each pair of model sections
        similarities = util.pytorch_cos_sim(embedding1, embedding2)


        # Calculate mean, median, and max scores
        mean_score = similarities.mean().item()
        median_score = similarities.median().item()
        max_score = similarities.max().item()

        all_similarities.append({
            'mean': mean_score,
            'median': median_score,
            'max': max_score,
            'similar': mean_score > 0.5 or median_score > 0.5 or max_score > 0.9
        })
    return all_similarities

def main():
    # Example usage:
    # Load your model
    model = SentenceTransformer('all-MiniLM-L6-v2')


    # For multiple models (as strings)
    models_list = ["Model 1 text section", "Model 2 text section"]
    sections_to_compare = ["section1", "section2"]
    results_multi_models = similarityCheck(models=models_list, sections=sections_to_compare, model=model)



    # Example DataFrame
    data = {
        'TextColumn': [
            'This is the first sentence.',
            'This is the second sentence.',
            'This is the third sentence.',
            'This is a duplicate sentence.'
        ]
    }
    df = pd.DataFrame(data)
    results_single_column = similarityCheck(models=df, model=model, column_name='TextColumn')


    # Print results
    print("Results for multiple models:", results_multi_models)
    print("Results for single column:", results_single_column)
