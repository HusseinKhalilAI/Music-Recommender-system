
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score
from pandas.api.types import is_numeric_dtype


def inter_matr_implicit(users: int,
                       items: int,
                       interactions: pd.DataFrame,
                       threshold=1) -> np.ndarray:
    """
    Create an implicit interaction matrix from user-item interactions.
    
    Parameters:
        users: DataFrame containing user information
        items: DataFrame containing item information
        interactions: DataFrame containing user-item interaction data
        threshold: Minimum value for a valid interaction (default: 1)
        
    Returns:
        2D numpy array where rows represent users and columns represent items
    """
    interactions = interactions.copy()
    res = np.zeros([users, items], dtype=np.int8)

    row = interactions['user_id'].to_numpy()
    col = interactions["item_id"].to_numpy()

    data = interactions['count'].to_numpy(copy=True)
    data[data < threshold] = 0
    data[data >= threshold] = 1

    res[row, col] = data

    return res


def filter_interactions_by_users(interactions, user_ids): # not used
    """
    Filter interaction data to only include specified users.
    
    Parameters:
        interactions: DataFrame containing interaction data
        user_ids: List of user IDs to keep
        
    Returns:
        Filtered DataFrame
    """
    return interactions[interactions['user_id'].isin(user_ids)]


def get_ndcg_score_sk(df_predictions, test_interaction_matrix: np.ndarray, topK=10) -> float:
    """
    Calculate the NDCG score for recommendation predictions.
    
    Parameters:
        df_predictions: DataFrame containing recommendation predictions
        test_interaction_matrix: Ground truth interaction matrix
        topK: Number of top recommendations to evaluate (default: 10)
        
    Returns:
        Average NDCG score across all users
    """
    ndcg_avg = 0
    
    for _, row in df_predictions.iterrows():
        g_truth = test_interaction_matrix[row["user_id"]]

        predicted_scores = np.zeros(len(g_truth),dtype=np.int8)

        predictions = list(map(int, row["recs"].split(",")))[:topK]

        for j, rec in enumerate(predictions):
            predicted_scores[rec] = topK - j

        ndcg_avg += ndcg_score(g_truth.reshape(1, -1), predicted_scores.reshape(1, -1), k=topK)

    return ndcg_avg/len(df_predictions)


def main():
    """Main function to evaluate recommendation predictions."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate recommendation system predictions")
    parser.add_argument("--submission", type=str, required=True, help="Path to submission file")
    parser.add_argument("--target", type=str, default=None, help="Path to target interactions file")
    parser.add_argument("--topK", type=int, default=10, help="Number of top recommendations to evaluate")
    args = parser.parse_args()
    
    # Load data
    df_submission = pd.read_csv(
        args.submission, 
        sep=r'(?<!,)\s+', 
        header=None, 
        names=['user_id', 'recs'], 
        engine='python'
    )
    
    print(args.target)
    
    test_interactions = pd.read_csv(
        args.target, 
        sep=r'(?<!,)\s+', 
        engine='python'
    )
        
    test_users = test_interactions['user_id'].unique()
    topK = args.topK

    # Create interaction matrix
    test_interaction_matrix = inter_matr_implicit(
        users=test_interactions['user_id'].max()+1, 
        items=test_interactions['item_id'].max()+1, 
        interactions=test_interactions
    )

    # Validate submission format
    if type(df_submission.index) is pd.MultiIndex:
        raise ValueError(
            "Submission format is not correct, please adhere to the format specified in the task description"
        )
    
    if not is_numeric_dtype(df_submission["user_id"]):
        raise ValueError("Provided user IDs are not numeric!")


    if len(df_submission) < len(test_users): # the input can contain more users, we just want to keep some of them
        raise ValueError(
            f"Recommendations for {len(test_users)} users expected but {len(df_submission)} were provided"
        )

    # Filter submissions to match test users
    df_submission_sorted = df_submission.sort_values(by=["user_id"])
    df_sk_input = df_submission_sorted[df_submission_sorted['user_id'].isin(test_users)].reset_index()

    if len(df_sk_input) != len(test_users):
        raise ValueError(
            f"Missing recommendations for test users."
        )

    # Calculate NDCG score
    result = get_ndcg_score_sk(df_sk_input, test_interaction_matrix, topK)
    
    return result


if __name__ == "__main__":
    # Print out the results that will be displayed in the challange
    print(main())