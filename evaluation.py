import numpy as np
import pandas as pd
from data_prep import build_interaction_matrix
from utility import get_ndcg_score_sk


def recommend(item_scores, seen_items_mask, top_k=10):
    maskable_scores = item_scores.astype(np.float64, copy=True)
    maskable_scores[seen_items_mask > 0] = -np.inf
    top_item_ids = np.argsort(-maskable_scores)[:top_k]
    return top_item_ids.tolist()


def evaluate(model, folds, n_users, n_items, top_k=10):
    per_fold_ndcg = []
    for train_fold, test_fold in folds:
        train_matrix = build_interaction_matrix(train_fold, n_users, n_items)
        test_matrix = build_interaction_matrix(test_fold, n_users, n_items)
        model.fit(train_matrix)

        test_user_ids = np.sort(test_fold["user_id"].unique())
        prediction_rows = []
        for user_id in test_user_ids:
            item_scores = model.score_user(user_id)
            recommended_item_ids = recommend(item_scores, train_matrix[user_id], top_k)
            recommendation_string = ",".join(str(item_id) for item_id in recommended_item_ids)
            prediction_rows.append({"user_id": int(user_id), "recs": recommendation_string})

        predictions = pd.DataFrame(prediction_rows)
        fold_ndcg = get_ndcg_score_sk(predictions, test_matrix, top_k)
        per_fold_ndcg.append(fold_ndcg)

    per_fold_ndcg = np.array(per_fold_ndcg, dtype=np.float64)
    return per_fold_ndcg, per_fold_ndcg.mean(), per_fold_ndcg.std()


def write_submission(model, interactions, n_users, n_items, submission_path, top_k=10):
    full_matrix = build_interaction_matrix(interactions, n_users, n_items)
    model.fit(full_matrix)
    with open(submission_path, "w") as submission_file:
        for user_id in range(n_users):
            item_scores = model.score_user(user_id)
            recommended_item_ids = recommend(item_scores, full_matrix[user_id], top_k)
            recommendation_string = ",".join(str(item_id) for item_id in recommended_item_ids)
            submission_file.write(f"{user_id}\t{recommendation_string}\n")