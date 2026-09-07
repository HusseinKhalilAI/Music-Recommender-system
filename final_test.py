import json
import data_prep as dp
from evaluation import evaluate
from checkpoint import n_users, n_items, musicnn_features
from models import (RandomRecommender, PopularityRecommender, ItemKNN,
                    EASE, ALS, BPRMF, BlendedRecommender)
import time



folds = dp.load_folds(folds_dir="folds")

with open("best_configs.json") as f:
    best = json.load(f)

knn_collab_k = best["ItemKNN_collaborative"][0]["n_neighbors"]
knn_content_k = best["ItemKNN_content"][0]["n_neighbors"]
ease_lambda = best["EASE"][0]["reg_lambda"]
als_config = best["ALS"][0]
bpr_config = best["BPRMF"][0]
blend_weight = best["Blend_EASE_ALS"][0]["weight"]

als_fixed = {"n_iterations": 15, "seed": 42}
bpr_fixed = {"n_epochs": 100, "batch_size": 1024, "seed": 42}

final_models = {
    "Random": RandomRecommender(),
    "POP": PopularityRecommender(),
    "ItemKNN_collab": ItemKNN(n_neighbors=knn_collab_k),
    "ItemKNN_content": ItemKNN(n_neighbors=knn_content_k, item_embeddings=musicnn_features),
    "BPRMF": BPRMF(**bpr_config, **bpr_fixed),
    "ALS": ALS(**als_config, **als_fixed),
    "EASE": EASE(reg_lambda=ease_lambda),
    "Blend_EASE_ALS": BlendedRecommender(
        EASE(reg_lambda=ease_lambda),
        ALS(**als_config, **als_fixed),
        weight=blend_weight),
}

final_results = {}

for name, model in final_models.items():

    start = time.time()
    per_fold, mean_ndcg, std_ndcg = evaluate(model, folds, n_users, n_items)
    elapsed = time.time() - start
    print(f"{name}: mean {mean_ndcg:.5f}  std {std_ndcg:.5f}  per_fold {[f'{x:.5f}' for x in per_fold]}")

    final_results[name] = {
    "mean": float(mean_ndcg),
    "std": float(std_ndcg),
    "per_fold": [float(x) for x in per_fold],
    "time_sec": elapsed,
}

with open("final_test_results.json", "w") as f:
    json.dump(final_results, f, indent=2)





"""
Final Test Results:

Random: mean 0.00205  std 0.00020  per_fold ['0.00178', '0.00224', '0.00214']
POP: mean 0.00609  std 0.00049  per_fold ['0.00678', '0.00573', '0.00575']
ItemKNN_collab: mean 0.08967  std 0.00211  per_fold ['0.09045', '0.09177', '0.08678']
ItemKNN_content: mean 0.00941  std 0.00074  per_fold ['0.00934', '0.00854', '0.01036']
BPRMF: mean 0.07765  std 0.00259  per_fold ['0.07438', '0.08072', '0.07785']
ALS: mean 0.10467  std 0.00305  per_fold ['0.10156', '0.10882', '0.10365']
EASE: mean 0.11788  std 0.00191  per_fold ['0.11618', '0.12055', '0.11691']
Blend_EASE_ALS: mean 0.12098  std 0.00197  per_fold ['0.11863', '0.12345', '0.12084']

"""