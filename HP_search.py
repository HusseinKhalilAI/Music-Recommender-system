from data_prep import make_fold
import data_prep as dp
from itertools import product
from evaluation import evaluate
from checkpoint import n_items,n_users,musicnn_features
from models import ItemKNN,EASE,ALS,BPRMF,BlendedRecommender
import json


HYPERPARAMETER_GRIDS = {
    "ItemKNN_collaborative": {
        "n_neighbors": [10, 25, 50, 100, 200, 400],
    },
    "ItemKNN_content": {
        "n_neighbors": [10, 25, 50, 100, 200, 400],
    },
    "EASE": {
        "reg_lambda": [50, 100, 250, 500, 1000, 2000],
    },
    "ALS": {
        "n_factors": [32, 64, 128],
        "reg": [0.01, 0.1, 1.0],
        "alpha": [10, 40, 100],
    },
    "BPRMF": {
        "embedding_size": [32, 64, 128],
        "learning_rate": [0.01, 0.05, 0.1],
        "reg": [1e-6, 1e-5, 1e-4],
    },
}

folds = dp.load_folds(folds_dir="folds")
train_fold, test_fold = folds[0]
inner_train, val = dp.make_fold(train_fold, seed=99, test_fraction=0.1)



def HP_search(grids: dict):

    def grid_product(model_grid):                     
        keys = list(model_grid.keys())
        for combo in product(*model_grid.values()):    
            yield dict(zip(keys, combo))
    
    MODELS = {
    "ItemKNN_collaborative": ItemKNN,
    "ItemKNN_content": ItemKNN,
    "EASE": EASE,
    "ALS": ALS,
    "BPRMF": BPRMF,
    }

    FIXED_HYPERPARAMETERS = {
    "ALS": {"n_iterations": 15, "seed": 42},
    "BPRMF": {"n_epochs": 100, "batch_size": 1024, "seed": 42},
    }



    results = {}
    overall_best_score = -1.0
    overall_best_model = None
    overall_best_config = None

    with open("hp_search_scores.txt", "w") as results_file:
        for model_name, model_class in MODELS.items():
            model_grid = grids[model_name]
            fixed = FIXED_HYPERPARAMETERS.get(model_name, {})

            best_score = -1.0
            best_config = None

            for config in grid_product(model_grid):
                if model_name == "ItemKNN_content":
                    model = model_class(**config, item_embeddings=musicnn_features)
                else:
                    model = model_class(**config, **fixed)

                _, val_score, _ = evaluate(model, [(inner_train, val)], n_users, n_items)

                results_file.write(f"{model_name} | {config} | Validation nDCG: {val_score:.5f}\n")
                results_file.flush()

                if val_score > best_score:
                    best_score = val_score
                    best_config = config

            results_file.write(f"> BEST {model_name} | {best_config} | Validation nDCG: {best_score:.5f} <\n\n")
            results_file.flush()
            results[model_name] = (best_config, best_score)

            if best_score > overall_best_score:
                overall_best_score = best_score
                overall_best_model = model_name
                overall_best_config = best_config

        results_file.write(f">>> OVERALL BEST | {overall_best_model} | {overall_best_config} | Validation nDCG: {overall_best_score:.5f} <<<\n")
        results_file.flush()

    return results




            
results = HP_search(HYPERPARAMETER_GRIDS)

with open("best_configs.json", "w") as f:
    json.dump(results, f)


best_weight = None
best_blend_score = -1.0

for w in [0.3, 0.5, 0.7, 0.9]:
    blend = BlendedRecommender(
        EASE(reg_lambda=100),
        ALS(n_factors=128, reg=0.01, alpha=10, n_iterations=15, seed=42),
        weight=w,
    )
    _, val_score, _ = evaluate(blend, [(inner_train, val)], n_users, n_items)
    print(f"weight={w}: val_ndcg={val_score:.5f}")

    if val_score > best_blend_score:
        best_blend_score = val_score
        best_weight = w

with open("best_configs.json") as f:
    all_best = json.load(f)

all_best["Blend_EASE_ALS"] = ({"weight": best_weight}, best_blend_score)

with open("best_configs.json", "w") as f:
    json.dump(all_best, f, indent=2)

print(f"BEST blend weight={best_weight}: Validation nDCG={best_blend_score:.5f}")    



"""
BlendedRecommender Results:

weight=0.3: val_ndcg=0.06828
weight=0.5: val_ndcg=0.07197
weight=0.7: val_ndcg=0.07354 
weight=0.9: val_ndcg=0.07255

"""