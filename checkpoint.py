from evaluation import evaluate
from data_prep import infer_dimensions, load_folds
from models import RandomRecommender,ItemKNN,EASE,ALS,BPRMF,PopularityRecommender
import data_prep as dp 
import os

DATA_DIR = os.environ.get("LFM_DATA_ROOT", "data")

interactions = dp.load_interactions(os.path.join(DATA_DIR, r"lfm-challenge.inter_train"))
items = dp.load_items(os.path.join(DATA_DIR, r"lfm-challenge.item"))
users = dp.load_users(os.path.join(DATA_DIR, r"lfm-challenge.user"))
n_users, n_items = dp.infer_dimensions(items, users)
musicnn_features, _ = dp.load_musicnn(os.path.join(DATA_DIR, "lfm-challenge.musicnn"), n_items=n_items)




folds = dp.create_folds(interactions, folds_dir="folds")

models = [
    RandomRecommender(),
    PopularityRecommender(),
    ItemKNN(n_neighbors=50),
    EASE(reg_lambda=500.0),
    ALS(n_factors=64, n_iterations=10),
    BPRMF(embedding_size=64, n_epochs=100, learning_rate=0.05, reg=1e-5),
]

def demo_train():
    for model in models:
        per_fold, mean_ndcg, std_ndcg = evaluate(model, folds, n_users, n_items)
        print(f"{type(model).__name__}: mean {mean_ndcg:.5f}  std {std_ndcg:.5f}  {per_fold}")


# demo_train()

"""
-------------------------------------- Results ----------------------------------------

RandomRecommender: mean 0.00205  std 0.00020  [0.0017777  0.00223806 0.00213715]

PopularityRecommender: mean 0.00609  std 0.00049  [0.00677823 0.00573307 0.00575213]

ItemKNN: mean 0.07382  std 0.00127  [0.07446215 0.07495754 0.07205171]

EASE: mean 0.10725  std 0.00114  [0.10602149 0.10877695 0.10695305]

ALS: mean 0.08316  std 0.00188  [0.08052134 0.08468617 0.08428236]

BPRMF: mean 0.05383  std 0.00096  [0.05252313 0.05416579 0.05480532]


"""
