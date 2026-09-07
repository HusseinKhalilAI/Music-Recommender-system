import data_prep as dp
from evaluation import write_submission
from checkpoint import n_users, n_items, interactions
from models import BlendedRecommender, EASE, ALS

best_model = BlendedRecommender(
    EASE(reg_lambda=100),
    ALS(n_factors=128, reg=0.01, alpha=10, n_iterations=15, seed=42),
    weight=0.7,
)

write_submission(best_model, interactions, n_users, n_items, "final.tsv", top_k=10)

print("Done.")