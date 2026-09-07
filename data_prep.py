import os
import ast
import numpy as np
import pandas as pd
from utility import inter_matr_implicit


DEFAULT_SEEDS = (13, 37, 61)
DEFAULT_TEST_FRACTION = 0.20


def load_interactions(interactions_path: str) -> pd.DataFrame:
    interactions = pd.read_csv(interactions_path, sep="\t")
    return interactions.reset_index(drop=True)


def load_items(items_path: str, parse_genre: bool = False) -> pd.DataFrame:
    items = pd.read_csv(items_path, sep="\t")
    if parse_genre:
        items["genre"] = items["genre"].apply(parse_genre_string)
    return items.reset_index(drop=True)


def load_users(users_path: str, clean_missing: bool = True) -> pd.DataFrame:
    users = pd.read_csv(users_path, sep="\t")
    if clean_missing:
        users["age_at_registration"] = users["age_at_registration"].replace(-1, np.nan)
        users["gender"] = users["gender"].replace("n", np.nan)
        users["country"] = users["country"].replace("", np.nan)
    return users.reset_index(drop=True)


def load_musicnn(musicnn_path: str, n_items: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    musicnn = pd.read_csv(musicnn_path, sep="\t")
    present_item_ids = musicnn["item_id"].to_numpy()
    embedding_columns = [column for column in musicnn.columns if column != "item_id"]
    present_embeddings = musicnn[embedding_columns].to_numpy(dtype=np.float32)

    embedding_dim = present_embeddings.shape[1]
    total_items = n_items if n_items is not None else int(present_item_ids.max()) + 1
    item_embeddings = np.zeros((total_items, embedding_dim), dtype=np.float32)
    item_embeddings[present_item_ids] = present_embeddings
    return item_embeddings, present_item_ids


def infer_dimensions(items: pd.DataFrame, users: pd.DataFrame) -> tuple[int, int]:
    n_users = int(users["user_id"].max()) + 1
    n_items = int(items["item_id"].max()) + 1
    return n_users, n_items


def build_interaction_matrix(interactions: pd.DataFrame,
                             n_users: int,
                             n_items: int,
                             threshold: int = 1) -> np.ndarray:
    return inter_matr_implicit(users=n_users,
                               items=n_items,
                               interactions=interactions,
                               threshold=threshold)


def make_fold(interactions: pd.DataFrame,
              seed: int,
              test_fraction: float = DEFAULT_TEST_FRACTION) -> tuple[pd.DataFrame, pd.DataFrame]:
    random_generator = np.random.default_rng(seed)
    test_row_indices: list[int] = []
    train_row_indices: list[int] = []

    for _, user_interactions in interactions.groupby("user_id", sort=True):
        user_row_indices = user_interactions.index.to_numpy().copy()
        n_user_interactions = len(user_row_indices)
        if n_user_interactions < 2:
            train_row_indices.extend(user_row_indices.tolist())
            continue
        random_generator.shuffle(user_row_indices)
        n_held_out = max(1, int(round(test_fraction * n_user_interactions)))
        n_held_out = min(n_held_out, n_user_interactions - 1)
        test_row_indices.extend(user_row_indices[:n_held_out].tolist())
        train_row_indices.extend(user_row_indices[n_held_out:].tolist())

    train_fold = interactions.loc[train_row_indices].sort_values(["user_id", "item_id"]).reset_index(drop=True)
    test_fold = interactions.loc[test_row_indices].sort_values(["user_id", "item_id"]).reset_index(drop=True)
    return train_fold, test_fold


def create_folds(interactions: pd.DataFrame,
                 folds_dir: str = "folds",
                 seeds: tuple[int, ...] = DEFAULT_SEEDS,
                 test_fraction: float = DEFAULT_TEST_FRACTION,
                 overwrite: bool = False) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    os.makedirs(folds_dir, exist_ok=True)
    folds: list[tuple[pd.DataFrame, pd.DataFrame]] = []

    for fold_index, seed in enumerate(seeds):
        train_path = os.path.join(folds_dir, f"fold{fold_index}_train.inter_train")
        test_path = os.path.join(folds_dir, f"fold{fold_index}_test.inter")

        if os.path.exists(train_path) and os.path.exists(test_path) and not overwrite:
            train_fold = pd.read_csv(train_path, sep="\t")
            test_fold = pd.read_csv(test_path, sep="\t")
        else:
            train_fold, test_fold = make_fold(interactions, seed=seed, test_fraction=test_fraction)
            train_fold.to_csv(train_path, sep="\t", index=False)
            test_fold.to_csv(test_path, sep="\t", index=False)

        folds.append((train_fold, test_fold))

    return folds


def load_folds(folds_dir: str = "folds",
               n_folds: int = len(DEFAULT_SEEDS)) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    folds = []
    for fold_index in range(n_folds):
        train_fold = pd.read_csv(os.path.join(folds_dir, f"fold{fold_index}_train.inter_train"), sep="\t")
        test_fold = pd.read_csv(os.path.join(folds_dir, f"fold{fold_index}_test.inter"), sep="\t")
        folds.append((train_fold, test_fold))
    return folds


def parse_genre_string(genre_value):
    if isinstance(genre_value, list):
        return genre_value
    if not isinstance(genre_value, str) or not genre_value.strip():
        return []
    try:
        parsed = ast.literal_eval(genre_value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


