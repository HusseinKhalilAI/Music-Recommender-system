# Music Recommendation from Implicit Listening Histories

Top-10 track recommendation from play counts alone, evaluated with nDCG@10 over three repeated holdout folds.

Six recommenders share one interface and one evaluation harness, so every number in this repo differs from its neighbour by exactly one design decision. The best system — an EASE/ALS score blend — reaches **nDCG@10 = 0.121**, roughly **20x** the popularity baseline and **59x** random ranking.

---

## Results

Mean ± std over three folds, on data never touched during hyperparameter search.

| Model | nDCG@10 | std |
|---|---|---|
| **EASE + ALS blend (w = 0.7)** | **0.1210** | 0.0020 |
| EASE | 0.1179 | 0.0019 |
| ALS | 0.1047 | 0.0031 |
| ItemKNN, collaborative | 0.0897 | 0.0021 |
| BPR-MF | 0.0777 | 0.0026 |
| ItemKNN, MusiCNN content features | 0.0094 | 0.0007 |
| Popularity | 0.0061 | 0.0005 |
| Random | 0.0021 | 0.0002 |

Two results are worth more than the ranking itself:

**Content features failed.** ItemKNN over MusiCNN audio embeddings scored 0.0094 — above random, but below a popularity list that ignores the user entirely. Audio similarity is not listening similarity: two tracks that sound alike are not evidence that the same person plays both. The collaborative version of the identical algorithm, differing only in where the item-item similarity comes from, scored 0.0897 — nearly ten times higher.

**A closed-form model beat everything learned.** EASE has no epochs, no learning rate, and one hyperparameter. It is a single ridge regression over the item-item Gram matrix with a zeroed diagonal, and it outperformed both matrix factorisation models at a fraction of the fitting cost.

---

## Data

**The dataset is not redistributable and is not included here.** This documents its structure precisely enough to point the pipeline at an equivalent corpus.

Scale: **2,795 users × 4,178 items**. Four tab-separated files.

### 1 — Interactions

```
lfm-challenge.inter_train  →  user_id, item_id, count
```

Play counts, one row per user-item pair. Binarised at load: any count ≥ 1 becomes a 1, everything else 0. The models never see the magnitude — this is implicit feedback, where a play is evidence of interest and a non-play is not evidence of dislike, only of absence.

Interactions per user are long-tailed; the fold construction below is built around that.

### 2 — Items

```
lfm-challenge.item  →  item_id, track/artist metadata, genre
```

`genre` arrives as a stringified list and is parsed on demand. Used for dimensioning and analysis; the recommenders do not consume metadata.

### 3 — Users

```
lfm-challenge.user  →  user_id, country, age_at_registration, gender
```

Loaded with sentinel values cleaned to `NaN` (`-1` ages, `"n"` genders, empty countries). Not consumed by any model here — kept because it is what a cold-start or fairness extension would need.

### 4 — MusiCNN audio embeddings

```
lfm-challenge.musicnn  →  item_id, e_0 ... e_<FILL: dim-1>
```

Precomputed audio embeddings, available for a subset of items. Loaded into a dense `(n_items, dim)` matrix with zeros for missing items, so item IDs index directly into it. This is the only content signal in the project, and it is the one that failed.

### Evaluation protocol

Held-out interactions are the relevance signal; no explicit ratings exist. For each test user the model scores all items, already-seen training items are masked to `-inf`, and the top 10 survivors are the recommendation. nDCG@10 is computed against the held-out interaction vector, then averaged over users and over folds.

Scoring calls the reference implementation directly rather than reimplementing it, so local numbers match the official ones by construction.

---

## What I did, step by step

### 1 — Fixed the evaluation before writing any model

Everything is built around one interface:

```python
model.fit(train_matrix)          # (n_users, n_items) binary
model.score_user(user_id)        # → (n_items,) scores
```

Nothing else is assumed. A neighbourhood model, a closed-form ridge solution, two factorisation models, and a blend all satisfy it, which is why `evaluate` is written once and every comparison is like-for-like. Adding a model means adding a class, not touching the harness.

### 2 — Per-user stratified folds, cached to disk

A global random split would leave some users with no training history and others with no test items, and would make results incomparable across runs. Instead each user's interactions are shuffled and split individually at 20%, with two guards: at least one interaction is always held out, and at least one is always kept in training. Users with a single interaction go entirely to training.

Three folds with fixed seeds are generated once and **written to disk**, then reloaded by every downstream script. Hyperparameter search, final evaluation, and plotting all read the same bytes, so nothing drifts between stages.

### 3 — Baselines that set the floor

Random ranking (0.0021) and popularity (0.0061) exist to make the other numbers interpretable. Popularity is the honest floor for any recommender: beating it is the minimum evidence that a model has learned something user-specific rather than reproducing global frequency.

### 4 — Neighbourhood models, two similarity sources

`ItemKNN` computes an item-item similarity matrix, zeroes the diagonal, and keeps only each item's top-k neighbours before scoring a user as their interaction vector times that sparsified matrix. Sparsification matters: a dense similarity matrix lets weak, noisy relationships accumulate into the score.

The same class runs in two modes:

- **Collaborative** — cosine over the item co-occurrence matrix. Similar means "played by the same people."
- **Content** — cosine over standardised MusiCNN embeddings. Similar means "sounds alike."

Holding the algorithm fixed and swapping only the similarity source is what makes the content result interpretable as a statement about the features rather than about ItemKNN.

### 5 — EASE

A linear autoencoder with a closed-form solution: invert the regularised item Gram matrix, rescale by its diagonal, then force the diagonal to zero so no item can predict itself. One hyperparameter, no iterations, and a single matrix inverse at 4,178 items.

**0.1179** — the strongest single model in the project.

### 6 — Matrix factorisation, two objectives

**ALS** for implicit feedback, with a confidence weight `α` on observed interactions, alternating ridge solves for user and item factors. **0.1047**.

**BPR-MF**, optimising a pairwise ranking loss in PyTorch: for each observed interaction, sample an unobserved item and push the positive's score above the negative's. **0.0777**.

Both lost to EASE. BPR-MF is also by far the most expensive model here — its negative sampling rejects and resamples per interaction per epoch in Python, which dominates its runtime.

### 7 — Hyperparameter search on a nested split

Tuning on the same folds used for reporting would leak. Instead the first fold's training portion is split again (90/10) into an inner train and validation set, and **every grid point is scored only on that inner split**. The three reporting folds are never touched during search.

| Model | Grid | Points |
|---|---|---|
| ItemKNN (both modes) | `n_neighbors` ∈ {10, 25, 50, 100, 200, 400} | 6 each |
| EASE | `reg_lambda` ∈ {50, 100, 250, 500, 1000, 2000} | 6 |
| ALS | `n_factors` × `reg` × `alpha` | 27 |
| BPR-MF | `embedding_size` × `learning_rate` × `reg` | 27 |

Winning configurations are written to `best_configs.json` and read back by the final evaluation, so the reported models are exactly the searched ones.

### 8 — Blending

EASE and ALS make different mistakes — one is a global item-item linear model, the other a low-rank user-item factorisation — so their errors are not identical and averaging should help. Raw scores are not comparable across models, so each is min-max normalised per user before the weighted sum.

The weight was swept over {0.3, 0.5, 0.7, 0.9} on the inner validation split, with 0.7 winning. Leaning toward EASE matches its solo advantage. Final test: **0.1210**, ahead of solo EASE by 0.0031 — small, but consistent across all three folds and larger than the fold-to-fold standard deviation of either model.

### 9 — Cost accounting

Fit and scoring time is recorded per model alongside nDCG, and plotted against it. A 0.003 gain that costs an order of magnitude more compute is a different proposition from a free one, and the plot makes that trade explicit rather than leaving it implied by a leaderboard.

---

## Repository structure

```
data.py             loading, binarisation, per-user fold construction and caching
models.py           all recommenders behind fit / score_user
evaluation.py       masked top-k recommendation, nDCG@10 over folds, submission writer
reference_metrics.py  reference nDCG implementation and matrix builder
hp_search.py        nested-split grid search + blend weight sweep
run_final_eval.py   score every tuned model on the three reporting folds
make_submission.py  fit the best system on all data, write top-10 per user
plots.py            performance, cost-vs-performance, interaction distribution
```

## Running it

```bash
pip install -r requirements.txt
export LFM_DATA_ROOT=/path/to/dataset

python hp_search.py        # writes best_configs.json
python run_final_eval.py   # writes final_test_results.json
python plots.py
python make_submission.py
```

Folds are created on first run and reused thereafter; delete `folds/` to regenerate.

---

## Findings

**Audio similarity is not listening similarity.** The single largest negative result here. Content features that describe how a track sounds carry almost no signal about who will play it next, at least as a drop-in replacement for collaborative similarity. Isolating this by changing one line inside a fixed algorithm is what makes it a finding rather than a guess.

**Simple and closed-form beat learned and iterative.** EASE — one hyperparameter, one matrix inverse — outperformed both factorisation models. Model capacity was not the constraint at this data scale.

**Evaluation design is load-bearing.** Per-user stratified folds, disk-cached splits, and a nested validation set are not ceremony; they are the difference between a number that transfers and one that does not. Most of the early work here went into the harness rather than the models.

**Blending is a real but small gain.** +0.0031 over solo EASE, consistent across folds, at roughly the sum of both models' cost. Whether that trade is worth making depends on the deployment, which is why runtime is reported next to accuracy.

## Limitations

- User and item metadata are loaded but unused; no cold-start path exists for a user or item with no interactions.
- Only accuracy is measured. Coverage, diversity, novelty, and popularity bias go unreported, and a popularity-driven model like ALS could be winning partly by concentrating on head items.
- The blend weight is tuned on a single inner split rather than cross-validated.
- BPR-MF's negative sampling is a Python loop; its result may understate the method given the compute budget it received.

## License

`<FILL: MIT / Apache-2.0>`
