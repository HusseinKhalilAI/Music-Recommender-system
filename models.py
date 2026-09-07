import numpy as np


class RandomRecommender:
    def __init__(self, seed=42):
        self.seed = seed
        self.n_items: int | None = None
        self.random_generator: np.random.Generator | None = None

    def fit(self, train_matrix):
        self.n_items = train_matrix.shape[1]
        self.random_generator = np.random.default_rng(self.seed)

    def score_user(self, user_id):
        assert self.random_generator is not None and self.n_items is not None
        return self.random_generator.random(self.n_items)


class PopularityRecommender:
    def __init__(self):
        self.item_popularity: np.ndarray | None = None

    def fit(self, train_matrix):
        self.item_popularity = train_matrix.sum(axis=0).astype(np.float64)

    def score_user(self, user_id):
        return self.item_popularity


class ItemKNN:
    def __init__(self, n_neighbors=10, item_embeddings=None):
        self.n_neighbors = n_neighbors
        self.item_embeddings: np.ndarray | None = item_embeddings
        self.item_similarity: np.ndarray | None = None
        self.train_matrix: np.ndarray | None = None

    def fit(self, train_matrix, item_embeddings=None):
        self.train_matrix = train_matrix
        features = item_embeddings if item_embeddings is not None else self.item_embeddings

        if features is None:
            binary_matrix = (train_matrix > 0).astype(np.float64)
            cooccurrence = binary_matrix.T @ binary_matrix
            item_norms = np.sqrt(np.diag(cooccurrence))
            safe_norms = np.where(item_norms == 0.0, 1.0, item_norms)
            similarity = cooccurrence / safe_norms[:, None] / safe_norms[None, :]
        else:
            features = features.astype(np.float64)
            features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-8)
            feature_norms = np.sqrt((features ** 2).sum(axis=1))
            safe_norms = np.where(feature_norms == 0.0, 1.0, feature_norms)
            normalized_features = features / safe_norms[:, None]
            similarity = normalized_features @ normalized_features.T

        np.fill_diagonal(similarity, 0.0)

        n_items = similarity.shape[0]
        neighbor_count = min(self.n_neighbors, n_items - 1)
        top_neighbor_indices = np.argpartition(-similarity, neighbor_count, axis=1)[:, :neighbor_count]
        sparsified = np.zeros_like(similarity)
        np.put_along_axis(sparsified, top_neighbor_indices,
                          np.take_along_axis(similarity, top_neighbor_indices, axis=1), axis=1)
        self.item_similarity = sparsified

    def score_user(self, user_id):
        assert self.item_similarity is not None and self.train_matrix is not None
        user_interactions = self.train_matrix[user_id].astype(np.float64)
        return user_interactions @ self.item_similarity


class EASE:
    def __init__(self, reg_lambda=500.0):
        self.reg_lambda = reg_lambda
        self.weight_matrix: np.ndarray | None = None
        self.train_matrix: np.ndarray | None = None

    def fit(self, train_matrix):
        self.train_matrix = train_matrix
        binary_matrix = (train_matrix > 0).astype(np.float64)
        gram_matrix = binary_matrix.T @ binary_matrix
        diagonal_indices = np.diag_indices(gram_matrix.shape[0])
        gram_matrix[diagonal_indices] += self.reg_lambda
        inverse_gram = np.linalg.inv(gram_matrix)
        weight_matrix = inverse_gram / (-np.diag(inverse_gram))[None, :]
        weight_matrix[diagonal_indices] = 0.0
        self.weight_matrix = weight_matrix

    def score_user(self, user_id):
        assert self.train_matrix is not None and self.weight_matrix is not None
        user_interactions = self.train_matrix[user_id].astype(np.float64)
        return user_interactions @ self.weight_matrix


class BPRMF:
    def __init__(self, embedding_size=64, learning_rate=0.01, n_epochs=20, reg=0.01, batch_size=1024, seed=42):
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.reg = reg
        self.batch_size = batch_size
        self.seed = seed
        self.user_embeddings: np.ndarray | None = None
        self.item_embeddings: np.ndarray | None = None

    def fit(self, train_matrix):
        import torch

        torch.manual_seed(self.seed)
        sampling_generator = np.random.default_rng(self.seed)
        n_users, n_items = train_matrix.shape

        positive_pairs = np.argwhere(train_matrix > 0)
        positive_users = positive_pairs[:, 0]
        positive_items = positive_pairs[:, 1]
        n_positive = len(positive_pairs)

        interacted_items_per_user = [set() for _ in range(n_users)]
        for user_id, item_id in positive_pairs:
            interacted_items_per_user[user_id].add(item_id)

        user_embedding_table = torch.nn.Embedding(n_users, self.embedding_size)
        item_embedding_table = torch.nn.Embedding(n_items, self.embedding_size)
        torch.nn.init.normal_(user_embedding_table.weight, std=0.01)
        torch.nn.init.normal_(item_embedding_table.weight, std=0.01)
        optimizer = torch.optim.Adam(
            list(user_embedding_table.parameters()) + list(item_embedding_table.parameters()),
            lr=self.learning_rate, weight_decay=self.reg)

        for epoch in range(self.n_epochs):
            shuffled_order = sampling_generator.permutation(n_positive)
            negative_items = sampling_generator.integers(0, n_items, size=n_positive)
            for position in range(n_positive):
                user_id = positive_users[shuffled_order[position]]
                while negative_items[position] in interacted_items_per_user[user_id]:
                    negative_items[position] = sampling_generator.integers(0, n_items)

            for batch_start in range(0, n_positive, self.batch_size):
                batch_indices = shuffled_order[batch_start:batch_start + self.batch_size]
                batch_users = torch.as_tensor(positive_users[batch_indices], dtype=torch.long)
                batch_positive_items = torch.as_tensor(positive_items[batch_indices], dtype=torch.long)
                batch_negative_items = torch.as_tensor(negative_items[batch_start:batch_start + self.batch_size], dtype=torch.long)

                user_vectors = user_embedding_table(batch_users)
                positive_vectors = item_embedding_table(batch_positive_items)
                negative_vectors = item_embedding_table(batch_negative_items)
                positive_scores = (user_vectors * positive_vectors).sum(dim=1)
                negative_scores = (user_vectors * negative_vectors).sum(dim=1)
                loss = -torch.nn.functional.logsigmoid(positive_scores - negative_scores).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        self.user_embeddings = user_embedding_table.weight.detach().numpy()
        self.item_embeddings = item_embedding_table.weight.detach().numpy()

    def score_user(self, user_id):
        assert self.user_embeddings is not None and self.item_embeddings is not None
        return self.user_embeddings[user_id] @ self.item_embeddings.T


class ALS:
    def __init__(self, n_factors=64, reg=0.1, alpha=40.0, n_iterations=15, seed=42):
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha
        self.n_iterations = n_iterations
        self.seed = seed
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train_matrix):
        initialization_generator = np.random.default_rng(self.seed)
        n_users, n_items = train_matrix.shape
        binary_matrix = (train_matrix > 0)

        items_per_user = [np.where(binary_matrix[user_id])[0] for user_id in range(n_users)]
        users_per_item = [np.where(binary_matrix[:, item_id])[0] for item_id in range(n_items)]

        user_factors = initialization_generator.normal(0.0, 0.01, (n_users, self.n_factors))
        item_factors = initialization_generator.normal(0.0, 0.01, (n_items, self.n_factors))
        regularization = self.reg * np.eye(self.n_factors)

        for iteration in range(self.n_iterations):
            item_gram = item_factors.T @ item_factors
            for user_id in range(n_users):
                interacted = items_per_user[user_id]
                if len(interacted) == 0:
                    continue
                interacted_factors = item_factors[interacted]
                system_matrix = item_gram + self.alpha * (interacted_factors.T @ interacted_factors) + regularization
                target_vector = (1.0 + self.alpha) * interacted_factors.sum(axis=0)
                user_factors[user_id] = np.linalg.solve(system_matrix, target_vector)

            user_gram = user_factors.T @ user_factors
            for item_id in range(n_items):
                interacting = users_per_item[item_id]
                if len(interacting) == 0:
                    continue
                interacting_factors = user_factors[interacting]
                system_matrix = user_gram + self.alpha * (interacting_factors.T @ interacting_factors) + regularization
                target_vector = (1.0 + self.alpha) * interacting_factors.sum(axis=0)
                item_factors[item_id] = np.linalg.solve(system_matrix, target_vector)

        self.user_factors = user_factors
        self.item_factors = item_factors

    def score_user(self, user_id):
        assert self.user_factors is not None and self.item_factors is not None
        return self.user_factors[user_id] @ self.item_factors.T
    

########################################### Experimenting Blending EASE + ALS #################################################################
class BlendedRecommender:
    def __init__(self, model_a, model_b, weight=0.5):
        self.model_a = model_a
        self.model_b = model_b
        self.weight = weight

    def fit(self, train_matrix):
        self.model_a.fit(train_matrix)
        self.model_b.fit(train_matrix)

    def score_user(self, user_id):
        scores_a = self._normalize(self.model_a.score_user(user_id))
        scores_b = self._normalize(self.model_b.score_user(user_id))
        return self.weight * scores_a + (1.0 - self.weight) * scores_b

    def _normalize(self, scores):
        finite_scores = scores[np.isfinite(scores)]
        lowest = finite_scores.min()
        highest = finite_scores.max()
        return (scores - lowest) / (highest - lowest + 1e-8)
    


