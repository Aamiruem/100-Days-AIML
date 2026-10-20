import random
import numpy as np


class KMeans:
    def __init__(self, n_clusters=2, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state
        self.centroids = None

    def fit_predict(self, X):
        if self.n_clusters <= 0:
            raise ValueError("n_clusters must be greater than 0")
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot be greater than the number of samples")

        rng = random.Random(self.random_state)
        random_index = rng.sample(range(0, X.shape[0]), self.n_clusters)
        self.centroids = X[random_index]

        for _ in range(self.max_iter):
            # assign clusters
            cluster_group = self.assign_clusters(X)
            old_centroids = self.centroids
            # move centroids
            self.centroids = self.move_centroids(X, cluster_group)
            # check finish
            if np.allclose(old_centroids, self.centroids):
                break

        return cluster_group

    def assign_clusters(self, X):
        cluster_group = []

        for row in X:
            distances = []
            for centroid in self.centroids:
                distances.append(np.sqrt(np.dot(row-centroid, row-centroid)))
            min_distance = min(distances)
            index_pos = distances.index(min_distance)
            cluster_group.append(index_pos)

        return np.array(cluster_group)

    def move_centroids(self, X, cluster_group):
        new_centroids = []

        for cluster_id in range(self.n_clusters):
            cluster_points = X[cluster_group == cluster_id]
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[cluster_id])
            else:
                new_centroids.append(cluster_points.mean(axis=0))

        return np.array(new_centroids)

