#!/usr/bin/env python3
"""K-means clustering algorithm."""

from typing import List, Tuple
import copy

import numpy as np


class KMeans:
    def __init__(self, num_centroids: int, num_iterations: int) -> None:
        self.k = num_centroids
        self.num_iter = num_iterations

    def fit(self, x: np.ndarray):
        """Entrypoint for K-means clustering.

        Args:
            x: data to be clustered. x.shape == [N, D],
            where N is the number of data points and D is the dimension
            of the data.
        Returns:
            Numpy array [num_centroids, D] corresponding to fitted clusters.
        """
        assert len(x.shape) == 2, "Data must be in format [N x D]!"
        self.x = x
        # Intialize the center points
        self.initialize()
        for i in range(self.num_iter):
            # Assign the clusters
            self.assign_clusters()
            # Get the centers before update to monitor convergence
            prev_centers = copy.deepcopy(self.centers)
            # Update the center points
            self.update_centers()
            # If no centerpoint movement, coverged
            if np.array_equal(prev_centers, self.centers):
                break

        return self.centers

    # TODO (alex) make more initialization techniques
    def initialize(self) -> None:
        """Randomly initialize a given k number of centroids."""
        num_points = self.x.shape[0]
        points = np.random.choice(num_points, self.k)
        self.centers = self.x[points]

        return None

    def assign_clusters(self) -> List[Tuple[int, np.ndarray]]:
        """Loop over the points and assign each points to a cluster 
        based on Euclidean distance."""
        self.clusters: List[Tuple[int, np.ndarray]] = []  # Cluster_id and point
        for point in self.x[:]:
            # Get the to all the centers from this point
            distances = [
                euclidean_distance(point, center) for center in self.centers[:]
            ]
            # Get the min distance, cooresponds to nearest cluster center.
            self.clusters.append((np.argmin(distances), point))

        return None

    def update_centers(self) -> None:
        """Get the average point out of each cluster which becomes 
        the new cluster center."""
        for i in range(self.k):
            new_x = np.mean(
                [point[0] for cluster_id, point in self.clusters if cluster_id == i]
            )
            new_y = np.mean(
                [point[1] for cluster_id, point in self.clusters if cluster_id == i]
            )
            self.centers[i] = np.array([new_x, new_y])

        return None


if __name__ == "__main__":

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt

    num_centers = 4
    X, y_true = make_blobs(
        n_samples=300, centers=num_centers, cluster_std=0.60, random_state=0
    )
    # num_centers of colors
    colors = np.random.choice(255, size=(num_centers, 3)) / 255

    k_means = KMeans(num_centroids=4, num_iterations=50)
    y_kmeans = k_means.fit(X)

    plt.scatter(X[:, 0], X[:, 1], c="blue", s=50, cmap="viridis")
    plt.scatter(y_kmeans[:, 0], y_kmeans[:, 1], c="red", s=200, alpha=0.5)
    plt.show()
