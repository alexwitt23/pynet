#!/usr/bin/env python3
"""Implementation of 
Density-based spatial clustering of applications with noise (DBSCAN).
This is the most cited clustering algorithm in scientific literature.
https://en.wikipedia.org/wiki/DBSCAN."""

from typing import List

import numpy as np 

from pynet.core import utils 


class DBSCAN:

    def __init__(self, epsilon: float, min_pts: int) -> None:
        """
        Args:
            epsilon: The distance between points in a cluster. In this case
            we will consider it Euclidean distance.
            min_pts: The minimum number of points in a cluster.
        """
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.already_counted: List[int] = []
        self.all_clusters: List[List[int]] = []

    def fit(self, x: np.ndarray) -> None:
        """Fit the data.
        
        Args:
            x: input data with shape [N x D] where N is the number of data 
            points and D is the dimension of each point.
        Returns:
        
        ??????
        """
        assert len(x.shape) == 2, "Input must be 2 dimensional "
        self.x = x
        for i in range(self.x.shape[0]):
            # Already found neighbors here, move on
            if i in self.already_counted:
                continue
            neighbor_ids = self._find_neighbors(self.x[i])

            # See if this counts as a cluster
            if len(neighbor_ids) >= self.min_pts:
                # Go through the cluster and expand to all possible points. 
                # Call recursive function to find the entire cluster starting from 
                # this list of neighbors
                self.all_clusters.append(
                    self._expand_cluster(i, neighbor_ids)
                )
                

        return self.all_clusters

    def _find_neighbors(self, source: np.ndarray) -> np.ndarray:
        """Loop through and get distance to this point."""
        # Record the _ids_ of points that are <= epsilon threshold
        return (
            [idx for idx, point in enumerate(self.x[:]) if utils.euclidean_distance(source, point) <= self.epsilon]
        )

    def _expand_cluster(
        self, 
        current_point: int, 
        neighbor_ids: np.ndarray
    ) -> np.ndarray:
        """Recursively go through cluster to expand to all points within epsilon.
        
        Args:
        ???????????/"""
        cluster = [current_point]
        for idx in neighbor_ids[:]:
            # Make sure this is a new point to consider
            if idx not in self.already_counted:
                self.already_counted.append(idx)
                # Get this neighbors of this point
                neighbor_ids = self._find_neighbors(self.x[idx])

                # Check that point is a core point
                if len(neighbor_ids) >= self.min_pts:
                    cluster.extend(self._expand_cluster(idx, neighbor_ids))
                else:
                    cluster.extend([idx])

        return cluster
        

if __name__ == '__main__':

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt
    
    num_centers = 4
    X, y_true = make_blobs(n_samples=300, centers=num_centers,
                        cluster_std=0.60, random_state=0)

    dbscan = DBSCAN(epsilon=.3, min_pts=3)
    clusters = dbscan.fit(X)
    
    plt.scatter(
        X[:, 0], 
        X[:, 1], 
        c='blue', 
        s=50, 
        cmap='viridis'
    )

    # Create colors for each cluster
    colors = np.random.rand(len(clusters), 3)
    for idx, cluster in enumerate(clusters):
        
        plt.scatter(
            X[cluster[:], 0], 
            X[cluster[:], 1], 
            c=colors[idx], 
            s=50, 
            cmap='viridis'
        )
        
    plt.show()