#!/usr/bin/env python3
"""Implementation of 
Density-based spatial clustering of applications with noise (DBSCAN).
This is the most cited clustering algorithm in scientific literature.
https://en.wikipedia.org/wiki/DBSCAN."""

from typing import List

import numpy as np 

from pynet.core import utils 


class DBSCAN:

    def __init__(self, epsilon, min_pts) -> None:
        """
        Args:
            epsilon: The distance between points in a cluster. In this case
            we will consider it Euclidean distance.
            min_pts: The minimum number of points in a cluster.
        """
        self.epsilon = epsilon
        self.min_pts = min_pts
        self.already_counted: List[int] = []
        self.all_clusters: List[np.ndarray] = []

    def fit(self, x: np.ndarray) -> None:
        """Fit the data.
        
        Args:
            x: input data with shape [N x D] where N is the number of data 
            points and D is the dimension of each point.
        Returns:
        
        ??????
        """
        assert len(x.shape) == 2, "Input must be shape "
        self.x = x
        for i in range(self.x.shape[0]):
            # Already found neighbors here, move on
            if i in self.already_counted:
                continue

            neighbors = self._find_neighbors(self.x[i])
            # See if this counts as a cluster
            if neighbors.shape[0] >= self.min_pts:
                # Go through the cluster and expand to all possible points. 
                # Keep track of the points with known neighbors
                self.already_counted.append(i)
                # Call recursive function to find the entire cluster starting from 
                # this list of neighbors
                cluster = self._expand_cluster(x[neighbors])
                self.all_clusters.append(cluster)
        return None


    def _find_neighbors(self, source: np.ndarray) -> np.ndarray:
        """Loop through and get distance to this point."""
        # Record the ids of points that are <= epsilon threshold
        cluster_ids = np.array(
            [idx2 for idx2, point in enumerate(self.x[:]) if utils.euclidean_distance(source, point) <= self.epsilon]
        )

        return cluster_ids

    def _expand_cluster(self, points: np.ndarray) -> np.ndarray:
        """Recursivly go through cluster to expand to all points within epsilon."""
        for point in points[:]:
            # Make sure this is a new point to consider
            print(point, self.already_counted)
            if point not in self.already_counted:
                # Get this neighbors of this point
                neighbors = self.already_counted.append(point)
                if neighbors.shape[0] >= self.min_pts:
                    additional_cluster = self._expand_cluster(neighbors)

        return additional_cluster
        

if __name__ == '__main__':

    from sklearn.datasets.samples_generator import make_blobs
    import matplotlib.pyplot as plt
    
    num_centers = 4
    X, y_true = make_blobs(n_samples=300, centers=num_centers,
                        cluster_std=0.60, random_state=0)

    dbscan = DBSCAN(epsilon=1, min_pts=3)
    dbscan.fit(X)