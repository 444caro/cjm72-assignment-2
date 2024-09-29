# Description: Implement the k-means algorithm
import numpy as np
from numpy.linalg import norm
import random

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, init_method='random', initial_centroids=None):
        # n_clusters: number of clusters, max_iter: maximum number of iterations, 
        # tol: tolerance, init_method: initialization method
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.centroids = None
        self.initial_centroids = initial_centroids
        
    def initialize_centroids(self, X):
        print(f"Initialization method: {self.init_method}")
        if self.init_method == 'random':
            centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)] 
            print(f"Randomly initialized centroids: {centroids}")
            return centroids
        elif self.init_method == 'kmeans++':
            centroids = [X[np.random.choice(X.shape[0], 1, replace=False)][0]]
            print(f"First kmeans++ centroid: {centroids[0]}")
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x - c)**2 for c in centroids]) for x in X])
                if np.any(np.isnan(distances)):
                    raise ValueError("NaN encountered in distance calculation during kmeans++ initialization")

                probabilities = distances / distances.sum()
                next_centroid = X[np.random.choice(X.shape[0], 1, p=probabilities)][0]
                centroids.append(next_centroid)
                print(f"Added kmeans++ centroid: {next_centroid}")
            print(f"Final initialized centroids (kmeans++): {centroids}")
            return np.array(centroids)
        elif self.init_method == 'farthest':
            centroids = [X[np.random.choice(X.shape[0], 1, replace=False)][0]]
            print(f"First farthest centroid: {centroids[0]}")
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x-c) for c in centroids]) for x in X])
                farthest = X[np.argmax(distances)]
                centroids.append(farthest)
                print(f"Added farthest centroid: {farthest}")
            print(f"Final initialized centroids (farthest): {centroids}")
            return np.array(centroids)
        elif self.init_method == 'manual':
            if self.initial_centroids is None or len(self.initial_centroids) != self.n_clusters:
                raise ValueError("Invalid initial centroids")
            print(f"Manually initialized centroids: {self.initial_centroids}")
            return np.array(self.initial_centroids)
        else:
            raise ValueError("Invalid initialization method")
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        if self.centroids is None:
            raise ValueError("Centroids not initialized")
        for _ in range(self.max_iter):
            labels = np.argmin(np.array([[norm(x - c) for c in self.centroids] for x in X]), axis=1)
            new_centroids = np.array([X[labels == i].mean(axis=0) for i in range(self.n_clusters)])
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            self.centroids = new_centroids
        return self
    
    def predict(self, X):
        distances = np.array([[norm(x-c) for c in self.centroids] for x in X])
        labels = np.argmin(distances, axis=1)
        return labels
    
    def fit_predict(self, X):
        return self.fit(X).predict(X)
    

