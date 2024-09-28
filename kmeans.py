# Description: Implement the k-means algorithm
import numpy as np
from numpy.linalg import norm
import random

class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4, init_method='random'):
        # n_clusters: number of clusters, max_iter: maximum number of iterations, 
        # tol: tolerance, init_method: initialization method
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_method = init_method
        self.centroids = None
        
    def initialize_centroids(self, X):
        if self.init_method == 'random':
            return X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        elif self.init_method == 'kmeans++':
            centroids = [X[np.random.choice(X.shape[0], 1, replace=False)][0]]
            for _ in range(1, self.n_clusters):
                distances = np.array([min([np.linalg.norm(x-c)**2 for c in centroids]) for x in X])
                probabilities = distances / distances.sum()
                centroids.append(X[np.random.choice(X.shape[0], 1, p=probabilities)])
            return np.array(centroids)
        else:
            raise ValueError("Invalid initialization method")
        return centroids
    
    def fit(self, X):
        self.centroids = self.initialize_centroids(X)
        for _ in range(self.max_iter):
            labels = np.argmin(np.array([[norm(x-c) for c in self.centroids] for x in X]), axis=1)
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
    

