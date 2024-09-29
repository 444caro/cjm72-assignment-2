import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

def plot_clusters(X, y, centroids):
    # X: data, y: labels, centroids: cluster centers
    # create a new figure and axis, then plot the data points and centroids
    plt.figure(figsize=(8, 8))
    for i in range(len(centroids)):
        plt.scatter(X[y == i, 0], X[y == i, 1], s=30, label=f'Cluster {i}')
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, label='Centroids')
    # set plot labels and title
    plt.title('K-means Clustering')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    buf=BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    im = Image.open(buf)
    return im
    
    


