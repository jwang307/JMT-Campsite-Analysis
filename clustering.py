from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from matplotlib import pyplot as plt


def find_best_k(campsite_data, max_clusters):
    sil = []

    for k in range(2, max_clusters + 1):
        clusters = KMeans(n_clusters=k).fit(campsite_data)
        labels = clusters.labels_
        sil.append(silhouette_score(campsite_data, labels, metric="euclidean"))

    x_axis_sil = np.linspace(2, max_clusters, num=max_clusters - 1)
    plt.plot(x_axis_sil, sil)
    plt.show()
    return np.argmax(sil) + 2


def kmeans(campsite_data, n_clusters):
    clustered = KMeans(n_clusters=n_clusters, n_init=100).fit(campsite_data)

    return clustered
