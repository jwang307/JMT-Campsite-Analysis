from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np


def plot_clusters(campsite_data, labels):
    '''
    plot pca plots of campsite data by score
    :param campsite_data: campsite data
    :param labels: label to color by (score or cluster)
    :return:
    '''
    pca_test = PCA(n_components=len(campsite_data[0]))
    pca_test.fit(campsite_data)
    plt.plot(pca_test.explained_variance_ratio_)
    plt.show()
    print(pca_test.explained_variance_)
    print(pca_test.components_)

    pca = PCA(n_components=2)
    pca.fit(campsite_data)
    transformed = pca.transform(campsite_data)
    plot = plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=labels)
    plt.title('PCA Plot for 2020 AAW Survey Campsites, Colored by Composite Score')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

    cov_mat = np.cov(campsite_data.T)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    print('Eigenvectors \n%s' % eig_vecs)
    print('\nEigenvalues \n%s' % eig_vals)
    print(pca.components_)
    print(pca.explained_variance_)

    return plot, np.array(pca.components_)


def plot_tsne(campsite_data, scores, clusters):
    '''
    plot tsne visualizations of campsite data colored by cluster
    :param campsite_data: campsite data
    :param scores: composite score label
    :param clusters: cluster id label
    :return:
    '''
    tsne = TSNE()
    transformed = tsne.fit_transform(campsite_data)

    plot_score = plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=scores)
    plt.title('t-SNE Plot for 2020 AAW Survey Campsites, Colored by Composite Score')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()

    plot_cluster = plt.figure()
    plt.scatter(transformed[:, 0], transformed[:, 1], c=clusters)
    plt.title('t-SNE Plot for 2020 AAW Survey Campsites, Colored by Cluster')
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.show()
    return plot_score, plot_cluster