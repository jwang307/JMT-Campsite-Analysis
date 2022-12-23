import os.path

import preprocess
import clustering
import logreg
import plotting


# TODO:
#   finish documentation of current functions
#   split data into train and test (and valid?) set
#   implement classification algorithms on dataset
#   generate plot and results of classification
#   finish documentation and formatting of everything

if __name__ == "__main__":
    csv_path = "AggregatedData.csv"
    annotated_path = "AnnotatedData.csv"
    scaled_path = "ScaledData.csv"
    results_dir = 'results'
    max_clusters = 30
    features = ["vegetation_diff", "grass_diff", "area", "condition_class", "exposed_soil", "dist_water"]
    # write encoded data and return scaled data along with calculated campsite scores
    scaled_data, scores = preprocess.preprocessed(csv_path, annotated_path, scaled_path, results_dir, features)

    # clustering
    best_k = clustering.find_best_k(scaled_data, max_clusters)
    print(best_k)
    clusters = clustering.kmeans(scaled_data, best_k)

    # plot dim reduction
    pca, components = plotting.plot_clusters(scaled_data, scores)
    tsne_score, tsne_cluster = plotting.plot_tsne(scaled_data, scores, clusters.labels_)

    result_file = 'result_{}.csv'

    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    pca.savefig(os.path.join(results_dir, 'pca_plot.png'), dpi=300)
    tsne_score.savefig(os.path.join(results_dir, 'tsne_plot.png'), dpi=300)
    tsne_cluster.savefig(os.path.join(results_dir, 'tsne_plot_cluster.png'), dpi=300)
