from numpy import unique
from numpy import where
from matplotlib import pyplot
from sklearn.datasets import make_classification
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import Birch
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture

# initialize the data set we'll work with
training_data, _ = make_classification(
    n_samples=1000,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_clusters_per_class=1,
    random_state=4
)

# define the models
affinity_model = AffinityPropagation(damping=0.7)
agglomerative_model = AgglomerativeClustering(n_clusters=2)
birch_model = Birch(threshold=0.03, n_clusters=2)
dbscan_model = DBSCAN(eps=0.25, min_samples=9)
kmeans_model = KMeans(n_clusters=2)
mean_model = MeanShift()
optics_model = OPTICS(eps=0.75, min_samples=10)
gaussian_model = GaussianMixture(n_components=2)

# train the model
affinity_model.fit(training_data)
birch_model.fit(training_data)
kmeans_model.fit(training_data)
gaussian_model.fit(training_data)

# assign each data point to a cluster
affinity_result = affinity_model.predict(training_data)
agglomerative_result = agglomerative_model.fit_predict(training_data)
birch_result = birch_model.predict(training_data)
dbscan_result = dbscan_model.fit_predict(training_data)
kmeans_result = kmeans_model.predict(training_data)
mean_result = mean_model.fit_predict(training_data)
optics_result = optics_model.fit_predict(training_data)
gaussian_result = gaussian_model.predict(training_data)

# get all of the unique clusters
affinity_clusters = unique(affinity_result)
agglomerative_clusters = unique(agglomerative_result)
birch_clusters = unique(birch_result)
dbscan_clusters = unique(dbscan_result)
kmeans_clusters = unique(kmeans_result)
mean_clusters = unique(mean_result)
optics_clusters = unique(optics_result)
gaussian_clusters = unique(gaussian_result)

# plot Affinity Propagation the clusters
for affinity_cluster in affinity_clusters:
    # get data points that fall in this cluster
    index = where(affinity_result == affinity_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Affinity Propagation plot
pyplot.show()

# plot the Agglomerative Hierarchy clusters
for agglomerative_cluster in agglomerative_clusters:
    # get data points that fall in this cluster
    index = where(agglomerative_result == agglomerative_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Agglomerative Hierarchy plot
pyplot.show()

# plot the BIRCH clusters
for birch_cluster in birch_clusters:
    # get data points that fall in this cluster
    index = where(birch_result == birch_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the DBSCAN plot
pyplot.show()

# plot the DBSCAN clusters
for dbscan_cluster in dbscan_clusters:
    # get data points that fall in this cluster
    index = where(dbscan_result == dbscan_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the DBSCAN plot
pyplot.show()

# plot the KMeans clusters
for kmeans_cluster in kmeans_clusters:
    # get data points that fall in this cluster
    index = where(kmeans_result == kmeans_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the KMeans plot
pyplot.show()

# plot Mean-Shift the clusters
for mean_cluster in mean_clusters:
    # get data points that fall in this cluster
    index = where(mean_result == mean_cluster)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Mean-Shift plot
pyplot.show()

# plot OPTICS the clusters
for optics_cluster in optics_clusters:
    # get data points that fall in this cluster
    index = where(optics_result == optics_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the OPTICS plot
pyplot.show()

# plot Gaussian Mixture the clusters
for gaussian_cluster in gaussian_clusters:
    # get data points that fall in this cluster
    index = where(gaussian_result == gaussian_clusters)
    # make the plot
    pyplot.scatter(training_data[index, 0], training_data[index, 1])

# show the Gaussian Mixture plot
pyplot.show()