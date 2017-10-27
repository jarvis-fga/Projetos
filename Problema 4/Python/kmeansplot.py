'''
ATENÇÃO: ESTE CÓDIGO ESTÁ COM PROBLEMAS E A EXECUÇÃO PODE EXTOURAR MEMÓRIA RAM.
'''
import pandas

features = [
    "mean_of_the_integrated_profile",
    "standard_deviation_of_the_integrated_profile",
    "excess_kurtosis_of_the_integrated_profile",
    "skewness_of_the_integrated_profile",
    "mean_of_the_DM-SNR_curve",
    "standard_deviation_of_the_DM-SNR_curve",
    "excess_kurtosis_of_the_DM-SNR_curve",
    "skewness_of_the_DM-SNR_curve",
    "class"
]

data = pandas.read_csv('../data/HTRU_2.csv', sep=",", names=features)
labels = data['class']

# #############################################

import numpy

def verify_missing_data(data, features):
    missing_data = []

    for feature in features:
        count = 0
        for x in range(0, len(data)):
            if type(data[feature][x]) is numpy.float64 or type(data[feature][x]) is numpy.int64:
                count = count + 1
        missing_data.append(count)
    print(missing_data)

verify_missing_data(data, features)

# ############################################

import numpy

number_samples, number_features = data.shape
number_labels = len(numpy.unique(labels))

from time import time
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

sample_size = 1900 # Pesquisar melhor valor para esse parâmetro usado na métrica silhouette

def bench_k_means(estimator, name, data):
    initial_time = time()
    estimator.fit(data)
    execution_time = time() - initial_time

    # metrics
    inertia = estimator.inertia_
    homogeneity_score = metrics.homogeneity_score(labels, estimator.labels_)
    completeness_score = metrics.completeness_score(labels, estimator.labels_)
    v_measure_score = metrics.v_measure_score(labels, estimator.labels_)
    adjusted_rand_score = metrics.adjusted_rand_score(labels, estimator.labels_)
    adjusted_mutual_info_score = metrics.adjusted_mutual_info_score(labels,  estimator.labels_)
    silhouette_score = metrics.silhouette_score(data, estimator.labels_, metric='euclidean', sample_size=sample_size)

    #show metrics
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, execution_time, inertia, homogeneity_score,completeness_score, v_measure_score,
             adjusted_rand_score, adjusted_mutual_info_score, silhouette_score))

print(90 * '_')
print('init\t\ttime\tinertia\t\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')

bench_k_means(KMeans(init='k-means++', n_clusters=number_labels, n_init=10),
              name="k-means++", data=data)

bench_k_means(KMeans(init='random', n_clusters=number_labels, n_init=10),
              name="random", data=data)

# in this case the seeding of the centers is deterministic, hence we run the
# kmeans algorithm only once with n_init=1
pca = PCA(n_components=number_labels).fit(data)
bench_k_means(KMeans(init=pca.components_, n_clusters=number_labels, n_init=1),
              name="PCA-based", data=data)
print(90 * '_')

# ###################################################################

import matplotlib.pyplot as plt

reduced_data = PCA(n_components=2).fit_transform(data)
print(reduced_data)
kmeans = KMeans(init='k-means++', n_clusters=len(labels), n_init=1)
print("2")
# kmeans.fit(reduced_data)
print("3")
h = .05 # Escala do grid pra plotar
# Quanto menor a escala, maior a precisão
'''
# Obtendo a grade de x's e y's
x_min = reduced_data[:, 0].min() - 1
x_max = reduced_data[:, 0].max() + 1
y_min = reduced_data[: ,1].min() - 1
y_max = reduced_data[:, 1].max() + 1
print(x_max, x_min, y_max, y_min)

xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Obtendo e colorindo os resultados
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(1)
plt.clf()
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()),
           cmap=plt.cm.Paired,
           aspect='auto', origin='lower')

plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k', markersize=2)

# Marcando os centróides
centroids = kmeans.cluster_centers_
plt.scatter(centroids[:, 0], centroids[:, 1],
            marker='x', s=169, linewidths=3,
            color='w', zorder=10)
plt.title("TOMA UM KMEANS AE CARAI")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.show()
'''
