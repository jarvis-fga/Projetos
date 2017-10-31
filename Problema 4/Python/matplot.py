
import pandas
import numpy
import numpy as np

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

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

data = pandas.read_csv('data/HTRU_2.csv', sep=",", names=features)
labels = data['class']


# #############################################################################
k_means = KMeans(init='k-means++', n_clusters=2, n_init=10)
