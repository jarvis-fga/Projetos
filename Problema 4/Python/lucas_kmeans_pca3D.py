# Create by Lucas Andrade
# On 30 / 10 / 2017

import codecs
import pandas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D

features = [
    "mean_of_the_integrated_profile",
    "standard_deviation_of_the_integrated_profile",
    "excess_kurtosis_of_the_integrated_profile",
    "skewness_of_the_integrated_profile",
    "mean_of_the_DM_SNR_curve",
    "standard_deviation_of_the_DM_SNR_curve",
    "excess_kurtosis_of_the_DM_SNR_curve",
    "skewness_of_the_DM_SNR_curve",
    "class"
]

data = pandas.read_csv('../data/HTRU_2.csv', sep=",", names=features)
labels = data['class']

f1 = data.mean_of_the_integrated_profile
f2 = data.standard_deviation_of_the_integrated_profile
f3 = data.excess_kurtosis_of_the_integrated_profile
f4 = data.skewness_of_the_integrated_profile
f5 = data.mean_of_the_DM_SNR_curve
f6 = data.standard_deviation_of_the_DM_SNR_curve
f7 = data.excess_kurtosis_of_the_DM_SNR_curve
f8 = data.skewness_of_the_DM_SNR_curve

X = np.matrix(zip(f1, f2, f3, f4, f5, f6, f7, f8))

pca = PCA(n_components=3)
pca.fit(X)
new_pca = pca.transform(X)

print("PCA done")

kmeans = KMeans(n_clusters=4).fit(new_pca)
print("KMeans done")

xx = np.array([])
yy = np.array([])
zz = np.array([])

x_pulsar = np.array([])
y_pulsar = np.array([])
z_pulsar = np.array([])
j=0
pulsar=0

# Build the arrays X and Y to plot
for i in new_pca:
	xx = np.append(xx, i[0])
	yy = np.append(yy, i[1])
	zz = np.append(zz, i[2])
	
	# See the pulsar results
	if (labels[j] == 1):
		pulsar+=1
		x_pulsar = np.append(x_pulsar, i[0])
		y_pulsar = np.append(y_pulsar, i[1])
		z_pulsar = np.append(z_pulsar, i[2])
	j+=1
print("Total pulsar: %d"%pulsar)

print("Start graph:")

import random


fig = plt.figure()
ax = Axes3D(fig)

# ax.scatter(x_pulsar, y_pulsar, z_pulsar, c='g')
ax.scatter(xx[kmeans.labels_==0], yy[kmeans.labels_==0], zz[kmeans.labels_==0], c='r')
ax.scatter(xx[kmeans.labels_==1], yy[kmeans.labels_==1], zz[kmeans.labels_==1], c='g')
ax.scatter(xx[kmeans.labels_==2], yy[kmeans.labels_==2], zz[kmeans.labels_==2], c='b')
ax.scatter(xx[kmeans.labels_==3], yy[kmeans.labels_==3], zz[kmeans.labels_==3], c='m')

ax.set_xlabel('EIxo X')
ax.set_ylabel('EIxo Y')
ax.set_zlabel('EIxo Z')
plt.title('3D PCA data with KMeans 4 groups')

plt.show()	