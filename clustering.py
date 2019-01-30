#!/usr/bin/env python
# File: clustering.py
# Author: Sharvari Deshpande <shdeshpa@ncsu.edu>

import pandas as pd
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as shc
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler
import matplotlib.cm as cm
from sklearn.mixture import GMM
from matplotlib.patches import Ellipse


#Reading of a CSV file
df = pd.read_csv('shdeshpa.csv', header=None)
print(df, df.shape)

# #----------------------------------------TASK 1-----------------------------------------------#
plt.figure(figsize=(10, 7))
plt.title("Dendrogram")
plt.axhline(y=3000, color='black')
den = shc.dendrogram(shc.linkage(df, method='ward'))
plt.show()

cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
prediction = cluster.fit_predict(df)
print(prediction)

#3D Scatter Plot
fig = plt.figure()
plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
x = df.iloc[:, 0]
y = df.iloc[:, 1]
z = df.iloc[:, 2]
bx = Axes3D(fig)
bx.scatter(x, y, z, c=cluster.labels_, cmap='rainbow')
plt.title('3D Scatter Plot using Hierarchical Clustering')
silhouette_avg1 = silhouette_score(df, cluster.labels_)
print('Silhouette Score Average of Hierarchical Clustering:', silhouette_avg1)

# #-------------------------------------------------TASK 2-------------------------------------------------#

Sum_of_squared_distances = []
K = range(2, 10)
for k in K:
    km = KMeans(n_clusters=k)
    km = km.fit(df)
    labels = km.predict(df)
    Sum_of_squared_distances.append(km.inertia_)
    centroids = km.cluster_centers_

print(len(K), len(Sum_of_squared_distances))
plt.plot(K, Sum_of_squared_distances, 'bx-')
plt.xlabel('k')
plt.ylabel('Sum of Squared Distances')
plt.title('Elbow Method For Optimal k')
plt.show()

range_n_clusters = [2, 3, 4, 5, 6, 7, 8, 9, 10]
sil_avg = []

for n_clusters in range_n_clusters:
    fig = plt.figure()
    ax1 = plt.subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')
    fig.set_size_inches(8, 7)
    ax1.set_xlim([-0.1, 1])
    ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = clusterer.fit_predict(df)
    silhouette_avg = silhouette_score(df, cluster_labels)
    sil_avg.append(silhouette_avg)
    print("For n_clusters =", n_clusters,
          "The average silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(df, cluster_labels)
    y_lower_1 = 10
    for i in range(n_clusters):
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower_1 + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower_1, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower_1 + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("The silhouette plot for the various clusters.")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
    a = df.iloc[:, 0]
    b = df.iloc[:, 1]
    c = df.iloc[:, 2]
    ax2.scatter(a, b, c, marker='.', s=30, lw=0, alpha=0.7, c=colors, edgecolor='k')
    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], centers[:, 2], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')

    ax2.set_title("The visualization of the clustered data")
    plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                  "with n_clusters = %d" % n_clusters), fontweight='bold')
    plt.show()
plt.plot(range_n_clusters, sil_avg)
plt.xlabel('k')
plt.ylabel('Silhouette Score Average')
plt.show()

# #----------------------------------TASK 3---------------------------------------#
stscaler = StandardScaler().fit(df)
df_1 = stscaler.transform(df)

minpts = 4
model_nn = NearestNeighbors(n_neighbors=len(df_1)).fit(df_1)
distances, indices = model_nn.kneighbors(df_1)
distance_to_k = distances[:, minpts]
print(distance_to_k)
# plt.figure(figsize=(, 10))
plt.plot(range(1, len(df_1)+1), sorted(distance_to_k, reverse=True))
plt.title("For Minpts = "+str(minpts))
plt.ylabel('Epsilon')
plt.xlabel('Data Points')
plt.show()

e = 0.375
mpt = 4
dbsc = DBSCAN(eps=e, min_samples=mpt).fit(df_1)
labels = dbsc.labels_
core_samples = np.zeros_like(labels, dtype=bool)
core_samples[dbsc.core_sample_indices_] = True
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

fig = plt.figure()
ax2 = fig.add_subplot(111, projection='3d')

# Black removed and is used for noise instead.
unique_labels = set(labels)
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
for k, col in zip(unique_labels, colors):
    if k == -1:
        # Black used for noise.
        col = 'k'

    class_member_mask = (labels == k)
    ax2.set_title('Estimated number of clusters: %d for minPts = %d, Radius = %.2f' % (n_clusters_, mpt, e), loc="left")
    #     ax1.set_title('Scatter Plot')
    ax2.set_xlabel('x axis')
    ax2.set_ylabel('y axis')
    ax2.set_zlabel('z axis')
    # Plotting core samples

    xy = df_1[class_member_mask & core_samples]
    s = xy[:, 0]
    h = xy[:, 1]
    d = xy[:, 2]
    ax2.plot(s, h, d , 'o', markerfacecolor=col, markeredgecolor='k', markersize=14)
    # PLotting outliers
    xy = df_1[class_member_mask & ~core_samples]
    ax2.plot(s, h, d, 'o', markerfacecolor=col,markeredgecolor='k', markersize=6)
plt.show()

print("Silhouette Coefficient for Minpts=4 is: %0.3f"
     % metrics.silhouette_score(df_1, labels))


#-----------------------------GAUSSIAN DECOMPOSITION CLUSTERING METHOD----------------------------#

#AIC and BIC
n_components = np.arange(1, 21)
models = [GMM(n, covariance_type='full', random_state=0).fit(df_1)
          for n in n_components]

plt.plot(n_components, [m.bic(df_1) for m in models], label='BIC')
plt.plot(n_components, [m.aic(df_1) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('k')
plt.show()
fig = plt.figure()
dx = fig.add_subplot(111, projection='3d')

gmm = GMM(n_components=5).fit(df_1)
labels = gmm.predict(df_1)
dx.scatter(df_1[:, 0], df_1[:, 1], df_1[:, 2], c=labels, s=40, cmap='viridis')
plt.show()
probs = gmm.predict_proba(df_1)
print(probs[:5].round(3))
silhouette_avg2 = silhouette_score(df_1, labels)
print('Silhouette Score Average of GMM Clustering:', silhouette_avg2)

