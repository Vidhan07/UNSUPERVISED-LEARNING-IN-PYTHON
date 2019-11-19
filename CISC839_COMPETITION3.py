import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as plt

df = pd.read_csv('Last Data.csv')

data_rmf = df[['Recency', 'Total Amount', 'Frequency']]

data_rmf_1 = np.sqrt(data_rmf)

sc = StandardScaler()
sc.fit(data_rmf_1)
dt = sc.transform(data_rmf_1)

kmean = KMeans(n_clusters=6, random_state=1)
kmean.fit(dt)
cluster_labels = kmean.labels_

data_rmf = data_rmf.assign(Cluster=cluster_labels)

data_group = data_rmf.groupby(['Cluster'])

final_cluster_1 = data_group.agg({'Recency': ['mean', 'median', 'max'],  'Cluster': ['count']})
final_cluster_2 = data_group.agg({'Frequency': ['mean', 'median', 'max'],'Cluster': ['count']})
final_cluster_3 = data_group.agg({'Total Amount': ['mean', 'median', 'max'],'Cluster': ['count']})

print(final_cluster_1)
print(final_cluster_2)
print(final_cluster_3)


data_rmf.to_csv('Cluster_Data.csv', index=False)

squared_sum_dist = []
slope = []
slope_diff = []
K = range(1, 35)
for k in K:
    kmean = KMeans(n_clusters=k, random_state=2019)
    kmean.fit(dt)
    preds = kmean.fit_predict(dt)
    score = silhouette_score(dt, preds, metric='euclidean')
    print("For 6 clusters the silhouette is", score)
    squared_sum_dist.append(kmean.inertia_ / 1000)

prev = 0
for i in K:
    if i > len(squared_sum_dist) - 2:
        break
    else:
        slp = (squared_sum_dist[i + 1] - squared_sum_dist[i])
        if slp - prev < 0:
            slope.append(slp)
        prev = slp
    #    slope_diff.append(slope[i+1]-slope[i])

plt.figure(figsize=(16, 10))
plt.title('Elbow Method to predict the value of k')
plt.plot(K, squared_sum_dist, 'bx-')
plt.xticks(np.arange(0, 40, 1))
plt.xlabel('k')
plt.ylabel('Squared Sum distances')
plt.show()

from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(data_rmf_1)
pca_2d = pca.transform(data_rmf_1)

for i in range(0, pca_2d.shape[0]):
    if data_rmf['Cluster'][i] == 0:
        c1 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='r', marker='+')
    elif data_rmf['Cluster'][i] == 1:
        c2 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='g', marker='o')
    elif data_rmf['Cluster'][i] == 2:
        c3 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='y', marker='<')
    elif data_rmf['Cluster'][i] == 3:
        c4 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='b', marker='x')
    elif data_rmf['Cluster'][i] == 4:
        c5 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='k', marker='>')
    elif data_rmf['Cluster'][i] == 5:
        c6 = plt.scatter(pca_2d[i, 0], pca_2d[i, 1], c='m', marker='x')


plt.legend([c1, c2, c3, c4,c5,c6], ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3','Cluster 4', 'Cluster 5'])
plt.title("K-Means clustering for k=6")
plt.show()

for l in range(0, len(slope) - 1):
    slp1 = -1
    if slope[l] < slp1:
        slp1 = slope[l] - slope[l + 1]
        print(l, slp1)
