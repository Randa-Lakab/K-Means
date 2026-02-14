# kmeans.py
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=3)

model = KMeans(n_clusters=3)
model.fit(X)

labels = model.labels_

plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
