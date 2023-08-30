import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
X, _ = make_blobs(n_samples=300, centers=3, random_state=42)
n_components = 3
gmm = GaussianMixture(n_components=n_components, random_state=42)
gmm.fit(X)
labels = gmm.predict(X)
probs = gmm.predict_proba(X)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis')
plt.colorbar()
plt.title('Gaussian Mixture Model Clustering')
plt.show()
