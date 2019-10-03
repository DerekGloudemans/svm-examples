from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np

data = datasets.make_moons(n_samples=2000, noise=0.3)
X,y = data
plt.figure()
plt.suptitle('Moons Data Sample')
colors = np.array(['r','b'])
plt.scatter(X[:, 0], X[:, 1], color=colors[y],s=3)