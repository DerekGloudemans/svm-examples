# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 11:04:21 2019

@author: derek
"""

from sklearn.svm import SVC
from sklearn import datasets
from bonnerlib import dfContour, df3D
import matplotlib.pyplot as plt
import numpy as np

# get 2200 random samples (from same distribution to make things a tiny bit easier)
X,y = datasets.make_moons(n_samples=2200, noise= 0.4)
X_train = X[0:200]
y_train = y[0:200]
X_test = X[200:]
y_test = y[200:]

# get a random item from X_test
idx = np.random.randint(0,2000)
print("Test item {}: X = {}, y = {}".format(idx,X[idx],y[idx]))

# define gammas
grid_size = 30
gammas = np.logspace(-3,5,grid_size)
Cs = np.logspace(-3,3,grid_size)
scores = np.zeros([grid_size,grid_size])

# loop through all combinations
for i in range(len(gammas)):
    for j in range(len(Cs)):
        gamma = gammas[i]
        C = Cs[j]
        
        # fit to training data
        clf = SVC(gamma=gamma,C=C)
        clf.fit(X_train,y_train)
        
        # score accuracy on test data
        scores[i,j] = clf.score(X_test,y_test)

fig = plt.imshow(scores, cmap='viridis', interpolation='nearest')
plt.ylabel("gamma")
plt.xlabel("C")

n_labels = 11
step = int(grid_size / (n_labels - 1)) # step between consecutive labels
positions = np.arange(0,grid_size,step) # pixel count at label position

x_labels = Cs[::step] # labels you want to see
x_labels = ["{:.3f}".format(lab) for lab in x_labels]
plt.xticks(positions, x_labels,rotation = -70)

y_labels = gammas[::step] # labels you want to see
y_labels = ["{:.3f}".format(lab) for lab in y_labels]
plt.yticks(positions, y_labels)

plt.show()

#print(scores)