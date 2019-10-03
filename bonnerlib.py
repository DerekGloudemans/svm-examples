#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 28 13:35:55 2017

@author: anthonybonner
"""

from sklearn.svm import SVC
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# plot the data points in red and blue (for 0 and 1, respectively).
# then plot the contours of the decision function (of classifier clf)
# and highlight the decision boundary in solid black.
# If margins=1 then highlight the margins in dashed black

def dfContour(clf,data,margins=0,colors = ['r','b'],title = ""):
    
    X,y = data
    # plot the data
    plt.scatter(X[:, 0], X[:, 1], color=colors[y],s=3)
    plt.title(title)
    
    # form a mesh/grid to cover the data
    h = 0.02
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.decision_function(mesh)
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=[-4,-3,-2,-1,0,1,2,3,4], cmap=cm.RdBu, alpha=0.5)
    
    # draw the decision boundary in solid black
    plt.contour(xx, yy, Z, levels=[0], colors='k', linestyles='solid')
    if margins:
        # draw the margins in dashed black
        plt.contour(xx, yy, Z, levels=[-1,1], colors='k', linestyles='dashed')
        


# plot the decision function of classifier clf in 3D.
# if Cflag=1, place a contour plot of the decision function beneath the 3D plot.
# (Use data to determine the range of the axes)

def df3D(clf,data,cFlag=1,colors = ['r','b'],title = ""):    
    
    # form a mesh/grid to cover the data
    h = 0.01
    X,y = data
    x_min = X[:, 0].min() - .5
    x_max = X[:, 0].max() + .5
    y_min = X[:, 1].min() - .5
    y_max = X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    mesh = np.c_[xx.ravel(),yy.ravel()]
    
    # evaluate the decision functrion at the grid points
    Z = clf.decision_function(mesh)
    Z = -Z    # to improve the 3D plot for the Moons data set, negate Z
    
    # plot the contours of the decision function
    Z = Z.reshape(xx.shape)
    plt.figure()
    

    ax = plt.axes(projection='3d')
    ax.plot_surface(xx, yy, Z, cmap=cm.RdBu, linewidth=0.2,edgecolor='k')
    plt.title(title)
    if cFlag == 1:
        # display a contour plot of the decision function
        Zmin = np.min(Z) - 1.0
        ax.contourf(xx, yy, Z, cmap=cm.RdBu, offset=Zmin)
        ax.set_zlim(bottom=Zmin)
    

