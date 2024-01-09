# -*- coding: utf-8 -*-
"""PCA.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w5I3gtjQhdA67M8Md5VmP73yM0hJFoO5

Explicit PCA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

d0 = pd.read_csv('/content/drive/MyDrive/Colab Notebooks/mnist_train.csv')
print(d0.head(5))
l = d0['label']

d = d0.drop('label', axis=1)

print(d.shape)
print(l.shape)

plt.figure(figsize=(7,7))
idx = 122

grid_data = d.iloc[idx].values.reshape(28,28)
plt.imshow(grid_data, interpolation = 'none', cmap = 'gray')
plt.show()
print(l[idx])

labels = l.head(15000)
data =  d.head(15000)

print(data.shape)

#data-preprocessing, stanardizing the data -> (x_i-u_i)/(standard deviation)_i
from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(data)
print(standardized_data.shape)

#Covariance matrix -> A^T * A (making a square matrix)
sample_data = standardized_data
#matrix multiplication
cover_matrix = np.matmul(sample_data.T, sample_data) #.t for transpose
print(cover_matrix.shape)

#finding top two eigen values and corresponding eigen vectors
from scipy.linalg import eigh
# 'sub_set_index' defined -> (low to high values), returns eigen vlues in ascending
values,vectors = eigh(cover_matrix, subset_by_index=(782,783)) #only top 2 eigen values (782 and  783)
vectors = vectors.T # 784x2 -> 2x784
print(vectors.shape)

#to tranform 784 to 2 dimenssions only top two eigen values required hence
#multiply vectors(2x784) an sample_data^T(784x15000)= 2x15000

new_cord = np.matmul(vectors, sample_data.T)
print(new_cord.shape, labels.shape)

#Adding label to 2d projected data
new_cord = np.vstack((new_cord, labels.T)).T
#New dataframe for plotting points
dataframe = pd.DataFrame(data=new_cord, columns=("1st Principle", "2nd Principle", 'label'))
print(dataframe.head())

# ploting the 2d data points with seaborn
import seaborn as sn
sn.FacetGrid(dataframe, hue='label',height=6).map(plt.scatter, '1st Principle','2nd Principle').add_legend()
plt.show()

"""Implicit PCA"""

#implicit PCA using sklearn->decomposition->pca
from sklearn import decomposition
pca = decomposition.PCA()

#configuring of parameter
#2 components needed bcz 2d visiulization is to be done
pca.n_components = 2

#transform data, standardization is done by pca itself (no dendrology)
#covariance matrix, eigen values and vectors all done by pca itself
pca_data = pca.fit_transform(sample_data)
print(pca_data.shape, labels.shape)

#Adding label to 2d projected data
pca_data = np.vstack((pca_data.T,labels.T)).T
#New dataframe for plotting points
pac_df = pd.DataFrame(data = pca_data, columns =('1st Principle', '2nd Principle', 'labels'))

# ploting the 2d data points with seaborn
sn.FacetGrid(dataframe, hue='label',height=6).map(plt.scatter, '1st Principle','2nd Principle').add_legend()
plt.show()

"""Meh job with PCA, waqt zaayi

"""