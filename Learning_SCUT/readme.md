# Learning on SCUT-FBP dataset

This folder contains all the code corresponding to the learning on SCUT-FBP dataset, with different training/test configurations. It also contains the results of using a silly algorithm which always predicts the mean of the labels of the training data (mean_predictor.m).


# Prerequisites

libsvm library for Matlab.


# Contents

* 1NN: 1-Nearest Neighbor is used with different distances: the Minkowski distance (with an exponent equal to 1), the Euclidean distance and the cosine distance.

* FME: Flexible Manifold embedding is used with different Laplacian matrices, corresponding to three different similarities.

* KFME: Kernel Flexible Manifold embedding is used with different Laplacian matrices, corresponding to three different similarities.

* LGC: Local and Global Consistency is used with different Laplacian matrices, corresponding to three different similarities.

* Ridge_Regression: Ridge Regression is used.

* SVR: epsilon-insensitive Support Vector Regression is used with a linear and a Gaussian kernel with libsvm. 

* ZwbSVR: epsilon-insensitive Support Vector Regression is applied (with a linear kernel) to the features preprocessed with the Flexible Graph-based Semi-supervised Manifold Embedding. In addition, 1-NN classifier is applied to the preprocessed features using only three discrete classes.

* Cumulative_curve: the REC curve of the different algorithms is generated in order to have a graphical comparison of their performance.
