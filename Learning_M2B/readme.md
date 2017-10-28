# Learning on M<sup>2</sup>B dataset

This folder contains all the code corresponding to the learning on M<sup>2</sup>B dataset, with different training/test configurations. It also contains the results of using a silly algorithm which always predicts the mean of the labels of the training data (mean_predictor_M2B.m).


# Prerequisites

libsvm library for Matlab.


# Contents

* FME: Flexible Manifold embedding is used with different Laplacian matrices, corresponding to three different similarities.

* KFME: Kernel Flexible Manifold embedding is used with different Laplacian matrices, corresponding to three different similarities.

* LGC: Local and Global Consistency is used with different Laplacian matrices, corresponding to three different similarities.

* linearSVR: epsilon-insensitive Support Vector Regression is used with a linear kernel, using libsvm library. 

* ZwbSVR: epsilon-insensitive Support Vector Regression is applied (with a linear kernel) to the features preprocessed with the Flexible Graph-based Semi-supervised Manifold Embedding. In addition, 1-NN classifier is applied to the preprocessed features using only three discrete classes.

