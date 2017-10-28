# Prerequisites

Scut-fbp and M2B datasets, as well as vgg-face neural network, have to be downloaded in order to run these scripts.


# Matlab codes in this folder

## feature_extraction_vgg.m 

It extracts the features of SCUT-FBP and M<sup>2</sup>B datasets from layer 6 and layer 7 in vgg-face and stores them in different matrices.


## initial_data_vgg_SCUT.m 

It generates the initial data corresponding to SCUT-FBP dataset, that is: 
* The training/test partitions: 10 training/test partitions are created according to 3 different configurations: 50%training/50%test, 70%training/30%test and 90%training/10%test. These partitions are created maintaining the distribution of the classes defined in initial_data_vgg_SCUT.m.
* The feature matrix: L2 normalization + pca is applied to the features extracted from vgg-face.
* The normalized labels: the labels are normalized so that they lie on the interval (0,1). This is done by dividing the original labels by 5.

## initial_data_vgg_M2B.m: 

It generates the initial data corresponding to M<sup>2</sup>B dataset similarly to initial_data_vgg_SCUT.m.


## histogram_classes.m 

It generates histograms of the discrete classes of SCUT-FBP and M<sup>2</sup>B datasets defined in initial_data_vgg_SCUT.m and initial_data_vgg_M2B.m.
