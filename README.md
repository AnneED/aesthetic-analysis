# aesthetic-analysis

With the progress made in deep learning and feature extraction, automatic facial beauty analysis has become an emerging research topic. However, the subjectivity of beauty still hinders the developement in this area, due to the cost of collecting reliable labeled data, since the beauty score of an individual has to be determined according to various raters. To address this problem, we study the performances of four different semi-supervised manifold based algorithms, which can take advantage of both labeled and unlabeled data in the training phase, and we use them in two different datasets: SCUT-FBP and M<sup>2</sup>B.
  
The semi-supervised algorithms used in this study are:
* Local and Global Consistency (LGC).
* Flexible Manifold Embedding (FME).
* Kernel Flexible Manifold Embedding (KFME).
* Flexible Graph-based Semi-supervised Embedding.

In addition, 3 supervised algorithms are studied:
* 1-Nearest Neighbor (NN).
* Ridge Regression (RR).
* Support Vector Regression (SVR).


## Content

There are 3 code folders:
* Feature_extraction: it contains the code to extract the features with the pretrained convolutional neural network [VGG-face](http://www.vlfeat.org/matconvnet/pretrained/).
* Learning_M2B: it contains the code of the learning algorithms on M<sup>2</sup>B dataset.
* Learning_SCUT: it contains the code of the learning algorithms on SCUT-FBP dataset.


## Data
SCUT-FBP dataset can be downloaded from [here](http://www.hcii-lab.net/data/SCUT-FBP/EN/download.html) and M<sup>2</sup>B dataset from [here](https://sites.google.com/site/vantam/beautysense).


## Prerequisites

The software prerequisites are the following:

```
Matlab 2016b or above
LIBSVM library
Caffe compiled for Matlab
```

