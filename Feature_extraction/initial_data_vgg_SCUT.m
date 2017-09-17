


% Obtaining the labels:
addpath('/home/john-san/Dropbox/Master/TFM/Data/SCUT-FBP/Rating_Collection');
labels0 = xlsread('Attractiveness_label.xlsx', 'B2:B501');
% std:
devs0 = xlsread('Attractiveness_label.xlsx', 'C2:C501');


% Scaling the labels:
labelsn = labels0/5;
devsn = devs0/5;


% Creation of 5 classes:
classes = zeros(length(labels0), 1);
for i = 1:length(labels0)
    if labels0(i) < 2
        classes(i) = 1;
    elseif labels0(i) < 3
        classes(i) = 2;
    elseif labels0(i) < 4
        classes(i) = 3;
    elseif labels0(i) < 4.5
        classes(i) = 4;
    else
        classes(i) = 5;
    end
end



% Creating training/test partitions with stratification (mantaining the
% original class distribution)
labeled_masks50 = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes, 'HoldOut', 0.5);
    labeled_masks50 = [labeled_masks50 partition.training];
end

labeled_masks70 = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes, 'HoldOut', 0.3);
    labeled_masks70 = [labeled_masks70 partition.training];
end

labeled_masks90 = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes, 'HoldOut', 0.1);
    labeled_masks90 = [labeled_masks90 partition.training];
end

labeled_masks50 = logical(labeled_masks50);
labeled_masks70 = logical(labeled_masks70);
labeled_masks90 = logical(labeled_masks90);


% Loading the data matrices and preprocessing
load('matrices_vgg.mat');
X = X_vgg_6_SCUT;
XX = X_vgg_7_SCUT;

% L2 normalization:
for kk = 1:size(X, 2)
X(:, kk) = X_vgg_6_SCUT(:, kk) / norm(X_vgg_6_SCUT(:, kk));
XX(:, kk) = X_vgg_7_SCUT(:, kk) / norm(X_vgg_7_SCUT(:, kk));
end

% pca:
[~, SCORE] = pca(X');
Xpca_6 = (SCORE(:, 1:200) )';

[~, SCORE] = pca(XX');
Xpca_7 = (SCORE(:, 1:200) )';

% Percentage of variability explained:
[~, ~, ~, ~, explained, ~] = pca(XX');
sum(explained(1:200))
% 90.3169

save('initial_data_SCUT_vgg', 'X_vgg_6_SCUT', 'X_vgg_7_SCUT', 'Xpca_6', 'Xpca_7', 'labeled_masks50', 'labeled_masks70', 'labeled_masks90', 'labels0', 'labelsn', 'devs0', 'devsn', 'classes');



% labelsn = Normalized labels ranging from 0.2 to 1
% labels0 = original labels
% Xpca_6 = X_vgg_6_SCUT, after normalizing each column (L2 normalization) and after applying PCA and reducing the dimensions to 200
% Xpca_7 = X_vgg_7_SCUT, after normalizing each column (L2 normalization) and after applying PCA and reducing the dimensions to 200











