


% Obtaining the labels:
addpath('/home/john-san/Dropbox/Master/TFM/Data/M2B');
load('ranks.mat');

% original labels [1, 10]
labels0_w = rank_W100;
labels0_e = rank_E100;

% scaled labels [0.1, 1]
labelsn_w = rank_W100/ max(rank_W100);
labelsn_e = rank_E100 / max(rank_E100);


% Creation of 5 classes:
classes_w = zeros(length(labels0_w), 1);
for i = 1:length(labels0_w)
    if labels0_w(i) < 3
        classes_w(i) = 1;
    elseif labels0_w(i) < 5
        classes_w(i) = 2;
    elseif labels0_w(i) < 7
        classes_w(i) = 3;
    elseif labels0_w(i) < 8
        classes_w(i) = 4;
    else
        classes_w(i) = 5;
    end
end


% Creation of 5 classes:
classes_e = zeros(length(labels0_e), 1);
for i = 1:length(labels0_e)
    if labels0_e(i) < 3
        classes_e(i) = 1;
    elseif labels0_e(i) < 5
        classes_e(i) = 2;
    elseif labels0_e(i) < 7
        classes_e(i) = 3;
    elseif labels0_e(i) < 8
        classes_e(i) = 4;
    else
        classes_e(i) = 5;
    end
end

%figure, hist(classes_e)
%figure, hist(classes_w)


% Creating training/test partitions with stratification (mantaining the
% original class distribution)
labeled_masks50_e = [];
labeled_masks50_w = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes_e, 'HoldOut', 0.5);
    labeled_masks50_e = [labeled_masks50_e partition.training];
    partition = cvpartition(classes_w, 'HoldOut', 0.5);
    labeled_masks50_w = [labeled_masks50_w partition.training];
end

labeled_masks70_e = [];
labeled_masks70_w = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes_e, 'HoldOut', 0.3);
    labeled_masks70_e = [labeled_masks70_e partition.training];
    partition = cvpartition(classes_w, 'HoldOut', 0.3);
    labeled_masks70_w = [labeled_masks70_w partition.training];
end

labeled_masks90_e = [];
labeled_masks90_w = [];
for i = 1:10
    % Obtaining the training and the test sets:
    partition = cvpartition(classes_e, 'HoldOut', 0.1);
    labeled_masks90_e = [labeled_masks90_e partition.training];
    partition = cvpartition(classes_w, 'HoldOut', 0.1);
    labeled_masks90_w = [labeled_masks90_w partition.training];
end


labeled_masks50_e = logical(labeled_masks50_e);
labeled_masks70_e = logical(labeled_masks70_e);
labeled_masks90_e = logical(labeled_masks90_e);

labeled_masks50_w = logical(labeled_masks50_w);
labeled_masks70_w = logical(labeled_masks70_w);
labeled_masks90_w = logical(labeled_masks90_w);



% Loading the data matrices and preprocessing
load('matrices_vgg.mat');
X = X_vgg_6_M2Be;
XX = X_vgg_7_M2Be;

% L2 normalization:
for kk = 1:size(X, 2)
X(:, kk) = X_vgg_6_M2Be(:, kk) / norm(X_vgg_6_M2Be(:, kk));
XX(:, kk) = X_vgg_7_M2Be(:, kk) / norm(X_vgg_7_M2Be(:, kk));
end

% pca:
[~, SCORE] = pca(X');
Xpca_6e = (SCORE(:, 1:200) )';

[~, SCORE] = pca(XX');
Xpca_7e = (SCORE(:, 1:200) )';

[~, ~, ~, ~, explained, ~] = pca(XX');
sum(explained(1:200))
% 82.6202

X = X_vgg_6_M2Bw;
XX = X_vgg_7_M2Bw;

% L2 normalization:
for kk = 1:size(X, 2)
X(:, kk) = X_vgg_6_M2Bw(:, kk) / norm(X_vgg_6_M2Bw(:, kk));
XX(:, kk) = X_vgg_7_M2Bw(:, kk) / norm(X_vgg_7_M2Bw(:, kk));
end

% pca:
[~, SCORE] = pca(X');
Xpca_6w = (SCORE(:, 1:200) )';

[~, SCORE] = pca(XX');
Xpca_7w = (SCORE(:, 1:200) )';

[~, ~, ~, ~, explained, ~] = pca(XX');
sum(explained(1:200))
% 74.8169


save('initial_data_M2Be_vgg', 'X_vgg_6_M2Be', 'X_vgg_7_M2Be', 'Xpca_6e', 'Xpca_7e', 'labeled_masks50_e', 'labeled_masks70_e', 'labeled_masks90_e', 'labels0_e', 'labelsn_e', 'classes_e');
save('initial_data_M2Bw_vgg', 'X_vgg_6_M2Bw', 'X_vgg_7_M2Bw', 'Xpca_6w', 'Xpca_7w', 'labeled_masks50_w', 'labeled_masks70_w', 'labeled_masks90_w', 'labels0_w', 'labelsn_w', 'classes_w');



% labelsn = Normalized labels ranging from 0.1 to 1
% labels0 = original labels
% Xpca_6 = X_vgg_6_M2B, after normalizing each column (L2 normalization) and after applying PCA and reducing the dimensions to 200
% Xpca_7 = X_vgg_7_M2B, after normalizing each column (L2 normalization) and after applying PCA and reducing the dimensions to 200











