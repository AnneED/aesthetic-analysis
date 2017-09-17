
% Ridge regression on SCUT-FBP
% labels are normalized
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )


load('initial_data_SCUT_vgg.mat');
X = Xpca_7';
var = devsn.^2;
devs = devsn;
labels = labelsn;

load('results_ridge70.mat');

%k = [0.0001 0.001 0.01 0.1 1 10 50 100 250 500 1000 5000 10000];
% size(k) = 14
% Hay size(k) modelos
% Con esto he visto que lo mejor es 50, ahora concretamos la busqueda:

k = [10 20 30 40 50 60 70 80 90 100];
[~, idx] = min(mean(MAE));




%% 90/10 training/test

AE_RR = [];
kk = idx;

for i = 1:10;
    mask = labeled_masks70(:, i);
    unlabeled = mask == 0;

    b = ridge(labels(mask), X(mask, :), k(kk), 0);
    predicted = X(unlabeled, :) * b(2:end) + b(1);
    test = labels(unlabeled);
    ae = abs(predicted - test);
    AE_RR = [AE_RR ae];        

end


save('results_AE_RR', 'AE_RR');









