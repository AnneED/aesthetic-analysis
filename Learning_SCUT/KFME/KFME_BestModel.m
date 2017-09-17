
% KFME 						
% The Laplacian is based on feature similarity (gaussian) + score similarity
% labels are normalized (to lie on the interval (0.2, 1))
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
devs = devsn;
labels = labelsn;
X_n = Xpca_7;



parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

load('results_KFME_70gauss_vgg7.mat')

pp = [];
for i = 1:length(parameters_Beta)
    for j = 1:length(parameters_Gamma)
        for k = 1:length(parameters_Mu)
            for ii = 1:length(parameters_T0)
                Beta = parameters_Beta(i);
                Gamma = parameters_Gamma(j);
                Mu = parameters_Mu(k);
                T0 = parameters_T0(ii);
                pp = [pp; Beta Gamma Mu T0];
            end
        end
    end
end

[~, idx] = min(mean(MAE));
Beta = pp(idx, 1);
Gamma = pp(idx, 2);
Mu = pp(idx, 3);
T0 = pp(idx, 4);

AE_KFME = [];

epsilon = 0.1;

for l = 1:10
    mask = labeled_masks70(:, l);
    unlabeled = (mask == 0);
    Y = labels; Y(unlabeled) = 0;
    W = Gauss_GraphConstruction(X_n, epsilon, Y, devs);
    L = diag(sum(W)) - W; 
    [F, Alphas] = KernelFME_Laplacian(X_n, labels, mask, Beta, Gamma, Mu, T0, L); 
    max_labels = max(Y);
    min_labels = min(labels(mask)); 
    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
    ae = abs(predicted(unlabeled) - labels(unlabeled));
    AE_KFME = [AE_KFME ae];
end

save('results_AE_KFME', 'AE_KFME');




