% KFME 						
% The Laplacian is based on feature similarity (Gaussian) + score similarity
% labels are normalized (to lie on the interval (0.2, 1))
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )
% A linear transfom is applied to the scores after KFME to adjust the min
% and max values.
% Master thesis: Table 3.8 (Laplacian: Gaussian + score).

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
devs = devsn;
labels = labelsn;
X_n = Xpca_7;


%% 50/50 training/test proportion

epsilon = 0.1; % Remember labels are normalized 0.1 = 0.5/5
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);
EPSILON = zeros(10, 6*5*6*7);

index = 1;
tic;
for i = 1:length(parameters_Beta)
    for j = 1:length(parameters_Gamma)
        for k = 1:length(parameters_Mu)
            for ii = 1:length(parameters_T0)
                Beta = parameters_Beta(i);
                Gamma = parameters_Gamma(j);
                Mu = parameters_Mu(k);
                T0 = parameters_T0(ii);
                parfor l = 1:10 
                    mask = labeled_masks50(:, l);
                    unlabeled = (mask == 0);
                    Y = labels; Y(unlabeled) = 0;
					W = Gauss_GraphConstruction(X_n, epsilon, Y, devs);
					L = diag(sum(W)) - W; 
					[F, Alphas] = KernelFME_Laplacian(X_n, labels, mask, Beta, Gamma, Mu, T0, L); 
                    max_labels = max(Y);
                    min_labels = min(labels(mask)); 
                    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
                    mae = mean(abs(predicted(unlabeled) - labels(unlabeled)));
                    pc = corr(predicted(unlabeled), labels(unlabeled));
                    rmse = sqrt( mean((predicted(unlabeled) - labels(unlabeled)).^2 ));
                    ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                    E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_KFME_50gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');





%% 70/30 training/test proportion

index = 1;
tic;
for i = 1:length(parameters_Beta)
    for j = 1:length(parameters_Gamma)
        for k = 1:length(parameters_Mu)
            for ii = 1:length(parameters_T0)
                Beta = parameters_Beta(i);
                Gamma = parameters_Gamma(j);
                Mu = parameters_Mu(k);
                T0 = parameters_T0(ii);
                parfor l = 1:10
                    mask = labeled_masks70(:, l);
                    unlabeled = (mask == 0);
                    Y = labels; Y(unlabeled) = 0;
					W = Gauss_GraphConstruction(X_n, epsilon, Y, devs);
					L = diag(sum(W)) - W; 
					[F, Alphas] = KernelFME_Laplacian(X_n, labels, mask, Beta, Gamma, Mu, T0, L); 
                    max_labels = max(Y);
                    min_labels = min(labels(mask)); 
                    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
                    mae = mean(abs(predicted(unlabeled) - labels(unlabeled)));
                    pc = corr(predicted(unlabeled), labels(unlabeled));
                    rmse = sqrt( mean((predicted(unlabeled) - labels(unlabeled)).^2 ));
                    ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                    E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_KFME_70gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 90/10 training/test proportion

index = 1;
tic;
for i = 1:length(parameters_Beta)
    for j = 1:length(parameters_Gamma)
        for k = 1:length(parameters_Mu)
            for ii = 1:length(parameters_T0)
                Beta = parameters_Beta(i);
                Gamma = parameters_Gamma(j);
                Mu = parameters_Mu(k);
                T0 = parameters_T0(ii);
                parfor l = 1:10
                    mask = labeled_masks90(:, l);
                    unlabeled = (mask == 0);
                    Y = labels; Y(unlabeled) = 0;
					W = Gauss_GraphConstruction(X_n, epsilon, Y, devs);
					L = diag(sum(W)) - W; 
					[F, Alphas] = KernelFME_Laplacian(X_n, labels, mask, Beta, Gamma, Mu, T0, L); 
                    max_labels = max(Y);
                    min_labels = min(labels(mask)); 
                    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
                    mae = mean(abs(predicted(unlabeled) - labels(unlabeled)));
                    pc = corr(predicted(unlabeled), labels(unlabeled));
                    rmse = sqrt( mean((predicted(unlabeled) - labels(unlabeled)).^2 ));
                    ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                    E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_KFME_90gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');





%% Results

load('results_KFME_50gauss_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0603
% rmse = 0.0784
% pc = 0.8243
% ee = 0.1325



load('results_KFME_70gauss_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0550
% rmse = 0.0729
% pc = 0.8445
% ee = 0.1140



load('results_KFME_90gauss_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0561
% rmse = 0.0710
% pc = 0.8454
% ee = 0.1140


% Parameters of the best model:
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

Beta = pp(idx, 1)
Gamma = pp(idx, 2)
Mu = pp(idx, 3)
T0 = pp(idx, 4)

% Beta = 10000
% Gamma = 50
% Mu = 1
% T0 = 2

for l = 1:10
    mask = labeled_masks90(:, l);
    unlabeled = (mask == 0);
    Y = labels; Y(unlabeled) = 0;
    W = Gauss_GraphConstruction(X_n, epsilon, Y, devs);
    L = diag(sum(W)) - W; 
    [F, Alphas] = KernelFME_Laplacian(X_n, labels, mask, Beta, Gamma, Mu, T0, L); 
    max_labels = max(Y);
    min_labels = min(labels(mask)); 
    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
    mae = mean(abs(predicted(unlabeled) - labels(unlabeled)))
    pc = corr(predicted(unlabeled), labels(unlabeled))
end
