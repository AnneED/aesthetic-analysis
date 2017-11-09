% KFME 						
% The Laplacian is based on feature similarity (Gaussian) + score similarity
% labels are normalized (to lie on the interval (0.2, 1))
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )
% Experiment with different values of epsilon (0.13, 0.17, 0.20) in score
% similarity with 90/10 training/test proportion.
% A linear transfom is applied to the scores after KFME to adjust the min
% and max values.
% Master thesis: Table 3.9.

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
devs = devsn;
labels = labelsn;
X_n = Xpca_7;

% epsilon = [0.13 0.17 0.2]; % Remember labels are normalized /5
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);
EPSILON = zeros(10, 6*5*6*7);


%% 90/10 training/test proportion
% epsilon = 0.13
epsilon = 0.13;
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


save('results_KFME_eps_13.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 90/10 training/test proportion
% epsilon = 0.17
epsilon = 0.17;
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


save('results_KFME_eps_17.mat', 'MAE', 'PC', 'RMSE', 'E');








%% 90/10 training/test proportion
% epsilon = 0.2
epsilon = 0.2;
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


save('results_KFME_eps_2.mat', 'MAE', 'PC', 'RMSE', 'E');







%% Results

% eps = 0.13
load('results_KFME_eps_13.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0559
% rmse = 0.0709
% pc = 0.8453
% ee = 0.1133


% eps = 0.17
load('results_KFME_eps_17.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0557
% rmse = 0.0707
% pc = 0.8458
% ee = 0.1126


% eps = 0.20
load('results_KFME_eps_2.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0556
% rmse = 0.0706
% pc = 0.8462
% ee = 0.1122


