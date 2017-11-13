% KFME 						
% Experiment with KFME and different values of delta (epsilon) using a
% 50/50 training/test partition in the Eastern images of M2B.
% Values of epsilon = 0.005, 0.03, 0.05, 0.08, 0.13, 0.17, 0.2.
% The Laplacian is based on feature similarity (Gaussian) + score
% similarity.
% Labels are normalized (to lie on the interval (0.1, 1)).
% Features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200
% dimensions) ).
% A linear transfom is applied to the scores after KFME to adjust the min
% and max values.
% Master thesis: Table 3.22 (Laplacian: Gaussian).

load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;


parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);





%% Epsilon = 0.005

epsilon = 0.005; %
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_005.mat', 'MAE', 'PC', 'RMSE');




%% Epsilon = 0.03

epsilon = 0.03; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_03.mat', 'MAE', 'PC', 'RMSE');





%% Epsilon = 0.05

epsilon = 0.05; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_05.mat', 'MAE', 'PC', 'RMSE');




%% Epsilon = 0.08

epsilon = 0.08; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_08.mat', 'MAE', 'PC', 'RMSE');






%% Epsilon = 0.13

epsilon = 0.13; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_13.mat', 'MAE', 'PC', 'RMSE');



%% Epsilon = 0.17

epsilon = 0.17; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_17.mat', 'MAE', 'PC', 'RMSE');




%% Epsilon = 0.2

epsilon = 0.2; % 
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
                    mask = labeled_masks50_e(:, l);
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
                   % ee = mean(1 - exp(- (predicted(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
                    MAE(l, index) = mae;
                    PC(l, index) = pc;
                    RMSE(l, index) = rmse;
                  %  E(l, index) = ee;
                end
                index = index + 1;
            end
        end
    end
end
toc;


save('results_eKFME_epsilon_20.mat', 'MAE', 'PC', 'RMSE');




%% Results


load('results_eKFME_epsilon_005.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1358
% rmse = 0.1668
% pc = 0.4503


load('results_eKFME_epsilon_03.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1358
% rmse = 0.1671
% pc = 0.4457


load('results_eKFME_epsilon_05.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1358
% rmse = 0.1670
% pc = 0.4461



load('results_eKFME_epsilon_08.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1357
% rmse = 0.1670
% pc = 0.4466



load('results_eKFME_epsilon_13.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1357
% rmse = 0.1670
% pc = 0.4466



load('results_eKFME_epsilon_17.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1357
% rmse = 0.1670
% pc = 0.4461



load('results_eKFME_epsilon_20.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1357
% rmse = 0.1670
% pc = 0.4462








