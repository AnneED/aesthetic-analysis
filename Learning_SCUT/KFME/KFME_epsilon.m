% KFME 						
% The Laplacian is based on feature similarity (gaussian) + score similarity
% labels are normalized (to lie on the interval (0.2, 1))
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )
% Experiment with different values of epsilon (0.005, 0.03, 0.05, 0.08) in
% score similarity with 90/10 training/test proportion.
% Master thesis: Table 3.9.

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
devs = devsn;
labels = labelsn;
X_n = Xpca_7;



% Remember labels are normalized /5
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);
EPSILON = zeros(10, 6*5*6*7);


%% 90/10 training/test proportion
% epsilon = 0.005
epsilon = 0.005;
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


save('results_KFME_eps_5.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 90/10 training/test proportion
% epsilon = 0.03
epsilon = 0.03;
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


save('results_KFME_eps_3.mat', 'MAE', 'PC', 'RMSE', 'E');








%% 90/10 training/test proportion
% epsilon = 0.08
epsilon = 0.08;
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


save('results_KFME_eps_8.mat', 'MAE', 'PC', 'RMSE', 'E');







%% Results

% eps = 0.005
load('results_KFME_eps_5.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0575
% rmse = 0.0740
% pc = 0.8385
% ee = 0.1241


% eps = 0.03
load('results_KFME_eps_3.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0569
% rmse = 0.0717
% pc = 0.8475
% ee = 0.1173


% eps = 0.08
load('results_KFME_eps_8.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0564
% rmse = 0.0713
% pc = 0.8451
% ee = 0.1148


%% Check with epsilon = 0.05

% 90/10 training/test proportion
% epsilon = 0.05
epsilon = 0.05;
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


save('results_KFME_eps_005.mat', 'MAE', 'PC', 'RMSE', 'E');


% eps = 0.05
load('results_KFME_eps_005.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0568
% rmse = 0.0716
% pc = 0.8454
% ee = 0.1163









