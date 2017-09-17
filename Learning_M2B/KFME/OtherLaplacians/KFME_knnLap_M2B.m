
% KFME 						
% The Laplacian is based on feature similarity (gaussian) 
% labels are normalized (to lie on the interval (0.1, 1))
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )

load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;




epsilon = 0.1; % 
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);




%% Eastern

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
                    [F, ~] = KernelFME_Fadi2(X_n, labels, mask, Beta, Gamma, Mu, T0);
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


save('results_eKFME_50knn_vgg7.mat', 'MAE', 'PC', 'RMSE');



%% Western
clear;
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_w;
X_n = Xpca_7w;



epsilon = 0.1; % 
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);


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
                    mask = labeled_masks50_w(:, l);
                    unlabeled = (mask == 0);
                    Y = labels; Y(unlabeled) = 0;
                    [F, ~] = KernelFME_Fadi2(X_n, labels, mask, Beta, Gamma, Mu, T0); 					 
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


save('results_wKFME_50knn_vgg7.mat', 'MAE', 'PC', 'RMSE');




%% Both

clear;
load('initial_data_M2Be_vgg.mat');
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = [labelsn_e; labelsn_w];
X_n = [Xpca_7e Xpca_7w];



epsilon = 0.1; % 
parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

MAE = zeros(10, 6*5*6*7);
PC = zeros(10, 6*5*6*7);
RMSE = zeros(10, 6*5*6*7);



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
                    mask = [labeled_masks50_e(:, l); labeled_masks50_w(:, l)];
                    unlabeled = (mask == 0);
                    Y = labels; Y(unlabeled) = 0;
                    [F, ~] = KernelFME_Fadi2(X_n, labels, mask, Beta, Gamma, Mu, T0);
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


save('results_bKFME_50knn_vgg7.mat', 'MAE', 'PC', 'RMSE');











%% Results

% Eastern
load('results_eKFME_50knn_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.0697
% rmse = 0.1177
% pc = 0.7757


% Western
load('results_wKFME_50knn_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.0747
% rmse = 0.1110
% pc = 0.8014


% Both
load('results_bKFME_50knn_vgg7.mat')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.0744
% rmse = 0.1265
% pc = 0.7306









