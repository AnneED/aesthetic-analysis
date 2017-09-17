
% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels


%% eastern

% Loading data:
load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;

% libsvm:
addpath /home/john-san/workspace/libsvm-3.22/matlab

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
var = devs.^2;
dims = 10:10:length(labels);

% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];
% length(parameters_p) =       ii
% length(parameters_c) =      k
npar = length(parameters_c) * length(parameters_p);





l = length(labels) * 0.5;
u = l;

load('results_optimization_Zwb_eM2B');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

pp = [];
for i = 1:7
    for j = 1:7
        for k = 1:7
            pp = [pp; parameters(i) parameters(j) parameters(k)];
        end
    end
end

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);


MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks50_e(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes_e(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
            end
            index = index + 1;
            ii
            k
        end
end

save('results_Zbw_linearSvr_eM2B', 'MAE', 'PC', 'RMSE');



%% western

clear;


% Loading data:
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_w;
X_n = Xpca_7w;

% libsvm:
addpath /home/john-san/workspace/libsvm-3.22/matlab

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
var = devs.^2;
dims = 10:10:length(labels);

% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];
% length(parameters_p) =       ii
% length(parameters_c) =      k
npar = length(parameters_c) * length(parameters_p);


l = length(labels) * 0.5;
u = l;

load('results_optimization_Zwb_wM2B');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

pp = [];
for i = 1:7
    for j = 1:7
        for k = 1:7
            pp = [pp; parameters(i) parameters(j) parameters(k)];
        end
    end
end

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);


MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks50_w(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes_w(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
            end
            index = index + 1;
            ii
            k
        end
end

save('results_Zbw_linearSvr_wM2B', 'MAE', 'PC', 'RMSE');








%% Results

% eastern

load('results_Zbw_linearSvr_eM2B')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1350
% rmse = 0.1672
% pc = 0.4460


% western

load('results_Zbw_linearSvr_wM2B')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1138
% rmse = 0.1421
% pc = 0.6349






