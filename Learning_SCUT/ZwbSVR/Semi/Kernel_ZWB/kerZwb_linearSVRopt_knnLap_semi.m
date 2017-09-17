
% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize MAE

% Loading data:
load('initial_data_SCUT_vgg.mat');
X200 = Xpca_7;
devs = devsn;
labels = labelsn;

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];
var = devs.^2;
dims = 10:10:500;


%% Optimization of linear SVR:

% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize PC

X_n = Xpca_7;

% libsvm:
addpath /home/john-san/workspace/libsvm-3.22/matlab

% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];% length(parameters_p) =       ii
% length(parameters_c) =      k
npar = length(parameters_c) * length(parameters_p);

pp = [];
for jj = 1:length(parameters_T0)
    for i = 1:7
        for j = 1:7
            for k = 1:7
                pp = [pp; parameters(i) parameters(j) parameters(k) parameters_T0(jj)];
            end
        end
    end
end



%% 50/50 training/test proportion


l = 250;
u = 250;

load('results_optimization_kerZwb_vgg7_knn50');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);

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
                mask = labeled_masks50(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                % The first columns correspond to labeled instances and the last ones to unlabeled ones
                X = zeros(l+u, l+u);
                X(1:l, 1:l) = Ker(mask, mask);
                X(l+1:l+u, l+1:l+u) = Ker(unlabeled, unlabeled);
                X(1:l, l+1:l+u) = Ker(mask, unlabeled);
                X(l+1:l+u, 1:l) = Ker(unlabeled, mask);


                % Building the Laplacian matrix (K = 10): 
                XX = [X200(:, mask) X200(:, unlabeled)];
                [~, W]= KNN_GraphConstruction(XX, 10);
                L = double(diag(sum(W)) - W) ; 

                % non-linear projection
                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end

save('results_kerZbw_linearSvr_vgg7_knn50', 'MAE', 'PC', 'RMSE', 'E');

disp('50 finished');








%% 70/30 training/test proportion


l = 350;
u = 150;

load('results_optimization_kerZwb_vgg7_knn70');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);

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
                mask = labeled_masks70(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end





save('results_kerZbw_linearSvr_vgg7_knn70', 'MAE', 'PC', 'RMSE', 'E');



disp('70 finished');







%% 90/10 training/test proportion


l = 450;
u = 50;

load('results_optimization_kerZwb_vgg7_knn90');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);


Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);


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
                mask = labeled_masks90(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end



save('results_kerZbw_linearSvr_vgg7_knn90', 'MAE', 'PC', 'RMSE', 'E');



disp('90 finished');




%% Results

para_SVR = [];
for i = 1:length(parameters_p)
    for j = 1:length(parameters_c)
        para_SVR = [para_SVR; parameters_p(i) parameters_c(j)];
    end
end



load('results_kerZbw_linearSvr_vgg7_knn50')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))

p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

% mae = 0.0601
% rmse = 0.0785
% pc = 0.8153
% ee = 0.1293

load('results_optimization_kerZwb_vgg7_knn50');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)

% dim = 170
% Alpha = 1000
% Gamma = 1000
% Mu = 1
% T0 = 4




load('results_kerZbw_linearSvr_vgg7_knn70')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0563
% rmse = 0.0748
% pc = 0.8368
% ee = 0.1217


p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

load('results_optimization_kerZwb_vgg7_knn70');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)
% dim = 210
% Alpha = 1e-06
% Gamma = 1
% Mu = 1000000
% T0 = 0.1250



load('results_kerZbw_linearSvr_vgg7_knn90')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0680
% rmse = 0.0887
% pc = 0.7305
% ee = 0.1592


p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

load('results_optimization_kerZwb_vgg7_knn90');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)

% dim = 70
% Alpha = 1
% Gamma = 1000000
% Mu = 1e^-3
% T0 = 8



















