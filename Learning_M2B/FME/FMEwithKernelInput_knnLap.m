
% FME with Kernel input (do not confuse with KFME)
% The Laplacian is based on gaussian feature similarity (with 10 neighbors)
% Layer 7 is used (preprocessing: L2 normalization + PCA)
% Normalized labels


load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
L = double(diag(sum(W_ind)) - W_ind);


%% Eastern

para.uu = 0;
parameters = [10^(-9) 10^(-6) 10^(-3) 1 10^3 10^6 10^9];

MAE = zeros(10, length(parameters)^3);
PC = zeros(10, length(parameters)^3);
RMSE = zeros(10, length(parameters)^3);
EPSILON = zeros(10, length(parameters)^3);
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];


tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           for ii = 1:length(parameters_T0)
               para.ul = parameters(i);    % beta in the paper
               para.mu =  parameters(j);
               para.lamda = parameters(k);          % gamma in the paper
               T0 = parameters_T0(ii);
               % Calculate kernel
               Ker = KernelMatrix( X_n, X_n, T0 );
               parfor l = 1:10
                   mask = labeled_masks50_e(:, l);
                   unlabeled = (mask == 0);
                   T = labels;
                   T(unlabeled) = 0;
                   % calculate FME on kernel
                   [W, b, F] = FME_semi2( Ker, L, T, para );
                   mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
                   pc = corr(labels(unlabeled), F(unlabeled));
                   rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
                   MAE(l, index) = mae;
                   PC(l, index) = pc;
                   RMSE(l, index) = rmse;
               end
           end
           index = index + 1;
       end
   end
end
toc;


save('results_FME_withKernelInput_knn_eM2B.mat', 'MAE', 'PC', 'RMSE');





%% Western

clear;

load('initial_data_M2Bw_vgg.mat');
devs = 0;
labels = labelsn_w;
X_n = Xpca_7w;


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
L = double(diag(sum(W_ind)) - W_ind);


para.uu = 0;
parameters = [10^(-9) 10^(-6) 10^(-3) 1 10^3 10^6 10^9];

MAE = zeros(10, length(parameters)^3);
PC = zeros(10, length(parameters)^3);
RMSE = zeros(10, length(parameters)^3);
EPSILON = zeros(10, length(parameters)^3);


tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           for ii = 1:length(parameters_T0)
               para.ul = parameters(i);    % beta in the paper
               para.mu =  parameters(j);
               para.lamda = parameters(k);          % gamma in the paper
               T0 = parameters_T0(ii);
               % Calculate kernel
               Ker = KernelMatrix( X_n, X_n, T0 );   % gamma in the paper
               parfor l = 1:10
                   mask = labeled_masks50_w(:, l);
                   unlabeled = (mask == 0);
                   T = labels;
                   T(unlabeled) = 0;
                   [W, b, F] = FME_semi2( Ker, L, T, para );
                   mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
                   pc = corr(labels(unlabeled), F(unlabeled));
                   rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
                   MAE(l, index) = mae;
                   PC(l, index) = pc;
                   RMSE(l, index) = rmse;
               end
           end
           index = index + 1;
       end
   end
end
toc;


save('results_FME_knn_withKernelInput_wM2B.mat', 'MAE', 'PC', 'RMSE');



%% Both



clear;
load('initial_data_M2Be_vgg.mat');
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = [labelsn_e; labelsn_w];
X_n = [Xpca_7e Xpca_7w];


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
L = double(diag(sum(W_ind)) - W_ind);


para.uu = 0;
parameters = [10^(-9) 10^(-6) 10^(-3) 1 10^3 10^6 10^9];

MAE = zeros(10, length(parameters)^3);
PC = zeros(10, length(parameters)^3);
RMSE = zeros(10, length(parameters)^3);
EPSILON = zeros(10, length(parameters)^3);


tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           for ii = 1:length(parameters_T0)
               para.ul = parameters(i);    % beta in the paper
               para.mu =  parameters(j);
               para.lamda = parameters(k);          % gamma in the paper
               T0 = parameters_T0(ii);
               % Calculate kernel
               Ker = KernelMatrix( X_n, X_n, T0 );       % gamma in the paper
               parfor l = 1:10
                   mask = [labeled_masks50_e(:, l); labeled_masks50_w(:, l)];
                   unlabeled = (mask == 0);
                   T = labels;
                   T(unlabeled) = 0;
                   [W, b, F] = FME_semi2( Ker, L, T, para );
                   mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
                   pc = corr(labels(unlabeled), F(unlabeled));
                   rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
                   MAE(l, index) = mae;
                   PC(l, index) = pc;
                   RMSE(l, index) = rmse;
               end
           end
           index = index + 1;
       end
   end
end
toc;


save('results_FME_knn_withKernelInput_bM2B.mat', 'MAE', 'PC', 'RMSE');




%% Results


% Eastern

load('results_FME_knn_withKernelInput_eM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))



% Western

load('results_FME_knn_withKernelInput_wM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))



% Both

load('results_FME_knn_withKernelInput_bM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
