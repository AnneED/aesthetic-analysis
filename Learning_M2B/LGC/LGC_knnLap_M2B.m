
% LGC 
% The similarity matrix is based on gaussian feature similarity
% labels are normalized
% features: vgg layer 7 is used (preprocessing: L2 normalization + pca (200 dimensions) ) 

%% eastern 

clear;

load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
W = (W_ind + W_ind')/2;




parameters = [10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 0.1 1 10 100 1000 10^4 10^5 10^6];
MAE = zeros(10, length(parameters));
PC = zeros(10, length(parameters));
RMSE = zeros(10, length(parameters));

for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks50_e(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = mean(labels(mask));
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        
    end
end


save('results_LGC_knn_eM2B.mat', 'MAE', 'PC', 'RMSE');






%% western 

clear;

load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_w;
X_n = Xpca_7w;


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
W = (W_ind + W_ind')/2;
L = double(diag(sum(W_ind)) - W_ind) ;   




parameters = [10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 0.1 1 10 100 1000 10^4 10^5 10^6];
MAE = zeros(10, length(parameters));
PC = zeros(10, length(parameters));
RMSE = zeros(10, length(parameters));

for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks50_w(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = mean(labels(mask));
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        
    end
end


save('results_LGC_knn_wM2B.mat', 'MAE', 'PC', 'RMSE');




%% both 

clear;

clear;
load('initial_data_M2Be_vgg.mat');
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = [labelsn_e; labelsn_w];
X_n = [Xpca_7e Xpca_7w];


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
W = (W_ind + W_ind')/2;
L = double(diag(sum(W_ind)) - W_ind) ;   



parameters = [10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 0.1 1 10 100 1000 10^4 10^5 10^6];
MAE = zeros(10, length(parameters));
PC = zeros(10, length(parameters));
RMSE = zeros(10, length(parameters));

for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = [labeled_masks50_e(:, j); labeled_masks50_w(:, j)];
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = mean(labels(mask));
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        
    end
end


save('results_LGC_knn_bM2B.mat', 'MAE', 'PC', 'RMSE');





%% Results


% Eastern

load('results_LGC_knn_eM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1502
% rmse = 0.1827
% pc = 0.2051


% Western

load('results_LGC_knn_wM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1438
% rmse = 0.1742
% pc = 0.3499


% Both

load('results_LGC_knn_bM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1484
% rmse = 0.1801
% pc = 0.2290



























