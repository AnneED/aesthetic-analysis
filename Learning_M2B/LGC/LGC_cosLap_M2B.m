
% LGC 
% The similarity matrix is based on cosine feature similarity + score similarity
% labels are normalized
% features: vgg layer 7 is used (preprocessing: L2 normalization + pca (200 dimensions) ) 

%% eastern 

clear;

load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X_n = Xpca_7e;
 

epsilon = 0.1;


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
        Y(unlabeled) = 0;
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs, 10);
        W = (W + W')/2;
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


save('results_LGC_cos_eM2B.mat', 'MAE', 'PC', 'RMSE');






%% western 

clear;

load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_w;
X_n = Xpca_7w;


epsilon = 0.1;


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
        Y(unlabeled) = 0;
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs, 10);
        W = (W + W')/2;
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


save('results_LGC_cos_wM2B.mat', 'MAE', 'PC', 'RMSE');




%% both 

clear;

clear;
load('initial_data_M2Be_vgg.mat');
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = [labelsn_e; labelsn_w];
X_n = [Xpca_7e Xpca_7w];

epsilon = 0.1;


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
        Y(unlabeled) = 0;
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs, 10);
        W = (W + W')/2;
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


save('results_LGC_cos_bM2B.mat', 'MAE', 'PC', 'RMSE');





%% Results


% Eastern

load('results_LGC_cos_eM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1501
% rmse = 0.1837
% pc = 0.2006


% Western

load('results_LGC_cos_wM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1423
% rmse = 0.1741
% pc = 0.3433


% Both

load('results_LGC_cos_bM2B.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1483
% rmse = 0.1813
% pc = 0.2169



























