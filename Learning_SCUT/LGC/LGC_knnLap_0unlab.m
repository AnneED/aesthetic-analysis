
% LGC 
% The similarity matrix is based on gaussian feature similarity
% labels are normalized
% features: vgg layer 7 is used (preprocessing: L2 normalization + pca (200 dimensions) ) 

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
labels = labelsn;
X_n = Xpca_7;

% Building the similarity matrix (K = 10):
addpath('/home/john-san/Dropbox/Master/TFM/Codes_drive');
[~, W]= KNN_GraphConstruction(X_n, 10);


%% 50/50 training/test proportion

parameters = [10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 0.1 1 10 100 1000 10^4 10^5 10^6];
MAE = zeros(10, length(parameters));
PC = zeros(10, length(parameters));
RMSE = zeros(10, length(parameters));
E = zeros(10, length(parameters));

for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks50(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = 0;
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        E(j, i) = ee;
        
    end
end


save('results_LGC_knn50_vgg7_0unlab.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 70/30 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks70(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = 0;
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        E(j, i) = ee;
        
    end
end


save('results_LGC_knn70_vgg7_0unlab.mat', 'MAE', 'PC', 'RMSE', 'E');





%% 90/10 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks90(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = 0;
        F = LGC(X_n, W, Mu, Y);
        predicted = F(unlabeled);
        test = labels(unlabeled);
        mae = mean(abs(predicted-test) );
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        MAE(j, i) = mae;
        PC(j, i) = pc;
        RMSE(j, i) = rmse;
        E(j, i) = ee;
        
    end
end


save('results_LGC_knn90_vgg7_0unlab.mat', 'MAE', 'PC', 'RMSE', 'E');




%% Results

load('results_LGC_knn50_vgg7_0unlab.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.2711
% rmse = 0.3032
% pc = 0.1002
% ee = 0.7395




load('results_LGC_knn70_vgg7_0unlab.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.1711
% rmse = 0.2137
% pc = 0.0976
% ee = 0.4388




load('results_LGC_knn90_vgg7_0unlab.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.1100
% rmse 0 0.1497
% pc = 0.1153
% ee = 0.2635





























