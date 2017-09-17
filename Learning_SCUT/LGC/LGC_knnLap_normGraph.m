
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

% Normalize the graph:

% Divide each row by its maximum, so the maximum value of each row is 1:
for i = 1:size(W,1)
W(i, :) = W(i, :) / sum(W(i, :));
end

% Make the graph symmetric:
W = (W + W')/2;




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
        Y(unlabeled) = mean(labels(mask));
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


save('results_LGC_normGraph50.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 70/30 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks70(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = mean(labels(mask));
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


save('results_LGC_normGraph70.mat', 'MAE', 'PC', 'RMSE', 'E');





%% 90/10 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks90(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = mean(labels(mask));
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


save('results_LGC_normGraph90.mat', 'MAE', 'PC', 'RMSE', 'E');




%% Results

load('results_LGC_normGraph50.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0928
% rmse = 0.1217
% pc = 0.5952
% ee = 0.2375



load('results_LGC_normGraph70.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0861
% rmse = 0.1132
% pc = 0.6905
% ee = 0.2169




load('results_LGC_normGraph90.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0830
% rmse = 0.1076
% pc = 0.7390
% ee = 0.2090





























