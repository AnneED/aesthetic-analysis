
% Results on RIDGE REGRESSION



%% 50/50 training/test proportion
load('results_ridge50.mat');

% Model with best mae:
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(EPSILON(:, idx))
% mae = 0.0814
% rmse = 0.1086
% pc = 0.5969
% ee = 0.1984





%% 70/30 training/test proportion
clear;
load('results_ridge70.mat');


% Model with best mae:
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(EPSILON(:, idx))

mean(MAE)
% mae = 0.0727
% rmse = 0.0962
% pc = 0.6867
% ee = 0.1708





%% 90/10 training/test proportion
clear;
load('results_ridge90.mat');

MAE(mean)

% Model with best mae:
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(EPSILON(:, idx))

% mae = 0.0645
% rmse = 0.0827
% pc = 0.7772
% ee = 0.1429









