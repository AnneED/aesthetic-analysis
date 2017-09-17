% RESULTS 1-NN SCUT-FBP

%% 50/50
load('results_1NN50_vgg7.mat')

% EUCLIDEAN

mae = mean(MAE_euc)
rmse = mean(RMSE_euc)
pc = mean(PC_euc)
ee = mean(EPSILON_euc)
% mae = 0.0891
% rmse = 0.1176
% pc = 0.5924
% ee = 0.2229


% MINKOWSKI

mae = mean(MAE_mink)
rmse = mean(RMSE_mink)
pc = mean(PC_mink)
ee = mean(EPSILON_mink)
% mae = 0.0976
% rmse = 0.1305
% pc = 0.4450
% ee = 0.2482





% COSINE

mae = mean(MAE_cos)
rmse = mean(RMSE_cos)
pc = mean(PC_cos)
ee = mean(EPSILON_cos)
% mae = 0.0902
% rmse = 0.1200
% pc = 0.6120
% ee = 0.2228






%% 70/30
load('results_1NN70_vgg7.mat')

% EUCLIDEAN

mae = mean(MAE_euc)
rmse = mean(RMSE_euc)
pc = mean(PC_euc)
ee = mean(EPSILON_euc)
% mae = 0.0876
% rmse = 0.1168
% pc = 0.6052
% ee = 0.2148



% MINKOWSKI

mae = mean(MAE_mink)
rmse = mean(RMSE_mink)
pc = mean(PC_mink)
ee = mean(EPSILON_mink)
% mae = 0.0934
% rmse = 0.1257
% pc = 0.4930
%ee = 0.2331


% COSINE

mae = mean(MAE_cos)
rmse = mean(RMSE_cos)
pc = mean(PC_cos)
ee = mean(EPSILON_cos)
% mae = 0.0885
% rmse = 0.1176
% pc = 0.6375
% ee = 0.2153






%% 90/10
load('results_1NN90_vgg7.mat')

% EUCLIDEAN

mae = mean(MAE_euc)
rmse = mean(RMSE_euc)
pc = mean(PC_euc)
ee = mean(EPSILON_euc)
% mae = 0.0909
% rmse = 0.1175
% pc = 0.5668
% ee = 0.2310




% MINKOWSKI

mae = mean(MAE_mink)
rmse = mean(RMSE_mink)
pc = mean(PC_mink)
ee = mean(EPSILON_mink)
% mae = 0.0939
% rmse = 0.1220
% pc = 0.5136
% ee = 0.2366




% COSINE

mae = mean(MAE_cos)
rmse = mean(RMSE_cos)
pc = mean(PC_cos)
ee = mean(EPSILON_cos)
% mae = 0.0891
% rmse = 0.1139
% pc = 0.6377
% ee = 0.2208












