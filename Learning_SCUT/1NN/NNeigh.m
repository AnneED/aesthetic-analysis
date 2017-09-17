% 1-NN
% Normalized labels
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )



% Loading data:
load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
labels = labelsn;
devs = devsn;
X_n = Xpca_7;


%% 50/50 training/test

MAE_euc = [];
MAE_cos = [];
MAE_mink = [];

PC_euc = [];
PC_cos = [];
PC_mink = [];

RMSE_euc = [];
RMSE_cos = [];
RMSE_mink = [];

EPSILON_euc = [];
EPSILON_cos = [];
EPSILON_mink = [];


for i = 1:10
    mask = labeled_masks50(:, i);
    unlabeled = mask == 0;
    test = labels(unlabeled);
    
    % EUCLIDEAN distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'euclidean', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_euc = [MAE_euc mae];
    PC_euc = [PC_euc pc];
    RMSE_euc = [RMSE_euc rmse];
    EPSILON_euc = [EPSILON_euc ee];
    
    
    % MINKOWSKI distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Exponent', 1, 'Distance', 'minkowski', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_mink = [MAE_mink mae];
    PC_mink = [PC_mink pc];
    RMSE_mink = [RMSE_mink rmse];
    EPSILON_mink = [EPSILON_mink ee];
    
    
    % COSINE distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'cosine', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');    
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_cos = [MAE_cos mae];
    PC_cos = [PC_cos pc];
    RMSE_cos = [RMSE_cos rmse];
    EPSILON_cos = [EPSILON_cos ee];
    
    
end

save('results_1NN50_vgg7.mat', 'MAE_euc', 'RMSE_euc', 'PC_euc', 'EPSILON_euc', 'MAE_mink', 'RMSE_mink', 'PC_mink', 'EPSILON_mink', 'MAE_cos', 'RMSE_cos', 'PC_cos', 'EPSILON_cos')




%% 70/30 training/test

MAE_euc = [];
MAE_cos = [];
MAE_mink = [];

PC_euc = [];
PC_cos = [];
PC_mink = [];

RMSE_euc = [];
RMSE_cos = [];
RMSE_mink = [];

EPSILON_euc = [];
EPSILON_cos = [];
EPSILON_mink = [];


for i = 1:10
    mask = labeled_masks70(:, i);
    unlabeled = mask == 0;
    test = labels(unlabeled);
    
    % EUCLIDEAN distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'euclidean', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_euc = [MAE_euc mae];
    PC_euc = [PC_euc pc];
    RMSE_euc = [RMSE_euc rmse];
    EPSILON_euc = [EPSILON_euc ee];
    
    
    % MINKOWSKI distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'minkowski', 'Exponent', 1, 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_mink = [MAE_mink mae];
    PC_mink = [PC_mink pc];
    RMSE_mink = [RMSE_mink rmse];
    EPSILON_mink = [EPSILON_mink ee];
    
    
    % COSINE distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'cosine', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');    
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_cos = [MAE_cos mae];
    PC_cos = [PC_cos pc];
    RMSE_cos = [RMSE_cos rmse];
    EPSILON_cos = [EPSILON_cos ee];
    
    
end

save('results_1NN70_vgg7.mat', 'MAE_euc', 'RMSE_euc', 'PC_euc', 'EPSILON_euc', 'MAE_mink', 'RMSE_mink', 'PC_mink', 'EPSILON_mink', 'MAE_cos', 'RMSE_cos', 'PC_cos', 'EPSILON_cos')





%% 90/10 training/test

MAE_euc = [];
MAE_cos = [];
MAE_mink = [];

PC_euc = [];
PC_cos = [];
PC_mink = [];

RMSE_euc = [];
RMSE_cos = [];
RMSE_mink = [];

EPSILON_euc = [];
EPSILON_cos = [];
EPSILON_mink = [];


for i = 1:10
    mask = labeled_masks90(:, i);
    unlabeled = mask == 0;
    test = labels(unlabeled);
    
    % EUCLIDEAN distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'euclidean', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_euc = [MAE_euc mae];
    PC_euc = [PC_euc pc];
    RMSE_euc = [RMSE_euc rmse];
    EPSILON_euc = [EPSILON_euc ee];
    
    
    % MINKOWSKI p = 1 distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'minkowski', 'Exponent', 1, 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_mink = [MAE_mink mae];
    PC_mink = [PC_mink pc];
    RMSE_mink = [RMSE_mink rmse];
    EPSILON_mink = [EPSILON_mink ee];
    
    
    % COSINE distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'cosine', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');    
    
    % Error in prediction:
    mae = mean(abs(predicted-test));
    pc = corr(predicted, test);
    rmse = sqrt( mean((predicted - test).^2 ));
    ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
    
    MAE_cos = [MAE_cos mae];
    PC_cos = [PC_cos pc];
    RMSE_cos = [RMSE_cos rmse];
    EPSILON_cos = [EPSILON_cos ee];
    
    
end


save('results_1NN90_vgg7.mat', 'MAE_euc', 'RMSE_euc', 'PC_euc', 'EPSILON_euc', 'MAE_mink', 'RMSE_mink', 'PC_mink', 'EPSILON_mink', 'MAE_cos', 'RMSE_cos', 'PC_cos', 'EPSILON_cos')










