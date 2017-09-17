
% Ridge regression on SCUT-FBP
% labels are normalized
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca(200 dimensions) )


load('initial_data_SCUT_vgg.mat');
X = Xpca_7';
var = devsn.^2;
devs = devsn;
labels = labelsn;


%k = [0.0001 0.001 0.01 0.1 1 10 50 100 250 500 1000 5000 10000];
% size(k) = 14
% Hay size(k) modelos
% Con esto he visto que lo mejor es 50, ahora concretamos la busqueda:

k = [10 20 30 40 50 60 70 80 90 100];


%% 50/50 training/test

MAE2 = zeros(10, length(k));
MAE = zeros(10, length(k));
PC = zeros(10, length(k));
RMSE = zeros(10, length(k));
EPSILON = zeros(10, length(k));

for kk = 1:length(k)
    for i = 1:10;
        mask = labeled_masks50(:, i);
        unlabeled = mask == 0;

        b = ridge(labels(mask), X(mask, :), k(kk), 0);
        predicted = X(unlabeled, :) * b(2:end) + b(1);
        test = labels(unlabeled);
        mae = mean(abs(predicted - test));
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        
        MAE(i, kk) = mae;
        RMSE(i, kk) = rmse;
        PC(i, kk) = pc;
        EPSILON(i, kk) = ee;
    end
end

mean(MAE)

save('results_ridge50.mat', 'MAE', 'PC', 'RMSE', 'EPSILON');


%% 70/30 training/test

MAE = zeros(10, length(k));
PC = zeros(10, length(k));
RMSE = zeros(10, length(k));
EPSILON = zeros(10, length(k));

for kk = 1:length(k)
    for i = 1:10;
        mask = labeled_masks70(:, i);
        unlabeled = mask == 0;

        b = ridge(labels(mask), X(mask, :), k(kk), 0);
        predicted = X(unlabeled, :) * b(2:end) + b(1);
        test = labels(unlabeled);
        mae = mean(abs(predicted - test));
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        
        MAE(i, kk) = mae;
        RMSE(i, kk) = rmse;
        PC(i, kk) = pc;
        EPSILON(i, kk) = ee;
    end
end



save('results_ridge70.mat', 'MAE', 'PC', 'RMSE', 'EPSILON');





%% 90/10 training/test

MAE = zeros(10, length(k));
PC = zeros(10, length(k));
RMSE = zeros(10, length(k));
EPSILON = zeros(10, length(k));

for kk = 1:length(k)
    for i = 1:10;
        mask = labeled_masks90(:, i);
        unlabeled = mask == 0;

        b = ridge(labels(mask), X(mask, :), k(kk), 0);
        predicted = X(unlabeled, :) * b(2:end) + b(1);
        test = labels(unlabeled);
        mae = mean(abs(predicted - test));
        pc = corr(predicted, test);
        rmse = sqrt( mean((predicted - test).^2 ));
        ee = mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
        
        MAE(i, kk) = mae;
        RMSE(i, kk) = rmse;
        PC(i, kk) = pc;
        EPSILON(i, kk) = ee;
    end
end






save('results_ridge90.mat', 'MAE', 'PC', 'RMSE', 'EPSILON');



