
% LGC 
% The similarity matrix is based on cosine feature similarity + score
% similarity
% labels are normalized
% Vgg-layer 7 is used (preprocessing: L2 normalization + pca (200 dimensions))


load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
devs = devsn;
labels = labelsn;
X_n = Xpca_7;



%% 50/50 training/test proportion

epsilon = 0.1;
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
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs);
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


save('results_LGC_cos50.mat', 'MAE', 'PC', 'RMSE', 'E');




%% 70/30 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks70(:, j);
        unlabeled = (mask == 0);
        Y = labels;
        Y(unlabeled) = 0;
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs);
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


save('results_LGC_cos70.mat', 'MAE', 'PC', 'RMSE', 'E');





%% 90/10 training/test proportion


for i = 1:length(parameters)
    Mu = parameters(i);
    for j = 1:10
        mask = labeled_masks90(:, j);
        unlabeled = (mask == 0);
        Y = labels; 
        Y(unlabeled) = 0;
        W = Cos_GraphConstruction4(X_n, epsilon, Y, devs);
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


save('results_LGC_cos90.mat', 'MAE', 'PC', 'RMSE', 'E');






%% Results

load('results_LGC_cos50.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0946
% rmse = 0.1212
% pc = 0.4755
% ee = 0.2483




load('results_LGC_cos70.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0889
% rmse = 0.1115
% pc = 0.5917
% ee = 0.2341




load('results_LGC_cos90.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0953
% rmse = 0.1144
% pc = 0.6417
% ee = 0.2684
















