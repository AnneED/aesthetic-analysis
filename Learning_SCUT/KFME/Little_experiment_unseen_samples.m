% Little experiment to check the performance of KFME on unseen data.
% Master thesis: Table 3.10.

load('initial_data_SCUT_vgg.mat');

vars = devsn.^2;
devs = devsn;

X = Xpca_7;

partition = cvpartition(classes, 'HoldOut', 0.2);
training_mask = partition.training;
% X
X_training = Xpca_7(:, training_mask); 
X_test = Xpca_7(:, training_mask == 0);
% labels
labels_training = labelsn(training_mask); 
labels_test = labelsn(training_mask == 0);
% var
var_training = vars(training_mask);
var_test = vars(training_mask == 0);
% classes
classes_training = classes(training_mask); 

partition = cvpartition(classes(training_mask), 'HoldOut', 0.5);
labeled_mask = partition.training;
% X
X_labeled = X_training(:, labeled_mask);
X_unlabeled = X_training(:, labeled_mask == 0);
% labels
labels_labeled = labels_training(labeled_mask);
labels_unlabeled = labels_training(labeled_mask == 0);
% var
var_labeled = var_training(labeled_mask);
var_unlabeled = var_training(labeled_mask == 0);
% classes
classes_labeled = classes_training(labeled_mask);

X = [X_labeled X_unlabeled X_test];
Labels = [labels_labeled; labels_unlabeled; labels_test];
var = [var_labeled; var_unlabeled; var_test];

save('partition_little_experiment', 'X', 'Labels', 'classes_labeled', 'var');

clear;


%% Knn graph. 

parameters_Beta = [0.1 1 10 100 1000 10000];
parameters_Gamma = [1 10 50 100 1000];
parameters_Mu = [0.0001 0.001 0.01 0.1 1 10];
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];

load('partition_little_experiment');

MAE = zeros(1, 6*5*6*7);
PC = zeros(1, 6*5*6*7);
RMSE = zeros(1, 6*5*6*7);
EPSILON = zeros(1, 6*5*6*7);

l = 200;
u = 200;

mask = zeros(l+u, 1);
mask(1:l) = 1;
mask = logical(mask);
unlabeled = (mask == 0);
test = Labels((l+u+1:end));
                    

index = 1;
tic;
for i = 1:length(parameters_Beta)
    for j = 1:length(parameters_Gamma)
        for k = 1:length(parameters_Mu)
            for ii = 1:length(parameters_T0)
                Beta = parameters_Beta(i);
                Gamma = parameters_Gamma(j);
                Mu = parameters_Mu(k);
                T0 = parameters_T0(ii);
                
% using KFME:
                    [F, Alphas] = KernelFME_Fadi2(X(:, 1:l+u), Labels(1:l+u), mask, Beta, Gamma, Mu, T0);
                    K = KernelMatrix(X(:, (l+u+1):end), X(:, 1:l+u),T0);
                    F = K*Alphas;
                    max_labels = max(Labels);
                    min_labels = min(Labels); 
                    predicted = (F-min(F))*(max_labels - min_labels)/(max(F)-min(F)) + min_labels;
                    mae = mean(abs(predicted - test));
                    pc = corr(predicted, test);
                    rmse = sqrt( mean((predicted - test).^2 ));
                    ee = mean(1 - exp(- (predicted - test).^2/2 ./var((l+u+1:end)) ));
                    MAE(1, index) = mae;
                    PC(1, index) = pc;
                    RMSE(1, index) = rmse;
                    E(1, index) = ee;
                
                index = index + 1;
            end
        end
    end
end
toc;


save('results_little_experiment_KFME', 'MAE', 'PC', 'RMSE', 'E');




%% Results

load('results_little_experiment_KFME');
[mae, idx] = min(MAE(:));
mae
rmse = RMSE(1, idx)
pc = PC(1, idx)
ee = E(1, idx)
% mae = 0.0751
% rmse = 0.0926
% pc = 0.7988
% ee = 0.1714
























 
