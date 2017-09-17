
% SVR with linear kernel
% 50/50 training/test proportion


% Loading data:
load('initial_data_SCUT_vgg.mat');
addpath /home/john-san/workspace/libsvm-3.22/matlab
labels = double(labelsn);
devs = devsn;
var = devs.^2;
X = double(Xpca_7');


% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];

npar = length(parameters_c) * length(parameters_p) ;


%% 50/50

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks50(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end


save('results_linearSVR_SCUT_vgg7_50', 'MAE', 'PC', 'RMSE', 'E');







%% 70/30

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)

                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks70(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end


save('results_linearSVR_SCUT_vgg7_70', 'MAE', 'PC', 'RMSE', 'E');



%% 90/10

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
            
                p = parameters_p(ii);
                c = parameters_c(k);
            
            parfor ll = 1:10
                mask = labeled_masks90(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 2 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                ee = mean(1 - exp(- (yfit - test).^2/2 ./var(unlabeled) ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
                E(ll, index) = ee; 
            end
            index = index + 1;
            ii
            k
        end
end


save('results_linearSVR_SCUT_vgg7_90', 'MAE', 'PC', 'RMSE', 'E');




%% Results before widenning the parameter range

parameters = [];
for ii = 1:length(parameters_p)
    for k = 1:length(parameters_c)
            parameters = [parameters; parameters_p(ii) parameters_c(k)];        
    end
end


load('results_linearSVR_SCUT_vgg7_50')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))

p = parameters(idx, 1)
c = parameters(idx, 2)
% mae = 0.0615
% rmse = 0.0792
% pc = 0.8133
% ee = 0.1346

% p = 0.03
% c = 0.3


load('results_linearSVR_SCUT_vgg7_70')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))

p = parameters(idx, 1)
c = parameters(idx, 2)

% mae = 0.0563
% rmse = 0.0734
% pc = 0.8368
% ee = 0.1180

% p = 0.03
% c = 0.2



load('results_linearSVR_SCUT_vgg7_90')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))


p = parameters(idx, 1)
c = parameters(idx, 2)
% mae = 0.0572
% rmse = 0.0721
% pc = 0.8375
% ee = 0.1169

% p = 0.03
% c = 20


