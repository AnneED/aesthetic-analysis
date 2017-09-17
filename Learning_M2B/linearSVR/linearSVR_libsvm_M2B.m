
% SVR with linear kernel
% 50/50 training/test proportion


%% Eastern:
clear;

load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = double(labelsn_e);
X = double(Xpca_7e)';

% adding the path of libsm
addpath /home/john-san/workspace/libsvm-3.22/matlab


% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];

npar = length(parameters_c) * length(parameters_p) ;


MAE = zeros(10, npar); 
PC = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks50_e(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
            end
            index = index + 1;
            ii
            k
        end
end


save('results_linearSVR_eM2B', 'MAE', 'PC', 'RMSE');



%% Eastern:
clear;

load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = double(labelsn_w);
X = double(Xpca_7w)';

addpath /home/john-san/workspace/libsvm-3.22/matlab


% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];

npar = length(parameters_c) * length(parameters_p) ;


MAE = zeros(10, npar); 
PC = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
                p = parameters_p(ii);
                c = parameters_c(k);

            parfor ll = 1:10
                mask = labeled_masks50_w(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

                mae = mean(abs(yfit - test));
                
                MAE(ll, index) = mae;
                pc = corr(yfit, test);
                rmse = sqrt( mean((yfit - test).^2 ));
                PC(ll, index) = pc;
                RMSE(ll, index) = rmse;
            end
            index = index + 1;
            ii
            k
        end
end


save('results_linearSVR_wM2B', 'MAE', 'PC', 'RMSE');



%% Results

load('results_linearSVR_eM2B')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1356
% rmse = 0.1679
% pc = 0.4397


load('results_linearSVR_wM2B')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
% mae = 0.1144
% rmse = 0.1435
% pc = 0.6239


