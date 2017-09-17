
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
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 1.1 1.25 5 10];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];
parameters_g = [1/32 1/16 1/8 0.2 1/4 0.3 0.4 1/2 0.6 0.7 0.8 0.9 1 2 4 8 16 32];

npar = length(parameters_c) * length(parameters_p) * length(parameters_g);


%% 50/50

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
            for kk = 1:length(parameters_g)

                p = parameters_p(ii);
                c = parameters_c(k);
                g = parameters_g(kk);

            parfor ll = 1:10
                mask = labeled_masks50(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 2 -p ' num2str(p) ' -c ' num2str(c) ' -g ' num2str(g)]);
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
end


save('results_gaussianSVR_SCUT_vgg7_50', 'MAE', 'PC', 'RMSE', 'E');
save('results_gaussianSVR_SCUT_vgg7', 'MAE', 'PC', 'RMSE', 'E');






%% 70/30

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
            for kk = 1:length(parameters_g)

                p = parameters_p(ii);
                c = parameters_c(k);
                g = parameters_g(kk);

            parfor ll = 1:10
                mask = labeled_masks70(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 2 -p ' num2str(p) ' -c ' num2str(c) ' -g ' num2str(g)]);
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
end


save('results_gaussianSVR_SCUT_vgg7_70', 'MAE', 'PC', 'RMSE', 'E');



%% 90/10

MAE = zeros(10, npar); 
PC = zeros(10, npar);
E = zeros(10, npar);
RMSE = zeros(10, npar);

index = 1;
tic;
for ii = 1:length(parameters_p)
        for k = 1:length(parameters_c)
            for kk = 1:length(parameters_g)

                p = parameters_p(ii);
                c = parameters_c(k);
                g = parameters_g(kk);

            parfor ll = 1:10
                mask = labeled_masks90(:, ll);
                unlabeled = mask == 0;
                test = labels(unlabeled);
                
                model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 2 -p ' num2str(p) ' -c ' num2str(c) ' -g ' num2str(g)]);
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
end


save('results_gaussianSVR_SCUT_vgg7_90', 'MAE', 'PC', 'RMSE', 'E');




%% Results before widenning the parameter range

parameters = [];
for ii = 1:length(parameters_p)
    for k = 1:length(parameters_c)
        for kk = 1:length(parameters_g)
            parameters = [parameters; parameters_p(ii) parameters_c(k) parameters_g(kk)];
        end 
    end
end


load('results_gaussianSVR_SCUT_vgg7_50')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0612
% rmse = 0.0789
% pc = 0.8166
% ee = 0.1341

% LOS RESULTADOS DEL CHICO SON MEJORES. VAMOS
% A VER LOS PARAMETROS MEJORES Y A AMPLIAR LA BUSQUEDA 
% EN TORNO A ELLOS.

p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% p = 0.03
% c = 0.7
% g = 0.25
 

load('results_gaussianSVR_SCUT_vgg7_70')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0560
% rmse = 0.0731
% pc = 0.8381
% ee = 0.1173

p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% p = 0.02
% c = 0.5
% g = 0.25



load('results_gaussianSVR_SCUT_vgg7_90')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0563
% rmse = 0.0713
% pc = 0.8432
% ee = 0.1163

p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% p = 0.02
% c = 0.25
% g = 0.5



%% Results after widenning the parameter range:


load('results_gaussianSVR_SCUT_vgg7_50')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))

p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% mae = 0.0612
% rmse = 0.0789
% pc = 0.8170
% ee = 0.1342



load('results_gaussianSVR_SCUT_vgg7_70')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))


p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% mae = 0.0560
% rmse = 0.0729
% pc = 0.8393
% ee = 0.1170



load('results_gaussianSVR_SCUT_vgg7_90')
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))


p = parameters(idx, 1)
c = parameters(idx, 2)
g = parameters(idx, 3)

% mae = 0.0561
% rmse = 0.0711
% pc = 0.8434
% ee = 0.1151




