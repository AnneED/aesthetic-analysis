
% FME 						
% The Laplacian is based on gaussian feature similarity + score similarity
% Layer 7 is used (preprocessing: L2 normalization + PCA)
% Normalized labels


load('initial_data_SCUT_vgg');
var = devsn.^2;
labels = labelsn;
devs = devsn;

X_n = Xpca_7;



%% 50/50 training/test proportion

epsilon = 0.1;    % 1/10
para.uu = 0;
parameters = [10^(-9) 10^(-6) 10^(-3) 1 10^3 10^6 10^9];

MAE = zeros(10, length(parameters)^3);
PC = zeros(10, length(parameters)^3);
RMSE = zeros(10, length(parameters)^3);
EPSILON = zeros(10, length(parameters)^3);


tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           para.ul = parameters(i);    % beta in the paper
           para.mu =  parameters(j);     
           para.lamda = parameters(k);          % gamma in the paper
           parfor l = 1:10
               mask = labeled_masks50(:, l);
               unlabeled = (mask == 0);
               T = labels;    
               T(unlabeled) = 0;
               W = Gauss_GraphConstruction(X_n, epsilon, T, devs);
               L = double(diag(sum(W)) - W) ;   
               [W, b, F] = FME_semi2(X_n, L, T, para);
               mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
               pc = corr(labels(unlabeled), F(unlabeled));
               rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
               ee = mean(1 - exp(- (F(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
               MAE(l, index) = mae;
               PC(l, index) = pc;
               RMSE(l, index) = rmse;                    
               E(l, index) = ee;
           end
           index = index + 1;
       end
   end    
end
toc;


save('results_FME_50gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');






%% 70/30 training/test proportion

tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           para.ul = parameters(i);    % beta in the paper
           para.mu =  parameters(j);     
           para.lamda = parameters(k);          % gamma in the paper
           parfor l = 1:10
               mask = labeled_masks70(:, l);
               unlabeled = (mask == 0);
               T = labels;    
               T(unlabeled) = 0;
               W = Gauss_GraphConstruction(X_n, epsilon, T, devs);
               L = double(diag(sum(W)) - W) ;   
               [W, b, F] = FME_semi2(X_n, L, T, para);
               mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
               pc = corr(labels(unlabeled), F(unlabeled));
               rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
               ee = mean(1 - exp(- (F(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
               MAE(l, index) = mae;
               PC(l, index) = pc;
               RMSE(l, index) = rmse;                    
               E(l, index) = ee;
           end
           index = index + 1;
       end
   end    
end
toc;


save('results_FME_70gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');








%% 90/10 training/test proportion


tic;
index = 1;
for i = 1:7
   for j = 1:7
       for k = 1:7
           para.ul = parameters(i);    % beta in the paper
           para.mu =  parameters(j);     
           para.lamda = parameters(k);          % gamma in the paper
           parfor l = 1:10
               mask = labeled_masks90(:, l);
               unlabeled = (mask == 0);
               T = labels;    
               T(unlabeled) = 0;
               W = Gauss_GraphConstruction(X_n, epsilon, T, devs);
               L = double(diag(sum(W)) - W) ;   
               [W, b, F] = FME_semi2(X_n, L, T, para);
               mae = mean( abs(labels(unlabeled) - F(unlabeled) ));
               pc = corr(labels(unlabeled), F(unlabeled));
               rmse = sqrt( mean((F(unlabeled) - labels(unlabeled)).^2 ));
               ee = mean(1 - exp(- (F(unlabeled) - labels(unlabeled)).^2/2 ./var(unlabeled) ));
               MAE(l, index) = mae;
               PC(l, index) = pc;
               RMSE(l, index) = rmse;                    
               E(l, index) = ee;
           end
           index = index + 1;
       end
   end    
end
toc;


save('results_FME_90gauss_vgg7.mat', 'MAE', 'PC', 'RMSE', 'E');




%% Results
load('results_FME_50gauss_vgg7.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0626
% rmse = 0.0814
% pc = 0.8162
% ee = 0.1421


load('results_FME_70gauss_vgg7.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0563
% rmse = 0.0746
% pc = 0.8396
% ee = 0.1219


load('results_FME_90gauss_vgg7.mat');
[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))
% mae = 0.0567
% rmse = 0.0724
% pc = 0.8429
% ee = 0.1197
















