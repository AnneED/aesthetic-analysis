
% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize MAE

% Loading data:
load('initial_data_SCUT_vgg.mat');
X200 = Xpca_7;
devs = devsn;
labels = labelsn;

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];
var = devs.^2;
dims = 10:10:500;




%% 50/50 training/test proportion

MAE = zeros(length(parameters)^3*length(parameters_T0), length(dims));
PC = zeros(length(parameters)^3*length(parameters_T0), length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 250;
u = 250;

for jj = 1:length(parameters_T0)
    T0 = parameters_T0(jj);
    Ker = KernelMat(X200, T0);
    for i = 1:7
        for j = 1:7
            for k = 1:7
                Alpha = parameters(i);            
                Gamma = parameters(j);
                Mu = parameters(k);

                % Training the classifier:
                for ii = 1:10
                    mask = labeled_masks50(:, ii);
                    unlabeled = (mask == 0);
                    test = labels(unlabeled);

                    % The first columns correspond to labeled instances and the last ones to unlabeled ones
                    X = zeros(l+u, l+u);
                    X(1:l, 1:l) = Ker(mask, mask);
                    X(l+1:l+u, l+1:l+u) = Ker(unlabeled, unlabeled);
                    X(1:l, l+1:l+u) = Ker(mask, unlabeled);
                    X(l+1:l+u, 1:l) = Ker(unlabeled, mask);
                    

                    % Building the Laplacian matrix (K = 10): 
                    XX = [X200(:, mask) X200(:, unlabeled)];
                
                    [~, W]= KNN_GraphConstruction(XX, 10);
                    L = double(diag(sum(W)) - W) ; 


                    % Non-linear transformation:
                    [Z, W, b] = ZWb_SemiSupervised(X, classes(mask), L, Alpha, Gamma, Mu);
                    % Z = W'*X200 + b*ones(1, 500); 
                    % column representation

                    % Z = X200'*W + ones(500, 1) * b';
                    % row representation


                    for kk = 1:length(dims)

                        Mdl = fitrlinear(Z(1:l, 1:dims(kk)), labels(mask));
                        % it uses the row representation

                        % Performance measurement:
                        yfit = predict(Mdl, Z(l+1:end, 1:dims(kk)));
                        mae = mean(abs(yfit - test));
                        pc = corr(yfit, test);
                        MAE(index, kk) = mae + MAE(index, kk);
                        PC(index, kk) = pc + PC(index, kk);
                    end
                end

                index = index + 1;
            end
        end
    end
end



MAE_zbw = MAE/10;
PC_zbw = PC/10;


save('results_optimization_kerZwb_vgg7_knn50', 'MAE_zbw', 'PC_zbw');
toc;

disp('50/50 finished');














%% 70/30 training/test proportion

MAE = zeros(length(parameters)^3*length(parameters_T0), length(dims));
PC = zeros(length(parameters)^3*length(parameters_T0), length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 350;
u = 150;
for jj = 1:length(parameters_T0)
T0 = parameters_T0(jj);
Ker = KernelMat(X200, T0);
for i = 1:7
    for j = 1:7
        for k = 1:7
            Alpha = parameters(i);            
            Gamma = parameters(j);
            Mu = parameters(k);
            
            % Training the classifier:
			for ii = 1:10
				mask = labeled_masks70(:, ii);
                unlabeled = (mask == 0);
                test = labels(unlabeled);


                    % The first columns correspond to labeled instances and the last ones to unlabeled ones
                    X = zeros(l+u, l+u);
                    X(1:l, 1:l) = Ker(mask, mask);
                    X(l+1:l+u, l+1:l+u) = Ker(unlabeled, unlabeled);
                    X(1:l, l+1:l+u) = Ker(mask, unlabeled);
                    X(l+1:l+u, 1:l) = Ker(unlabeled, mask);
                    

                    % Building the Laplacian matrix (K = 10): 
                    XX = [X200(:, mask) X200(:, unlabeled)];
                
                    [~, W]= KNN_GraphConstruction(XX, 10);
                    L = double(diag(sum(W)) - W) ; 


				% Non-linear transformation:
				[Z, W, b] = ZWb_SemiSupervised(X, classes(mask), L, Alpha, Gamma, Mu);
				% Z = W'*X200 + b*ones(1, 500); 
                % column representation
                
                % Z = X200'*W + ones(500, 1) * b';
				% row representation
  
                
	            for kk = 1:length(dims)
                                     
                    Mdl = fitrlinear(Z(1:l, 1:dims(kk)), labels(mask));
                    % it uses the row representation
                
                    % Performance measurement:
                    yfit = predict(Mdl, Z(l+1:end, 1:dims(kk)));
                    mae = mean(abs(yfit - test));
                    pc = corr(yfit, test);
                    MAE(index, kk) = mae + MAE(index, kk);
                    PC(index, kk) = pc + PC(index, kk);
                end
            end

            index = index + 1;
        end
    end
end
end



MAE_zbw = MAE/10;
PC_zbw = PC/10;


save('results_optimization_kerZwb_vgg7_knn70', 'MAE_zbw', 'PC_zbw');
toc;

disp('70/30 finished');





%% 90/10 training/test proportion

MAE = zeros(length(parameters)^3*length(parameters_T0), length(dims));
PC = zeros(length(parameters)^3*length(parameters_T0), length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 450;
u = 50;
for jj = 1:length(parameters_T0)
T0 = parameters_T0(jj);
Ker = KernelMat(X200, T0);
for i = 1:7
    for j = 1:7
        for k = 1:7
            Alpha = parameters(i);            
            Gamma = parameters(j);
            Mu = parameters(k);
            
            % Training the classifier:
			for ii = 1:10
				mask = labeled_masks90(:, ii);
                unlabeled = (mask == 0);
                test = labels(unlabeled);

                    % The first columns correspond to labeled instances and the last ones to unlabeled ones
                    X = zeros(l+u, l+u);
                    X(1:l, 1:l) = Ker(mask, mask);
                    X(l+1:l+u, l+1:l+u) = Ker(unlabeled, unlabeled);
                    X(1:l, l+1:l+u) = Ker(mask, unlabeled);
                    X(l+1:l+u, 1:l) = Ker(unlabeled, mask);
                    

                    % Building the Laplacian matrix (K = 10): 
                    XX = [X200(:, mask) X200(:, unlabeled)];
                
                    [~, W]= KNN_GraphConstruction(XX, 10);
                    L = double(diag(sum(W)) - W) ; 

				% Non-linear transformation:
				[Z, W, b] = ZWb_SemiSupervised(X, classes(mask), L, Alpha, Gamma, Mu);
				% Z = W'*X200 + b*ones(1, 500); 
                % column representation
                
                % Z = X200'*W + ones(500, 1) * b';
				% row representation
  
                
	            for kk = 1:length(dims)
                                     
                    Mdl = fitrlinear(Z(1:l, 1:dims(kk)), labels(mask));
                    % it uses the row representation
                
                    % Performance measurement:
                    yfit = predict(Mdl, Z(l+1:end, 1:dims(kk)));
                    mae = mean(abs(yfit - test));
                    pc = corr(yfit, test);
                    MAE(index, kk) = mae + MAE(index, kk);
                    PC(index, kk) = pc + PC(index, kk);
                end
            end

            index = index + 1;
        end
    end
end
end



MAE_zbw = MAE/10;
PC_zbw = PC/10;


save('results_optimization_kerZwb_vgg7_knn90', 'MAE_zbw', 'PC_zbw');
toc;













%% Optimization of linear SVR:

% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize PC

X_n = Xpca_7;

% libsvm:
addpath /home/john-san/workspace/libsvm-3.22/matlab

% parameters:
parameters_c = [0.00001 0.0001 0.001 0.01 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1 2 3 4 5 10 20 30];
parameters_p = [0 0.001 0.0025 0.005 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.15];% length(parameters_p) =       ii
% length(parameters_c) =      k
npar = length(parameters_c) * length(parameters_p);

pp = [];
for jj = 1:length(parameters_T0)
    for i = 1:7
        for j = 1:7
            for k = 1:7
                pp = [pp; parameters(i) parameters(j) parameters(k) parameters_T0(jj)];
            end
        end
    end
end



%% 50/50 training/test proportion


l = 250;
u = 250;

load('results_optimization_kerZwb_vgg7_knn50');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);

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
                
                % The first columns correspond to labeled instances and the last ones to unlabeled ones
                X = zeros(l+u, l+u);
                X(1:l, 1:l) = Ker(mask, mask);
                X(l+1:l+u, l+1:l+u) = Ker(unlabeled, unlabeled);
                X(1:l, l+1:l+u) = Ker(mask, unlabeled);
                X(l+1:l+u, 1:l) = Ker(unlabeled, mask);


                % Building the Laplacian matrix (K = 10): 
                XX = [X200(:, mask) X200(:, unlabeled)];
                [~, W]= KNN_GraphConstruction(XX, 10);
                L = double(diag(sum(W)) - W) ; 

                % non-linear projection
                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

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
            k;
        end
end

save('results_kerZbw_linearSvr_vgg7_knn50', 'MAE', 'PC', 'RMSE', 'E');









%% 70/30 training/test proportion


l = 350;
u = 150;

load('results_optimization_kerZwb_vgg7_knn70');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);

Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);

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
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

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





save('results_kerZbw_linearSvr_vgg7_knn70', 'MAE', 'PC', 'RMSE', 'E');










%% 90/10 training/test proportion


l = 450;
u = 50;

load('results_optimization_kerZwb_vgg7_knn90');
% we optimize Zwb according to the MAE


% We choose Alpha, Gamma and Mu according to the model with the best PC.
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col);


Alpha = pp(I_row, 1);
Gamma = pp(I_row, 2);
Mu = pp(I_row, 3);
T0 = pp(I_row, 4);
Ker = KernelMat(X200, T0);


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
                
                % non-linear projection
                X = [X_n(:, mask) X_n(:, unlabeled)];

                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 

                [Z, W, b] = ZWb_SemiSupervised (X, classes(mask), L, Alpha, Gamma, Mu);

                Z = double(Z(:, 1:dim));
          
                model = svmtrain(labels(mask), Z(1:l,:), ['-s 3 -t 0 -p ' num2str(p) ' -c ' num2str(c)]);
                yfit = svmpredict(labels(unlabeled), Z(l+1:end,:),model);

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



save('results_kerZbw_linearSvr_vgg7_knn90', 'MAE', 'PC', 'RMSE', 'E');







%% Results

para_SVR = [];
for i = 1:length(parameters_p)
    for j = 1:length(parameters_c)
        para_SVR = [para_SVR; parameters_p(i) parameters_c(j)];
    end
end



load('results_kerZbw_linearSvr_vgg7_knn50')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))

p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

load('results_optimization_kerZwb_vgg7_knn50');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)





load('results_kerZbw_linearSvr_vgg7_knn70')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))



p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

load('results_optimization_kerZwb_vgg7_knn70');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)




load('results_kerZbw_linearSvr_vgg7_knn90')

[mae, idx] = min(mean(MAE));
mae
rmse = mean(RMSE(:, idx))
pc = mean(PC(:, idx))
ee = mean(E(:, idx))



p = para_SVR(idx, 1)
c = para_SVR(idx, 2)

load('results_optimization_kerZwb_vgg7_knn90');
[~, I] = min(MAE_zbw(:));
[I_row, I_col] = ind2sub(size(MAE_zbw), I);
dim = dims(I_col)

Alpha = pp(I_row, 1)
Gamma = pp(I_row, 2)
Mu = pp(I_row, 3)
T0 = pp(I_row, 4)




















