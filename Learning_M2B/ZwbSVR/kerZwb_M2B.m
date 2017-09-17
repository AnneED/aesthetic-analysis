
% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize MAE


%% eastern:

% Loading data:
load('initial_data_M2Be_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_e;
X200 = Xpca_7e;

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];
var = devs.^2;
dims = 10:10:length(labels);




MAE = zeros(length(parameters)^3*length(parameters_T0), length(dims));
PC = zeros(length(parameters)^3*length(parameters_T0), length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = length(labels)*0.5;
u = l;

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
                    mask = labeled_masks50_e(:, ii);
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
                    [Z, W, b] = ZWb_SemiSupervised(X, classes_e(mask), L, Alpha, Gamma, Mu);
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


save('results_kerZwb_eM2B', 'MAE_zbw', 'PC_zbw');
toc;






clear;


%% western:

clear;
% Loading data:
load('initial_data_M2Bw_vgg.mat');
%var = devsn.^2;
devs = 0;
labels = labelsn_w;
X200 = Xpca_7w;

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
parameters_T0 = [1/8 1/4 1/2 1 2 4 8];
var = devs.^2;
dims = 10:10:length(labels);




MAE = zeros(length(parameters)^3*length(parameters_T0), length(dims));
PC = zeros(length(parameters)^3*length(parameters_T0), length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = length(labels)*0.5;
u = l;

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
                    mask = labeled_masks50_w(:, ii);
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
                    [Z, W, b] = ZWb_SemiSupervised(X, classes_w(mask), L, Alpha, Gamma, Mu);
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


save('results_kerZwb_wM2B', 'MAE_zbw', 'PC_zbw');
toc;










