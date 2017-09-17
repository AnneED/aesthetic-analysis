
% Zwb + SVR 
% features: vgg-face layer 7 (preprocessing: L2 normalization + pca (200 dimensions) )
% normalized labels
% optimize PC

% Loading data:
load('initial_data_SCUT_vgg.mat');
X200 = Xpca_7;
devs = devsn;
labels = labelsn;

parameters = [1e-09 1e-06 1e-03 1 1e+3 1e+6 1e+9]; 
var = devs.^2;
dims = 10:10:500;

%% 50/50 training/test proportion

MAE = zeros(7*7*7, length(dims));
PC = zeros(7*7*7, length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 250;
u = 250;
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
                X = [X200(:, mask) X200(:, unlabeled)];
                
                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
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


MAE_zbw = MAE/10;
PC_zbw = PC/10;


save('results_optimization_Zwb_vgg7_semi_knn50', 'MAE_zbw', 'PC_zbw');
toc;

disp('50/50 finished');














%% 70/30 training/test proportion

MAE = zeros(7*7*7, length(dims));
PC = zeros(7*7*7, length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 350;
u = 150;
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
                X = [X200(:, mask) X200(:, unlabeled)];
                
                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 


				% Non-linear transformation:
				[Z, W, b] = ZWb_SemiSupervised(X, classes(mask), L, Alpha, Gamma, Mu);
				
  
                
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


MAE_zbw = MAE/10;
PC_zbw = PC/10;

save('results_optimization_Zwb_vgg7_semi_knn70', 'MAE_zbw', 'PC_zbw');
toc;

disp('70/30 finished');





%% 90/10 training/test proportion

MAE = zeros(7*7*7, length(dims));
PC = zeros(7*7*7, length(dims));

% each row in MAE (RMSE/PC/E) will contain the mean average error
% depending on dim for some fixed values of Alpha, Gamma and Mu

% Option 1: using the Z given by the output of the algorithm
index = 1;
tic;
l = 450;
u = 50;
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
                X = [X200(:, mask) X200(:, unlabeled)];
                
                % Building the Laplacian matrix (K = 10): 
                [~, W]= KNN_GraphConstruction(X, 10);
                L = double(diag(sum(W)) - W) ; 


				% Non-linear transformation:
				[Z, W, b] = ZWb_SemiSupervised(X, classes(mask), L, Alpha, Gamma, Mu);
  
                
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
toc;

MAE_zbw = MAE/10;
PC_zbw = PC/10;

save('results_optimization_Zwb_vgg7_semi_knn90', 'MAE_zbw', 'PC_zbw');
toc;

































