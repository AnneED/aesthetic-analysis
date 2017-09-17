
% FME 						
% The Laplacian is based on gaussian feature similarity (with 10 neighbors)
% Layer 7 is used (preprocessing: L2 normalization + PCA)
% Normalized labels


load('initial_data_SCUT_vgg');
var = devsn.^2;
labels = labelsn;

X_n = Xpca_7;


% Building the Laplacian matrix (K = 10):
[~, W_ind]= KNN_GraphConstruction(X_n, 10);
L = double(diag(sum(W_ind)) - W_ind) ;   


%% 50/50 training/test proportion

para.uu = 0;
parameters = [10^(-9) 10^(-6) 10^(-3) 1 10^3 10^6 10^9];

pp = [];
for i = 1:7
   for j = 1:7
       for k = 1:7
           para.ul = parameters(i);    % beta in the paper
           para.mu =  parameters(j);     
           para.lamda = parameters(k);          % gamma in the paper
           pp = [pp; para.ul para.mu para.lamda];
       end
   end
end

load('results_FME_70knn_vgg7.mat');
[~, idx] = min(mean(MAE));
para.ul = pp(idx, 1);
para.mu = pp(idx, 2);
para.lamda = pp(idx, 3);


AE_FME = [];




%% 90/10 training/test proportion


for l = 1:10
   mask = labeled_masks70(:, l);
   unlabeled = (mask == 0);
   T = labels;    
   T(unlabeled) = 0;
   [W, b, F] = FME_semi2(X_n, L, T, para);
   ae = abs(labels(unlabeled) - F(unlabeled) );
   AE_FME = [AE_FME ae];

end


save('results_AE_FME', 'AE_FME');


























