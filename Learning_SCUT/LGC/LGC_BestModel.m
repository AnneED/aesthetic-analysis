
% LGC 
% The similarity matrix is based on gaussian feature similarity
% labels are normalized
% features: vgg layer 7 is used (preprocessing: L2 normalization + pca (200 dimensions) ) 

load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
labels = labelsn;
X_n = Xpca_7;

% Building the similarity matrix (K = 10):
addpath('/home/john-san/Dropbox/Master/TFM/Codes_drive');
[~, W]= KNN_GraphConstruction(X_n, 10);

% Normalize the graph:

% Divide each row by its maximum, so the maximum value of each row is 1:
for i = 1:size(W,1)
W(i, :) = W(i, :) / sum(W(i, :));
end

% Make the graph symmetric:
W = (W + W')/2;

parameters = [10^(-6) 10^(-5) 10^(-4) 10^(-3) 10^(-2) 0.1 1 10 100 1000 10^4 10^5 10^6];

load('results_LGC_normGraph70.mat');
[mae, idx] = min(mean(MAE));
Mu = parameters(idx);




%% 90/10 training/test proportion

AE_LGC = [];

for j = 1:10
    mask = labeled_masks70(:, j);
    unlabeled = (mask == 0);
    Y = labels; 
    Y(unlabeled) = mean(labels(mask));
    F = LGC(X_n, W, Mu, Y);
    predicted = F(unlabeled);
    test = labels(unlabeled);
    ae = abs(predicted-test);
            AE_LGC = [AE_LGC ae];

end


save('results_AE_LGC', 'AE_LGC');


































