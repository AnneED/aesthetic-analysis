% 1-NN
% Best model: Euclidean distance, 70/30


% Loading data:
load('initial_data_SCUT_vgg.mat');
var = devsn.^2;
labels = labelsn;
devs = devsn;
X_n = Xpca_7;



%% 70/30 training/test

AE_euc = [];

for i = 1:10
    mask = labeled_masks70(:, i);
    unlabeled = mask == 0;
    test = labels(unlabeled);
    
    % EUCLIDEAN distance
    mdl = fitcknn((X_n(:, mask))', labels(mask), 'Distance', 'euclidean', 'NumNeighbors', 1);
    predicted = predict(mdl, (X_n(:, unlabeled))');
    
    % Error in prediction:
    ae = abs(predicted-test);
    % 500x1

    AE_euc = [AE_euc ae];    
    
end

AE_NN = AE_euc; 

save('results_AE_NN', 'AE_NN');







