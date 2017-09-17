
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

load('results_gaussianSVR_SCUT_vgg7_70');

parameters = [];
for ii = 1:length(parameters_p)
    for k = 1:length(parameters_c)
        for kk = 1:length(parameters_g)
            parameters = [parameters; parameters_p(ii) parameters_c(k) parameters_g(kk)];
        end 
    end
end

[mae, idx] = min(mean(MAE));
p = parameters(idx, 1);
c = parameters(idx, 2);
g = parameters(idx, 3);




%% 70/30

AE_SVR = [];

for ll = 1:10
    mask = labeled_masks70(:, ll);
    unlabeled = mask == 0;
    test = labels(unlabeled);

    model = svmtrain(labels(mask), X(mask,:), ['-s 3 -t 2 -p ' num2str(p) ' -c ' num2str(c) ' -g ' num2str(g)]);
    yfit = svmpredict(labels(unlabeled), X(unlabeled,:),model);

    ae = abs(yfit - test);

    AE_SVR = [AE_SVR ae];
end



save('results_AE_SVR', 'AE_SVR');





