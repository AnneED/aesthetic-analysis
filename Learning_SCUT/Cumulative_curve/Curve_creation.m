


load('results_AE_NN.mat')
load('results_AE_RR.mat')
load('results_AE_SVR.mat')
load('results_AE_LGC.mat')
load('results_AE_FME.mat')
load('results_AE_KFME.mat')
load('results_AE_mean.mat')

% M = upper bound
M = [0:0.00001:0.6];
P_NN = [];
P_RR = [];
P_SVR = [];
P_LGC = [];
P_FME = [];
P_KFME = [];
P_mean = [];


for m = M
    p = mean(sum(AE_NN < m)/size(AE_NN, 1));
    P_NN = [P_NN; p];
    
    p = mean(sum(AE_RR < m)/size(AE_RR, 1));
    P_RR = [P_RR; p];
    
    p = mean(sum(AE_SVR < m)/size(AE_SVR, 1));
    P_SVR = [P_SVR; p];
    
    p = mean(sum(AE_LGC < m)/size(AE_LGC, 1));
    P_LGC = [P_LGC; p];
    
    p = mean(sum(AE_FME < m)/size(AE_FME, 1));
    P_FME = [P_FME; p];
    
    p = mean(sum(AE_KFME < m)/size(AE_KFME, 1));
    P_KFME = [P_KFME; p];  
    
    p = mean(sum(AE_mean < m)/size(AE_mean, 1));
    P_mean = [P_mean; p];

end


hold on;
plot(M, P_mean, 'color', [1 0.7 0], 'LineWidth', 1.1);
plot(M, P_NN, 'color', [0.7 0.3 0], 'LineWidth', 1.1);
hold on;
plot(M, P_RR, 'color', [1 0 0], 'LineWidth', 1.1);
hold on;
plot(M, P_SVR, 'color', [0 0.5 0.6], 'LineWidth', 1.1);
hold on;
plot(M, P_LGC, 'color', [0 0.65 0], 'LineWidth', 1.1);
hold on;
plot(M, P_FME, 'magenta', 'LineWidth', 1.1);
hold on;
plot(M, P_KFME, 'blue', 'LineWidth', 1.1);

legend('Mean', 'NN', 'RR', 'SVR', 'LGC', 'FME', 'KFME');
title('REC curves')
xlabel('Tolerance')
ylabel('Proportion of images')







