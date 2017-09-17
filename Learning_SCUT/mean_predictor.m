
load('initial_data_SCUT_vgg.mat')

var = devsn.^2;
labels = labelsn;


mae50 = 0; rmse50 = 0; pc50 = 0; ee50 = 0;
mae70 = 0; rmse70 = 0; pc70 = 0; ee70 = 0;
mae90 = 0; rmse90 = 0; pc90 = 0; ee90 = 0;

AE_mean = [];

for i = 1:10
   
   % 50/50
   mask = labeled_masks50(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 250, 1);
   mae50 =  mae50 + mean(abs(predicted - test));    
   pc50 = pc50 + corr(predicted, test);
   rmse50 = rmse50 + sqrt( mean((predicted - test).^2 ));
   ee50 = ee50 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 70/30
   mask = labeled_masks70(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 150, 1);
   ae = abs(predicted - test);
   AE_mean = [AE_mean ae];  
   mae70 =  mae70 + mean(abs(predicted - test));    
   pc70 = pc70 + corr(predicted, test);
   rmse70 = rmse70 + sqrt( mean((predicted - test).^2 ));
   ee70 = ee70 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 90/10
   mask = labeled_masks90(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 50, 1);
   mae90 =  mae90 + mean(abs(predicted - test)); 
   pc90 = pc90 + corr(predicted, test);
   rmse90 = rmse90 + sqrt( mean((predicted - test).^2 ));
   ee90 = ee90 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
       
end

mae50 = mae50/10; rmse50 = rmse50/10; pc50 = pc50/10; ee50 = ee50/10;
mae70 = mae70/10; rmse70 = rmse70/10; pc70 = pc70/10; ee70 = ee70/10;
mae90 = mae90/10; rmse90 = rmse90/10; pc90 = pc90/10; ee90 = ee90/10;



%% Errors: 

% 50/50

mae50
rmse50
pc50 
ee50
% mae = 0.1009
% rmse = 0.1351
% pc = 0
% ee = 0.2587



% 70/30

mae70
rmse70
pc70 
ee70
% mae = 0.0979
% rmse = 0.1322
% pc = 0
% ee = 0.2479



% 90/10

mae90
rmse90
pc90 
ee90
% mae = 0.0987
% rmse = 0.1309
% pc = 0
% ee = 0.2557



save('results_AE_mean', 'AE_mean');




