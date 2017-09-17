
% M2B_e

load('initial_data_M2Be_vgg.mat')


labelsn = labelsn_e;
labels = labelsn_e;

mae50 = 0; rmse50 = 0; pc50 = 0; ee50 = 0;
mae70 = 0; rmse70 = 0; pc70 = 0; ee70 = 0;
mae90 = 0; rmse90 = 0; pc90 = 0; ee90 = 0;



for i = 1:10
   
   % 50/50
   mask = labeled_masks50_e(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 310, 1);
   mae50 =  mae50 + mean(abs(predicted - test));    
   pc50 = pc50 + corr(predicted, test);
   rmse50 = rmse50 + sqrt( mean((predicted - test).^2 ));
  % ee50 = ee50 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 70/30
   mask = labeled_masks70_e(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 186, 1);
   mae70 =  mae70 + mean(abs(predicted - test));    
   pc70 = pc70 + corr(predicted, test);
   rmse70 = rmse70 + sqrt( mean((predicted - test).^2 ));
  % ee70 = ee70 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 90/10
   mask = labeled_masks90_e(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 62, 1);
   mae90 =  mae90 + mean(abs(predicted - test));    
   pc90 = pc90 + corr(predicted, test);
   rmse90 = rmse90 + sqrt( mean((predicted - test).^2 ));
  % ee90 = ee90 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
       
end

mae50 = mae50/10; rmse50 = rmse50/10; pc50 = pc50/10; ee50 = ee50/10;
mae70 = mae70/10; rmse70 = rmse70/10; pc70 = pc70/10; ee70 = ee70/10;
mae90 = mae90/10; rmse90 = rmse90/10; pc90 = pc90/10; ee90 = ee90/10;



%% Errors: 

% 50/50

mae50
rmse50
pc50
% mae = 0.1536
% rmse = 0.1866
% pc = 0




% 70/30

mae70
rmse70
pc70 
% mae = 0.1542
% rmse = 0.1876
% pc = 0




% 90/10

mae90
rmse90
pc90 
% mae = 0.1550
% rmse = 0.1899
% pc =  0






% M2B_w

load('initial_data_M2Bw_vgg.mat')


labels = labelsn_w;
labelsn = labelsn_w;


mae50 = 0; rmse50 = 0; pc50 = 0; ee50 = 0;
mae70 = 0; rmse70 = 0; pc70 = 0; ee70 = 0;
mae90 = 0; rmse90 = 0; pc90 = 0; ee90 = 0;



for i = 1:10
   
   % 50/50
   mask = labeled_masks50_w(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 310, 1);
   mae50 =  mae50 + mean(abs(predicted - test));    
   pc50 = pc50 + corr(predicted, test);
   rmse50 = rmse50 + sqrt( mean((predicted - test).^2 ));
%   ee50 = ee50 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 70/30
   mask = labeled_masks70_w(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 186, 1);
   mae70 =  mae70 + mean(abs(predicted - test));    
   pc70 = pc70 + corr(predicted, test);
   rmse70 = rmse70 + sqrt( mean((predicted - test).^2 ));
  % ee70 = ee70 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 90/10
   mask = labeled_masks90_w(:, i);
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 62, 1);
   mae90 =  mae90 + mean(abs(predicted - test));    
   pc90 = pc90 + corr(predicted, test);
   rmse90 = rmse90 + sqrt( mean((predicted - test).^2 ));
  % ee90 = ee90 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
       
end

mae50 = mae50/10; rmse50 = rmse50/10; pc50 = pc50/10; ee50 = ee50/10;
mae70 = mae70/10; rmse70 = rmse70/10; pc70 = pc70/10; ee70 = ee70/10;
mae90 = mae90/10; rmse90 = rmse90/10; pc90 = pc90/10; ee90 = ee90/10;



%% Errors: 

% 50/50

mae50
rmse50
pc50
% mae = 0.1512
% rmse = 0.1831
% pc = 0




% 70/30

mae70
rmse70
pc70
% mae = 0.1488
% rmse = 0.1799
% pc = 0


% 90/10

mae90
rmse90
pc90
% mae = 0.1530
% rmse = 0.1841
% pc = 0



% M2B_e + M2B_w

load('initial_data_M2Be_vgg.mat')
load('initial_data_M2Bw_vgg.mat')

labels = [labelsn_e; labelsn_w];
labelsn = [labelsn_e; labelsn_w];


mae50 = 0; rmse50 = 0; pc50 = 0; ee50 = 0;
mae70 = 0; rmse70 = 0; pc70 = 0; ee70 = 0;
mae90 = 0; rmse90 = 0; pc90 = 0; ee90 = 0;



for i = 1:10
   
   % 50/50
   mask = [labeled_masks50_e(:, i); labeled_masks50_w(:, i)];
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 620, 1);
   mae50 =  mae50 + mean(abs(predicted - test));    
   pc50 = pc50 + corr(predicted, test);
   rmse50 = rmse50 + sqrt( mean((predicted - test).^2 ));
   %ee50 = ee50 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 70/30
   mask = [labeled_masks70_e(:, i); labeled_masks70_w(:, i)];
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 372, 1);
   mae70 =  mae70 + mean(abs(predicted - test));    
   pc70 = pc70 + corr(predicted, test);
   rmse70 = rmse70 + sqrt( mean((predicted - test).^2 ));
   %ee70 = ee70 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
   
   
   % 90/10
   mask = [labeled_masks90_e(:, i); labeled_masks90_w(:, i)];
   unlabeled = mask == 0;
   test = labelsn(unlabeled);
   predicted = repmat(mean(labels(mask)), 124, 1);
   mae90 =  mae90 + mean(abs(predicted - test));    
   pc90 = pc90 + corr(predicted, test);
   rmse90 = rmse90 + sqrt( mean((predicted - test).^2 ));
   %ee90 = ee90 + mean(1 - exp(- (predicted - test).^2/2 ./var(unlabeled) ));
       
end

mae50 = mae50/10; rmse50 = rmse50/10; pc50 = pc50/10; ee50 = ee50/10;
mae70 = mae70/10; rmse70 = rmse70/10; pc70 = pc70/10; ee70 = ee70/10;
mae90 = mae90/10; rmse90 = rmse90/10; pc90 = pc90/10; ee90 = ee90/10;



%% Errors: 

% 50/50

mae50
rmse50
pc50
% mae = 0.1524
% rmse = 0.1849
% pc = 0




% 70/30

mae70
rmse70
pc70 
% mae = 0.1513
% rmse = 0.1838
% pc = 0



% 90/10

mae90
rmse90
pc90 
% mae = 0.1540
% rmse = 0.1872
% pc = 0




