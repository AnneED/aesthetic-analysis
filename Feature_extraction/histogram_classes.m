
load('initial_data_M2Be_vgg.mat');
load('initial_data_M2Bw_vgg.mat');
load('initial_data_SCUT_vgg.mat');

figure, hist(classes, [1 2 3 4 5])
title('Classes of SCUT-FBP')

figure, hist(classes_e, [1 2 3 4 5])
title('Classes of easterners in M2B')

figure, hist(classes_w, [1 2 3 4 5])
title('Classes of westerners in M2B')
