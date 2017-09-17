
% Measuring time:
tic;

% Download a pre-trained CNN from the web
% urlwrite('http://www.vlfeat.org/matconvnet/models/vgg-face.mat', 'vgg-face.mat') ;


% Setup MatConvNet.
cd /home/john-san/Dropbox/Master/TFM/My_scripts/matconvnet-1.0-beta23;
run matlab/vl_setupnn;

% Load the model and upgrade it to MatConvNet current version.
net = load('vgg-face.mat');
net = vl_simplenn_tidy(net);

cd /home/john-san/Dropbox/Master/TFM/My_scripts/Correct_face_order/Feature_extraction;

% Set up image data
%imds_M2B = imageDatastore('/home/john-san/Dropbox/Master/TFM/Data/M2B/Faces128', 'LabelSource', 'foldernames');
%imds_SCUT_FBP = imageDatastore('/home/john-san/Dropbox/Master/TFM/Data/SCUT-FBP/Data_Collection', 'LabelSource', 'foldernames');
%FileNames_M2B = imds_M2B.Files;
%FileNames_SCUT_FBP = imds_SCUT_FBP.Files;

FileNames_SCUT = strings(500, 1);

for i = 1:500
    FileNames_SCUT(i) = strcat('/home/john-san/Dropbox/Master/TFM/Data/SCUT-FBP/Data_Collection/SCUT-FBP-', num2str(i), '.jpg');
end

FileNames_M2B_East = strings(620, 1);
FileNames_M2B_West = strings(620, 1);

for i = 1:620
    FileNames_M2B_East(i) = strcat('/home/john-san/Dropbox/Master/TFM/Data/M2B/Faces128/eface_', num2str(i), '.png');   
    FileNames_M2B_West(i) = strcat('/home/john-san/Dropbox/Master/TFM/Data/M2B/Faces128/wface_', num2str(i), '.png');
end


% Data matrices
X_vgg_6_M2Be = [];
X_vgg_7_M2Be = [];
X_vgg_6_M2Bw = [];
X_vgg_7_M2Bw = [];
X_vgg_6_SCUT = [];
X_vgg_7_SCUT = [];


% We go through all the images in M2B
for k = 1:length( FileNames_M2B_East )

% Eastern faces    
    data_M2Be_vgg{k}.filename = FileNames_M2B_East(k);

    % Pre-processing
    im = imread( char(FileNames_M2B_East(k)) );
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(net, im_);
    % Store features of the layer previous to the classification
    Features_6 = squeeze(gather(res(end-4).x));
    Features_7 = squeeze(gather(res(end-2).x));
    
    % Save the features in the cellarray:
    data_M2Be_vgg{k}.fc6 = Features_6; 
    data_M2Be_vgg{k}.fc7 = Features_7;

    % Creation of the data matrices (each sample is a column):
    X_vgg_6_M2Be = [X_vgg_6_M2Be Features_6];
    X_vgg_7_M2Be = [X_vgg_7_M2Be Features_7];



% Western faces
    data_M2Bw_vgg{k}.filename = FileNames_M2B_West(k);        

    % Pre-processing
    im = imread( char(FileNames_M2B_West(k)) );
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(net, im_);
    % Store features of the layer previous to the classification
    Features_6 = squeeze(gather(res(end-4).x));
    Features_7 = squeeze(gather(res(end-2).x));
    
    % Save the features in the cellarray:
    data_M2Bw_vgg{k}.fc6 = Features_6; 
    data_M2Bw_vgg{k}.fc7 = Features_7;

    % Creation of the data matrices (each sample is a column):
    X_vgg_6_M2Bw = [X_vgg_6_M2Bw Features_6];
    X_vgg_7_M2Bw = [X_vgg_7_M2Bw Features_7];


end


% We go through all the images in SCUT-FBP
for k = 1:length( FileNames_SCUT )
    
    data_SCUT_vgg{k}.filename = FileNames_SCUT(k);

    % Pre-processing
    im = imread( char(FileNames_SCUT(k)) );
    im_ = single(im);
    im_ = imresize(im_, net.meta.normalization.imageSize(1:2));
    im_ = im_ - net.meta.normalization.averageImage;
    % Network pass
    res = vl_simplenn(net, im_);
    % Store features of the layer previous to the classification
    Features_6 = squeeze(gather(res(end-4).x));
    Features_7 = squeeze(gather(res(end-2).x));
    
    % Save the features in the cellarray:
    data_SCUT_vgg{k}.fc6 = Features_6; 
    data_SCUT_vgg{k}.fc7 = Features_7;

    % Creation of the data matrices (each sample is a column):
    X_vgg_6_SCUT = [X_vgg_6_SCUT Features_6];
    X_vgg_7_SCUT = [X_vgg_7_SCUT Features_7];

    
end


% Saving features

%cd /home/john-san/Dropbox/Master/TFM/My_scripts;
save('vgg_features.mat', 'data_SCUT_vgg', 'data_M2Bw_vgg', 'data_M2Be_vgg');
toc;


% Saving matrices
save('matrices_vgg.mat', 'X_vgg_6_M2Be', 'X_vgg_7_M2Be', 'X_vgg_6_M2Bw', 'X_vgg_7_M2Bw', 'X_vgg_6_SCUT', 'X_vgg_7_SCUT');




% MAT-file version 7.3 is used to store variables larger than 2GB.

