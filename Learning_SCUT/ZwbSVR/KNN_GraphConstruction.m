function [W_dir W_ind]= KNN_GraphConstruction(data,K)
%% construct adjacency matrix based on fix neighbourhood size K
% 
%    Inputs :: 
%             data -> a DxN matrix containing the column vector features
%             K    -> Neighbourhood size, Default = 2
%             
%    Outputs ::
%             W_ind -> Indirected symmetric adjacency graph
%             W_dir -> Directed graph(not necesarily symetric)
% 
% 
%     Comments ::
%             the features are normalized such that their l2 norm is equal one
%             the weight between nodes is set by the gaussian fuction
%             exp(||x(i)-x(j)||/2*sigma^2)  and the sigma is average of the
%             squared distance 
% 
% V1.0 20 Feb 2014
%  

if nargin < 2
    K=2;
    if nargin < 1
        error('No input data!!')
    end
end

%% Normalizing data
norm_val = sqrt(sum(data.^2));
data= data ./ repmat(norm_val,size(data,1),1);

%%
M_Crit_dir = zeros(size(data,2));

%% Calculate similarity between samples
[ ~ , nSmp] = size(data);
aa = sum(data .* data);
ab = data' * data;
M_distance =      repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
M_distance (abs(M_distance )<1e-10)=0;
sigma2       =   mean( M_distance(:));
MatSim      =   exp( -(M_distance) / (2*sigma2) );
clear aa ab sigma2 

%% Sort samples acording to their pairwise distance
[~, idx] = sort(M_distance,2);

%Select the K nearest samples
Sel_NN = idx( : , 2:K );
for i=1:size(Sel_NN,1)
    M_Crit_dir( i , Sel_NN(i,:)) = 1;
end

%calculate the symmetric criteria
M_Crit_ind = max(M_Crit_dir,M_Crit_dir');%make it symmetric(indirect)

%calculate the adjacency graph
W_dir = MatSim .* M_Crit_dir;
W_ind = MatSim .* M_Crit_ind;

