function [W] = Cos_GraphConstruction4(data, epsilon, scores, devs, K)
% This function creates the similarity matrix combining the cosine feature similarity and the score similarity. The cosine similarity is rescaled so that it is a positive number ranging from 0 to 1.


% INPUT:
%
% data: data matrix. Columns represent samples and rows features.
%
% epsilon: To compute the similarity of a given image with the
% others only images which have a difference in the score less or equal
% to epsilon are considered.
%
% scores: vector with the (mean) scores of the images. Unlabeled images
% will have a 0.
%
% devs: standard deviation of the scores of the images.
%
% OUTPUT:
%
% L: Laplacian matrix of the data based on cosine similarity and score
% similarity.


if nargin < 5
  K = 10;
end


% Calculate cosine similarity between samples:

norm_val = sqrt(sum(data.^2));
Nmat = norm_val' * norm_val;
Dotmat = data' * data;
MatSim = Dotmat./Nmat;    % cosine similarity
MatSim = (1 + MatSim)/2;

nsmp = size(data, 2);

% Remove all the edges between faces with very different scores
for i = 1:nsmp
    for j = i:nsmp
        if scores(i) ~= 0 & scores(j) ~= 0
            if abs(scores(i) - scores(j)) > epsilon
                MatSim(i, j) = 0;
                MatSim(j, i) = 0;
            end
        end
    end
end

M_distance = 1 - MatSim;

%% Sort samples acording to their pairwise distance
[~, idx] = sort(M_distance,2);

%Select the K nearest samples
M_Crit_dir = zeros(size(data,2));

Sel_NN = idx( : , 2:K );
for i=1:size(Sel_NN,1)
    M_Crit_dir( i , Sel_NN(i,:)) = 1;
end

%calculate the symmetric criteria
M_Crit_ind = max(M_Crit_dir,M_Crit_dir');%make it symmetric(indirect)

%calculate the adjacency graph
W = MatSim .* M_Crit_ind;



% L = diag(sum(W)) - W;




