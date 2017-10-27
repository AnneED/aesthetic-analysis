function [F, Alphas] = KernelFME_Fadi(X, labels, labeled_mask, Beta, Gamma, Mu, T0) 

% Create kernel matrix
Ker = KernelMatrix(X, X, T0);

% Store feature and ground truth for labeled and
% unlabeled samples
LbMatrix.fea = X(:,labeled_mask);
LbMatrix.gnd = labels(labeled_mask);
UnlbMatrix.fea = X(:,~labeled_mask);
UnlbMatrix.gnd = labels(~labeled_mask);

% Place labeled samples followed by unlabeled ones
% in variable X
X = [LbMatrix.fea, UnlbMatrix.fea];
% number of labeled samples
n = size(LbMatrix.fea,2);
% total number of samples
m = size(X,2);
% adjust mask to new sample placement in matrix X
labeled_mask(1:n) = 1;
labeled_mask(n+1:m) = 0;
% sample locations
UnLabeled_loc = labeled_mask == 0;
Labeled_loc   = labeled_mask == 1;
% unlabeled samples matrix
X_U =  X(: , UnLabeled_loc);
% labeled samples matrix
X_L =  X(: , Labeled_loc  );
% concatenated labeled and unlabeled matrices
X_T    = [X_L X_U];
% calculate Laplacian
[~, Lt] = W_KNN_Infunc(X_T,10, 10*exp(-1));
% number of classes
c = size(unique(LbMatrix.gnd),2);
% matrix of sample classes (number of samples x number of classes)
Y=zeros(m ,c);
for i = 1 : n
    % set class position to 1 for labeled samples
    Y(i, LbMatrix.gnd(i)) = 1;
end
% indicator matrix (matrix indicating which samples are labeled)
U = zeros(m);
for i = 1 : n 
    U(i,i) = 1;
end

% identity matrix
Im = eye(m);

A = Gamma / Mu * inv(Im + Gamma / Mu * Ker);

F = Beta * (Beta * U + Lt + Gamma * (Ker * A - Im)' * (Ker * A - Im) + Mu * A' * Ker * A) \ U * Y;

Alphas = A * F;

end


function K = KernelMatrix(X,Y,T0)

s2 = Sigma2([X, Y]);
K = zeros(size(X,2),size(Y,2));
for i = 1:size(X,2)
    for j = 1:size(Y,2)
        K(i,j) = KernelFunction(X(:,i),Y(:,j), T0, s2);
    end
end
end


function k = KernelFunction(x, y, T0, s2)
k = exp(-1/( (2^T0) * s2) * norm(x-y)^2);
end

function s2 = Sigma2(X)
nSmp = size(X,2);
aa = sum(X .* X);
ab = X' * X;
M_distance = repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
M_distance(abs(M_distance )<1e-10)=0;
s2 = mean((M_distance(:)));
end


function [W, L] = W_KNN_Infunc(data,K,ks)
% construct W matrix based on fix neighbourhood size


% Normalizing data
norm_val = sqrt(sum(data.^2));
data= data ./ repmat(norm_val,size(data,1),1);

%
% K=40;
M_Crit = zeros(size(data,2));

% Calculate distance and similarity
[ ~ , nSmp] = size(data);
aa = sum(data .* data);
ab = data' * data;
M_distance =      repmat(aa',1,nSmp) + repmat(aa, nSmp,1) - 2*ab';
M_distance (abs(M_distance )<1e-10)=0;
clear aa ab

sigma2       =  - mean( M_distance(:)) / log(ks/K) ; % eqution 42 article SODA L normalise
MatSim      =   exp( -(M_distance) / (2*sigma2) );


[~, idx] = sort(M_distance,2);
K = K + 1 ;%% because of the image itself
Sel_NN = idx( : , 1:K );
for i=1:size(Sel_NN,1)
    M_Crit( i , Sel_NN(i,:)) = 1;
end
M_Crit = max(M_Crit,M_Crit');%make it symmetric

W = MatSim .* M_Crit; %
W = (1-eye(size(W))) .* W;
% W  =( W + W') ;

L = diag(sum(W)) - W ;

end