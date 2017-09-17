function [F, Alphas] = KernelFME_Fadi(X, labels, labeled_mask, Beta, Gamma, Mu, T0) 

Ker = KernelMatrix(X, X, T0);
LbMatrix.fea = X(:,labeled_mask);
LbMatrix.gnd = labels(labeled_mask);
UnlbMatrix.fea = X(:,~labeled_mask);
UnlbMatrix.gnd = labels(~labeled_mask);



X = [LbMatrix.fea, UnlbMatrix.fea];
n = size(LbMatrix.fea,2);
m = size(X,2);
labeled_mask(1:n) = 1;
labeled_mask(n+1:m) = 0;
UnLabeled_loc = labeled_mask == 0;
Labeled_loc   = labeled_mask == 1;
X_U =  X(: , UnLabeled_loc);
X_L =  X(: , Labeled_loc  );
X_T    = [X_L X_U];
[~, Lt] = W_KNN_Infunc(X_T,10, 10*exp(-1));
c = size(unique(LbMatrix.gnd),2);
Y=zeros(m ,c);
%for i = 1 : n
%    Y(i, LbMatrix.gnd(i)) = 1;
%end     %%%% Modified to work on regression
Y = zeros(m, 1);    %%% Modified to work on regression
Y(1:n) = labels(Labeled_loc);      %%% Modified to work on regression

U = zeros(m);
for i = 1 : n 
    U(i,i) = 1;
end

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
