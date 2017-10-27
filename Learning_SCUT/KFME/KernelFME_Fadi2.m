function [F, Alphas] = KernelFME_Fadi2(X, labels, labeled_mask, Beta, Gamma, Mu, T0)

% Calculate kernel
Ker = KernelMatrix(X, X, T0);

% Store number of samples
m = size(X,2);

% Calculate Laplacian
[~, Lt] = W_KNN_Infunc(X, 10, 10*exp(-1));

% Target labels (now real numbers)
Y = zeros(m, 1);    %%% Modified to work on regression
Y( labeled_mask ) = labels( labeled_mask );      %%% Modified to work on regression

% Create U matrix using labeled mask
U = single( diag( labeled_mask ) );

% Identity matrix
Im = eye(m);

% Solve analitically KFME
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
