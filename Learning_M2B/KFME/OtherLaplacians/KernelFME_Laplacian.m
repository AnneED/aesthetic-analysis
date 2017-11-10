function [F, Alphas] = KernelFME_Laplacian(X, labels, labeled_mask, Beta, Gamma, Mu, T0, Lt) 

% Laplacian has to be provided

% Calculate kernel
Ker = KernelMatrix(X, X, T0);

% Store number of samples
m = size(X,2);

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






