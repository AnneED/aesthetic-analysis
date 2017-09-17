function [F] = LGC(X, W, Mu, Y)

% Input:
%       X: data matrix (rows are features and columns are samples).
%       Y: column vector with labels (unlabeled samples have a 0).
%       W: similarity matrix.
%       Mu: parameter pondering the fitness criterion.

% Output:
%       F: predicted labels.

[~, nSmp] = size(W);
D = diag(sum(W).^(-1/2));
L = eye(nSmp) - D * W * D;
F = (eye(nSmp) + L/Mu)\Y;
% A\B "=" inv(A)*B

end

