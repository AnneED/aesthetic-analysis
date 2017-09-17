function [Z, W, b] = ZWb_SemiSupervised (TrainMatrix, Labels, L, Alpha, Gamma, Mu)

D = size(TrainMatrix,1);
m=size(TrainMatrix,2);
l=length(Labels);
u=m-l;

LbMatrix=TrainMatrix(:,1:l);
[Dl, Ml] = MlMatrix(LbMatrix,Labels);

ExtMl = zeros(m);
ExtMl(1:size(Ml,2),1:size(Ml,2)) = Ml;

InvExtDl = zeros(m);
InvExtDl(1:l,1:l) = diag(diag(Dl).^-1);
InvExtDl(l+1:end, l+1:end) = 1000 * eye(u);

L2 = L + Alpha * ExtMl;

one = ones(m,1);
Im = eye(m);
Hc = Im - 1/m * (one*one');
Xc = TrainMatrix * Hc;
N = (Xc' * Xc) / (Gamma * (Xc' * Xc) + Im);

[Z, eigVal] = eig(InvExtDl * (L2 + Mu * Gamma * Hc - Mu * (Gamma^2)*N)); %Y = eigVect';

[~, idx ] = sort(real(diag(eigVal)) ,'ascend');
Z = real(Z(:,idx));

A = Gamma * (Gamma * TrainMatrix * Hc * TrainMatrix' + eye(D)) \ Xc;
W = A * Z;
b = 1/m * (Z' * one - W' * TrainMatrix * one);


end