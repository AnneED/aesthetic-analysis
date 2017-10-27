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
