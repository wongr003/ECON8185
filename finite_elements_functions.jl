# This script
using Optim, LinearAlgebra, FastGaussQuadrature

# Defining element here to test code
K = zeros(15)
for i=2:length(K)
    global K
    K[i] = K[i-1] +0.0005*exp(0.574*(i-2))
end

polk(k,α) = min(max(eps(),A*k.^θ+(1-δ)*k-cn(k,α)),K[end])

function ψi(x, X, i::Int)
    if i > 1 && i < length(X)
        if X[i-1] <= x <= X[i] # i is not in a boundary
            f = (x - X[i - 1]) / (X[i] - X[i - 1]);
        elseif X[i] <= x <= X[i + 1]
            f = (X[i + 1] - x) / (X[i + 1] - X[i]);
        else 
            f = 0
        end
    end
end
