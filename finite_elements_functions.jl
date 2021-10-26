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
    elseif i == 1 # i is in the boundary(1)
        if X[i] <= x <= X[i+1]
            f = (X[i + 1] - x) / (X[i + 1] - X[i]);
        else
            f = 0;
        end
    elseif i == length(X) # i is in the top boundary
        if X[i - 1] <= x <= X[i]
            f = (x - X[i - 1]) / (X[i] - X[i - 1]);
        else
            f = 0
        end
    end
    return f
end

# Defining consumption approximate function
function cn(k, α, K)
    n = length(K);
    c = 0;
    for i = 1:n
        c = c + α[i] * ψi(k, K, i);
    end
    return c
end

function integra(k, α, K)
#This function calculates the function that will be integrated:
#integra(k;α):= ϕi(k)R(k;α), where ϕi are the weights and R is the residual function
#In the Galerkin methods, the weights are the same as the approximating functions
    T = zeros(length(K));
    for i = 1:length(K)
        T[i] = ψi(k, K, i) * residual(k, α, K);
    end
    return T
end

#This function calculates the integral (the norm of the integrated functions), as a functions of the parameters to minimized
#We define that way since this is the format accepted by the solver:
#mini(α):= ∫integra(k;α)dk
nodes, weights = gausslegendre(3*(length(K)-1)) #Gauss Legendre nodes and weights,this function is just a Quadrature table
function mini(α;nodes=nodes,weights=weights,K=K)
    if length(α)<length(K)
        α = vcat(0,α)
    end
    #g = quadgk.(integra,K[1],K[end])[1] #Integral
    #See Judd's book pg 261 on numerical integration and the gausslegendre formula:
    gaussleg = zeros(length(K))
    for j=1:length(nodes)
        gaussleg .+= (K[end]-K[1])/2 .* weights[j] .* integra((nodes[j] .+1).*
        (K[end]-K[1])/2 .+ K[1],α, K)
    end
    return norm(gaussleg,1)
end

