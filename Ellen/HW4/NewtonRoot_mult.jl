# This function solves for multivariable roots of a function g
using LinearAlgebra
using ForwardDiff

function NewtonRoot_mult(g, init_guess)
"""
    #### Fields

    - 'g': a function we want to find root of
    - 'init_guess': initial guess

    #### Returns

    - 'xn1' : a root of g
"""
    
    xn1 = copy(init_guess);  
    gprime = x -> ForwardDiff.jacobian(g, x)
    xn = zeros(length(init_guess));

    tol = 1e-10;
    maxiter = 1000;
    iter = 0;
    distance = 10;
    
    while distance >= tol && iter < maxiter
        iter = iter + 1;
        xn_temp = xn1 - inv(gprime(xn1))*g(xn1);
        for i in 1:length(xn)
            xn[i] = maximum([xn_temp[i], 1e-10])
        end
        distance = norm(xn - xn1);
        xn1 = copy(xn);    
    end

    return xn1
end

## Checks
# f1(x) = log.(x) 
# NewtonRoot(f1, [10.0,10.0])