# Since we'll be working with economic varible which are all positive 
# We'll modify this algorithm to limit search for root only in the postive quadrant.
using Calculus

function NewtonRoot(g, init_guess)
    
    xn1 = init_guess;  
    tol = 10^(-8);
    distance = 1;
    iter = 1;

    while distance >= tol
        xn = xn1;
        xn1 = xn - g(xn) / derivative(g, xn);
        xn1 = max(xn1, 0.0001);

        distance = abs(xn1 - xn);
        iter = iter + 1;
    end

    return xn1
end

## Checks
# f1(x) = log(x) 
# NewtonRoot(f1, 10)