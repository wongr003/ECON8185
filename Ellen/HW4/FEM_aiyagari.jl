## Packages
using Parameters

################################### Create the Household instance #########################
Household = @with_kw (na = 8, # number of asset grid
    amax = 6.0, # asset max
    β = 0.96, # discount factor
    μ = 2.0, # inverse elasticity of substitution
    bc = 0, # borrowing constraint (must be weakly negative)
    lGrid = [0.2;0.9], # grid for labor process
    P_ϵ = [0.5 0.5; 0.5 0.5], # transition matrix
    nl = 2, # number of states for labor process
    aGrid = collect(LinRange(sqrt(bc), sqrt(amax), na).^2), # asset grid
    w = 1.0, # wage
    r = 0.02, # interest rate
    ζ = 50000) 
    
################################### Useful Subroutines #########################
function ψ(x::Real,X::Vector,i::Real)
"""
    - This function constructs a linear basis, which is a function of a variable 'x'

    #### Fields

    - 'x': a variable
    - 'X': a vector of the domain of x
    - 'i': order of a basis function, ψ_{i}

    #### Returns

    - ψ_{i}(x): a piecewise function
"""
    if i == 1
        return (x >= X[i] && x <= X[i+1]) ? (X[i+1]-x)/(X[i+1]-X[i]) : 0.0
    elseif i == length(X)
        return (x >= X[i-1] && x <= X[i]) ? (x-X[i-1])/(X[i]-X[i-1]) : 0.0
    else
        if x >= X[i-1] && x <= X[i]
            return (x-X[i-1])/(X[i]-X[i-1]);
        elseif x >= X[i] && x <= X[i+1]
            return (X[i+1]-x)/(X[i+1]-X[i]);
        else 
            return 0.0
        end
    end
end

function num_quad(f::Function, a::Real, b::Real, n::Int)
"""
    - This function performs numerical qaudrature using Newton-Cotes Method

    #### Fields

    - 'f': a function we want to integrate
    - 'a': lower bound
    - 'b': upper bound
    - 'n': number of intervals

    #### Returns

    - 'integral' : a numerical integration
"""
    h = (b-a)/(n-1)
    x = zeros(n)
    w = zeros(n)

    for i in 1:n
        x[i] = a + (i-1)*h
    end
    
    for i in 2:(n-1)
        w[i] = h
    end
    w[1] = h/2
    w[n] = h/2

    integral = sum([w[i]*f(x[i]) for i in 1:n]) 
    return integral
end
    
################################### Main algorithm #########################
function get_apol(hh,a,l,θ)
"""
    - This function represents capital policy function as a piecewise linear function: g^{a}(a,l;θ) = ∑θ_{i}ψ_{i}(a)

    #### Fields

    - 'hh': a household instance
    - 'a': asset 
    - 'l': labor
    - 'θ': vector of coefficients

    #### Returns

    - apol: an estimated policy function, g^{a}(a,l;θ)

"""
    @unpack aGrid,na,nl,lGrid = hh
    l_index = findall(k->k==l,lGrid) # get index of l
    if first(l_index) == 1 # first(l_index) converts 1-element vector to a float
        apol = sum([θ[i]*ψ(a,aGrid,i) for i = 1:na])
    else
        apol = sum([θ[i+na]*ψ(a,aGrid,i) for i = 1:na])
    end
    return apol
end

function get_expected(hh,a,l,θ)
"""
    - This function calculates the expectation term in Euler equation

    #### Fields

    - 'hh': a household instance
    - 'a': asset 
    - 'l': labor
    - 'θ': vector of coefficients

    #### Returns

    - expected: an expected value on the RHS of Euler

"""
    @unpack w,r,lGrid,ζ,μ = hh

    apol = get_apol(hh,a,l,θ)
    appol_L = get_apol(hh,first(apol),lGrid[1],θ) # first(apol) converts 1-element vector to a float
    appol_H = get_apol(hh,first(apol),lGrid[2],θ)

    expected = (w*lGrid[1]+(1+r)*first(apol)-first(appol_L))^(-μ) +
    (w*lGrid[2]+(1+r)*first(apol)-first(appol_H))^(-μ) +
    ζ*(min(first(apol),0)^2)
  
    return expected
end

function get_resid(hh,a,l,θ)
"""
    - This function represents the residual equation

    #### Fields

    - 'hh': a household instance
    - 'a': asset 
    - 'l': labor
    - 'θ': vector of coefficients

    #### Returns

    - resid: a residual equation, R(a,l:θ)

"""
    @unpack w,r,μ,β = hh
    apol = get_apol(hh,a,l,θ)
    resid = (w*l+(1+r)*a-first(apol))^(-μ)-((β*(1+r))/2)*get_expected(hh,a,l,θ);
    
    return resid
end

function weighted_resid(hh,θ,i)
"""
    - This function represents the numerical integration of weighted residuals with Galerkin weights: ∫ψ_{i}(a)*R(a,l;θ)da 

    #### Fields

    - 'hh': a household instance
    - 'θ': vector of coefficients
    - 'i': an integer indicates which weighted residual to compute

    #### Returns

    - w_resid: a weighted residual equation, ∫ψ_{i}(a)*R(a,l:θ)da
"""
    @unpack na,aGrid,lGrid = hh
    if i <= na
        l = lGrid[1];
        w_resid = num_quad(a->ψ(a,aGrid,i)*get_resid(hh,a,l,θ),aGrid[1],aGrid[end],100)
    else
        l = lGrid[2];
        w_resid = num_quad(a->ψ(a,aGrid,i-na)*get_resid(hh,a,l,θ),aGrid[1],aGrid[end],100)
    end
    return w_resid
end

function G(hh,θ)
"""
    - This function stacks up all weighted residuals into a matrix G

    #### Fields

    - 'hh': a household instance
    - 'θ': vector of coefficients

    #### Returns

    - 'G': a stacked system of weighted residuals
"""
    @unpack na,nl = hh
    return [weighted_resid(hh,θ,i) for i = 1:na*nl]
end




    
