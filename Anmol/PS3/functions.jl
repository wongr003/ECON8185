## Packages
using Parameters
using Interpolations
using Plots
using LinearAlgebra
using BenchmarkTools

## Files
include("Tauchen.jl")

################################### Create the Household instance #########################
Household = @with_kw (na = 10, # number of asset grid
    amax = 40.0, # asset max
    β = 0.98, # discount factor
    θ = 0.3, #capital share 
    δ = 0.075, # depreciation rate
    γ = 1, # elasticity of substitution
    bc = 0, # borrowing constraint (must be weakly negative)
    ρ = 0.6, # autocorr of income process
    nϵ = 5, # number of states for income process
    σ = 0.3, # stand. dev. of income process
    μ_ϵ = 0.0, # mean of income process
    ϵGrid = exp.(Tauchen(μ_ϵ, ρ, σ, nϵ)[1]), # grid for income process
    P_ϵ = Tauchen(μ_ϵ, ρ, σ, nϵ)[2], # transition matrix
    Amat = repeat(collect(LinRange(sqrt(bc), sqrt(amax), na)).^2,1,nϵ), # asset grid
    Ymat = repeat(ϵGrid',na,1)) # income grid

################################### Useful functions used in main EGM function #########################
# Marginal utilities and their Inverse
Uc(c,γ) = c.^(-γ);
invUc(x,γ) = x.^(-1/γ);

# Euler equation to get current consumption, given future interest rate and future consumption grid
function getc(γ,β,P_ϵ;rnext,cnext)
    Ucnext = β.*(1+rnext).*Uc(cnext,γ)*P_ϵ'; # future marginal utility
    c = invUc(Ucnext, γ); # current consumption
    return c
end

# Obtain current assets, given consumption today defined on asset grid tomorrow
geta(Amat,Ymat;r,w,c) = 1/(1+r).*(c.+Amat.-w.*Ymat)

################################### Main EGM function #########################
# This is the main EGM function. It iterates on the Euler equation c_{t} = β*(1+r_{t+1})*E_{t}[c_{t+1}],
# given a guess for the consumption policy function, c_{t+1} (cnext in the function).
# Note that we no longer need a root finding procedure, but still need to interpolate the optimal policy 
# on our defined grid
function egm(hh;w,cnext,cbinding,r,rnext)
"""
    use endogenous grid method to obtain c_{t} and a_{t} given c_{t+1} 'cnext'

    #### Fields

    - 'hh': household tuple
    - 'w': wage rate
    - 'cnext': time t+1 consumption grid
    - 'cbinding': consumption grid when borrowing constraint binds
    - 'r': interest rate at time t
    - 'rnext': interest rate at time t+1

    #### Returns
    - 'c': time t consumption grid
    - 'anext': time t policy function for saving
"""

    @unpack γ,β,P_ϵ,Amat,Ymat,nϵ = hh

    # Current policy functions on current grid
    c = getc(γ,β,P_ϵ;rnext = rnext, cnext = cnext);
    a = geta(Amat,Ymat;r=r,w=w,c=c);

    cnonbinding = similar(Amat);

    # Interpolate consumption policy function for current grid
    for i = 1:nϵ
        cnonbinding[:,i] = LinearInterpolation(a[:,i], c[:,i], extrapolation_bc = Line()).(Amat[:,i]);
    end

    # update elements of consumption policy when borrowing constraint binds
    # a[1,j] is the level of current assets that induces the borrowing constraint to bind exactly.
    # Therefore, whenever current assets are below a[1,j], the borrowing constraint will be STRICTLY binding.
    # Note that this uses the monotonicity of the policy rule.

    for j = 1:nϵ
        c[:,j] = (Amat[:,j] .> a[1,j]) .*cnonbinding[:,j] .+ (Amat[:,j] .<= a[1,j]).*cbinding[:,j];
    end
        
    # update saving policy function with new consumption function
    anext = @. (1+r)*Amat+w*Ymat-c;
 
    return c,anext
end

# This is the function that iterates on the EGM function above to solve for the optimal policy rule.
function iterate_egm(hh;r,tol=1e-8,maxiter=1000)
"""
    iterates on EGM method until c converged

    #### Fields

    - 'hh': household tuple
    - 'r': interest rate 

    #### Returns
    - 'c': policy function for consumption, given r
    - 'anext': policy function for saving, given r
"""
    
    @unpack δ,θ,Amat,Ymat,bc = hh

    A = ((r+δ)/θ)^θ; # normalize aggregate income so that y = Y/N =1
    w = A*(1-θ)*((r+δ)/(A*θ))^(θ/(θ-1)); # wage rate given guess for r

    cnext = @. r*Amat+w*Ymat; # initial guess for policy function iteration

    cbinding = @. (1+r)*Amat+w*Ymat-bc; # get consumption when borrowing constraint binds

    iter = 1;

    for i = 1:maxiter
        #println("iteration: $i"," ")
        c = egm(hh;w=w,rnext=r,r=r,cnext=cnext,cbinding=cbinding)[1];
        if norm(c-cnext,Inf)<tol
            #println("Solved for policy functions in $i iterations")
            return egm(hh;w=w,rnext=r,r=r,cnext=cnext,cbinding=cbinding)
        else
            cnext=c;
            iter = iter+1;
        end
    end

    error("Does not converged")
end

################################### Equilibrium #########################
# This function solves for the market clearing interest rate in the Aiyagari model. It uses the EGM routines for the 
# policy functions and Young's method for the distribution, as well as a bisection procedure to obtain interest rate.
function MakeTransMat(hh,apol)
"""
    constructs transition matrix Q for asset-skill distribution using Young's methood

    #### Fields

    - 'hh': household tuple
    - 'apol': policy function for savings, array na × nϵ

    #### Returns
    - 'Qmat': (na*nϵ)×(na*nϵ) array 
"""

    @unpack na,nϵ,Amat,P_ϵ = hh     
    
    # preallocate memory for the transition matrix
    Qmat = zeros(na*nϵ,na*nϵ);
    
    for i = 1:na
        for j = 1:nϵ
            aStar = apol[i,j];
            l = searchsortedlast(Amat[:,j],aStar);

            (l > 0 && l < na) ? l = l :
            (l == na) ? l = na-1 :
            l = 1

            p = (aStar-Amat[l,j])/(Amat[l+1,j]-Amat[l,j]);
            p = min(max(p,0.0),1.0);

            sj = (j-1)*na;
            for k = 1:nϵ
                sk = (k-1)*na;
                Qmat[sk+l,sj+i] = (1-p)*P_ϵ[k,j];
                Qmat[sk+l+1,sj+i] = p*P_ϵ[k,j];
            end
        end
    end

    return Qmat
end

# This functions solves for the stationary distribution of assets given the matrix Q
function StationaryDistribution(hh,Qmat,tol=1e-10,maxiter=1000)
"""
    iterate on λ_{j+1} = Q*λ_{j} until converged

    #### Fields

    - 'hh': household tuple
    - 'Qmat': transition matrix, (na*nϵ)×(na*nϵ)

    #### Returns
    - 'λCurrent': array (na*nϵ) of stationary distribution of people 
"""

    @unpack na,nϵ = hh

    λCurrent = ones(na*nϵ);
    λCurrent = λCurrent./sum(λCurrent);

    for i = 1:maxiter
        λnext = Qmat*λCurrent
        if (i-1) % 50 == 0
            test = maximum(abs.(λCurrent - λnext));
            if test < tol
                #println("Solved in $i iterations")
                break
            end
        end
        λCurrent = copy(λnext);
    end

    return λCurrent
end

# This function computes steady state of aggregate capital supply
function Kss(hh;r)
 """
    compute aggregate supply of capital: A(r,w) = ∫a'(a,ϵ;r,w)dλ(a,ϵ;r) = ā' ⋅ λ̄

    #### Fields

    - 'hh': household tuple
    - 'r': interest rate

    #### Returns
    - 'K': aggregate supply of capital 
 """
    @unpack β = hh
    @assert r < 1/β-1 "r too large for convergence"

    apol = iterate_egm(hh;r=r)[2]; # get converged policy function for savings
    Qmat = MakeTransMat(hh,apol); # get transition matrix
    λ = StationaryDistribution(hh,Qmat); # get invariant distribution

    K = reshape(apol,(length(λ),1));    
    K = sum(K.*λ);

    return K
end

# This function computes r that clears the market
function market_clearing(hh,Z;r=0.015,tol=1e-5,maxiter=20,bisection_param=0.8)
"""
    bisection procedure until r converged.

    #### Fields

    - 'hh': household tuple

    #### Returns
    - 'r': equilibrium interest rate
 """

    @unpack θ,δ = hh

    for iter = 1:maxiter
        println("r=$r: ")
        Ksupply,λ = Kss(hh;r=r);

        #Z = ((r+δ)/θ)^θ;
        rsupply = Z*θ*Ksupply^(θ-1) - δ;
        
        if abs(r-rsupply)<tol
            return (r+rsupply)/2
        else
            r = (bisection_param)*r+(1-bisection_param)*rsupply
        end
    end

    error("no convergence: did not find market clearing interest rate")
end

# This function plots market clearing graph
function plot_market_clearing(hh)
"""
    Aiyagari's classic picture

    #### Fields

    - 'hh': household tuple

    #### Returns
    - the plot
 """
    
    @unpack θ, δ = hh
    
    rgrid = 0.005:0.002:0.02
    Ksupply = zeros(length(rgrid))
    
    for (index,r) in enumerate(rgrid)
        Ksupply[index] = Kss(hh; r=r)
    end
    
    plot(Ksupply,rgrid,label = "capital supply")

    A(r) = ((r+δ)/θ)^θ; # normalize so y = 1
    K(r) = ((r+δ)/(A(r)*θ))^(1/(θ-1));
    
    plot!(K.(rgrid),rgrid,label= "capital demand")
    xlabel!("capital")
    ylabel!("r")

end
