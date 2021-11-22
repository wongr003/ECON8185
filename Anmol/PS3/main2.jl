## Packages



## Utility function, and wages

l = 0.5; # for now, assume labor is exogenous
U(c::Real) = (((c^η)*(l^(1-η)))^(1-μ))/(1-μ); # Utility function
Uc(c) = ForwardDiff.derivative(U, c); # Marginal utility 
Uc_inv(y) = NewtonRoot(c -> Uc(c) - y, 1); # Inverse of marginal utility

R = 1.01;
w = 1.0;

################################### Model types #########################
struct AiyagariParameters{T <: Real}
    η::T
    μ::T
    β::T
    ρ::T
    σ::T
end

struct AiyagariModel{T <: Real, I <: Integer}
    params::AiyagariParameters{T}
    a_vals::Array{T,1} # Policy grid
    na::I
    ϵ_vals::Array{T,1} # Labor shock grid
    nϵ::I
    P_ϵ::Array{T,2} # Transition matrix for labor shock process
end

function AiyagariEGM(
    η::T = 0.3,
    μ::T = 1.5,
    β::T = 0.98,
    ρ::T = 0.6,
    σ::T = 0.3) where {T <: Real}

    params = AiyagariParameters(η,μ,β,ρ,σ)
    
    return params
end

