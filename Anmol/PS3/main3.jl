## Packages
using ForwardDiff
using UnPack

## Files
include("Tauchen.jl")
include("NewtonRoot.jl")

## Interest Rate and Wage (exogenously determined for now)
R = 1.01;
w = 1.0;

################################### Model types #########################
mutable struct AiyagariParameters{T <: Real}
    η::T
    μ::T
    β::T
    ρ::T
    σ::T
    μ_ϵ::T
end

mutable struct AiyagariModel{T <: Real, I <: Integer}
    params::AiyagariParameters{T}
    aGrid::Array{T,1} # Policy grid
    aGridl::Array{T,1} # Grid for a with different ϵ stacked up
    na::I
    ϵGrid::Array{T,1} # Labor shock grid
    nϵ::I
    P_ϵ::Array{T,2} # Transition matrix for labor shock process
end

function AiyagariEGM(
    η::T = 0.3,
    μ::T = 1.5,
    β::T = 0.98,
    ρ::T = 0.6,
    σ::T = 0.3,
    μ_ϵ::T = 0.0,
    amin::T = 1e-9,
    amax::T = 18.0,
    na::I = 5,
    nϵ::I = 2) where {T <: Real, I <: Integer}

    ########### Parameters
    params = AiyagariParameters(η,μ,β,ρ,σ,μ_ϵ)

    ########### Policy grid
    aGrid = collect(LinRange(amin, amax, na));
    aGridl = repeat(aGrid, nϵ)

    ########### Transition
    ϵGrid, P_ϵ = Tauchen(μ_ϵ, ρ, σ, nϵ);
    ϵGrid = exp.(collect(ϵGrid));

    cGuess = R*aGridl + w*repeat(ϵGrid, inner = na)
    
    return AiyagariModel(params,aGrid,aGridl,na,ϵGrid,nϵ,P_ϵ), cGuess
end

function interpC(cbar::AbstractArray,
                 abar::AbstractArray,
                 a_tmr::T,
                 ϵ::T,
                 na::Integer) where {T <: Real}

    if a_tmr <= abar[1]
        cbar_interp = R*a_tmr + w*ϵ;
    else
        np = searchsortedlast(abar,a_tmr);

        (np < na) ? np = np :
        np = na-1

        abar_l, abar_h = abar[np], abar[np+1];
        cbar_l, cbar_h = cbar[np], cbar[np+1];
        cbar_interp = cbar_l+(cbar_h-cbar_l)/(abar_h-abar_l)*(a_tmr-abar_l); # y = y0 + (y1-y0)/(x1-x0)*(x-x0)
    end

    return cbar_interp
end

function updateC(model::AiyagariModel,
                 cbar::AbstractArray,
                 abar::AbstractArray,
                 cbar_interp::AbstractArray)

    @unpack na,nϵ,aGridl,ϵGrid = model
    cbar = reshape(cbar,na,nϵ);
    abar = reshape(abar,na,nϵ);

    for ϵi = 1:nϵ
        for ai = 1:na
            aϵi = (ϵi-1)*na+ai
            cbar_interp[aϵi] = interpC(cbar[:,ϵi],abar[:,ϵi],aGridl[aϵi],ϵGrid[ϵi],na) 
        end
    end

    return cbar_interp
end


function EGM(model::AiyagariModel,
             cpol::AbstractArray,
             abar::AbstractArray)

    @unpack params,na,nϵ,aGridl,P_ϵ,ϵGrid = model
    @unpack β,η,μ = params

    U(c) = (((c^η)*(0.5^(1-η)))^(1-μ))/(1-μ); # Utility function, for now, assume labor is exogenous
    Uc(c) = ForwardDiff.derivative(U, c); # Marginal utility 
    Uc_inv(y) = NewtonRoot(c -> Uc(c) - y, 1); # Inverse of marginal utility

    Uc_cj = Uc.(cpol)
    Uc_cj = reshape(Uc_cj, na, nϵ);

    EUc = (P_ϵ*Uc_cj')';
    EUc = reshape(EUc, na*nϵ);
    cbar = Uc_inv.(R*β*EUc);

    # Compute abar
    for ai = 1:na
        for ϵi = 1:nϵ
            aϵi = (ϵi-1)*na+ai
            abar[aϵi] = (cbar[aϵi]+aGridl[aϵi]-w*ϵGrid[ϵi])/R
        end
    end

    cbar_interp = zeros(length(cbar))
    cbar_interp = updateC(model,cbar,abar,cbar_interp)

    return cbar_interp
end

function SolveEGM(model::AiyagariModel,
                  cCurrent::AbstractArray,
                  abar::AbstractArray,
                  tol = 1e-10)

    @unpack na,aGridl,nϵ,ϵGrid = model

    for i = 1:10000
        cNext = EGM(model, cCurrent, abar)
        if (i-1) % 50 == 0
            test = maximum(abs.(cCurrent - cNext));
            println("iteration: $i"," ")
            if test < tol
                println("Solved in $i iterations")
                break
            end
        end
        cCurrent = copy(cNext);
    end

    apol = zeros(na*nϵ)
    for ai = 1:na
        for ϵi = 1:nϵ
            aϵi = (ϵi-1)*na+ai
            apol[aϵi] = R*aGridl[aϵi] + w*ϵGrid[ϵi] - cCurrent[aϵi] 
        end
    end

    return cCurrent, apol
end

## Test Code ##
AA0, cpol0 = AiyagariEGM()
abar0 = zeros(length(cpol0))
#cbar_test, abar_test = EGM(AA0, cpol0, abar0)
cpol_EGM, apol_EGM = SolveEGM(AA0,cpol0,abar0)
