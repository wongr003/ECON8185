## Packages
using ForwardDiff
using UnPack

## Files
include("Tauchen.jl")
include("NewtonRoot.jl")


## Wages
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

    ########### Transition
    ϵGrid, P_ϵ = Tauchen(μ_ϵ, ρ, σ, nϵ);
    ϵGrid = exp.(collect(ϵGrid));

    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

    return AiyagariModel(params,aGrid,vcat(aGrid,aGrid),na,ϵGrid,nϵ,P_ϵ), guess
end

function interpEGM(apol::AbstractArray,
                aGrid::AbstractArray,
                x::T,
                na::Integer) where{T <: Real}
    ## This function interpolates a function f(x) = y (which is called aGrid) on grid of x (which is called apol)
    ## Inputs: apol -- vector of abar, aGrid -- grid on future asset a', x -- interested a', na -- number of points on aGrid
    ## Outputs: above*np -- interpolated a', on aGrid, set to zero if this interpolated value less than first apol 

    np = searchsortedlast(apol,x)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np : 
        (np == na) ? np = na-1 : 
            np = 1        
    
    ap_l,ap_h = apol[np],apol[np+1]
    a_l,a_h = aGrid[np], aGrid[np+1] 
    ap = a_l + (a_h-a_l)/(ap_h-ap_l)*(x-ap_l) # y = y0 + (y1-y0)/(x1-x0)*(x-x0) 
    
    above =  ap > apol[1] 
    return above*ap,np
end

#apol_test = LinRange(21.0, 38.0, 18);
#aGrid_test = LinRange(11.0, 28.0, 18);
#x_test = 20.5;
#na_test = length(aGrid_test);
#interpEGM(apol_test,aGrid_test,x_test,na_test)


function get_cEGM(pol::AbstractArray,
               CurrentAssets::AbstractArray,
               AiyagariModel::AiyagariModel,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,nϵ,ϵGrid = AiyagariModel
    pol = reshape(pol,na,nϵ)
    aprime = zeros(na*nϵ)
    for ϵi = 1:nϵ
        for ai = 1:na
            aϵi = (ϵi - 1)*na + ai
            aprime[aϵi] = interpEGM(pol[:,ϵi],aGrid,CurrentAssets[aϵi],na)[1]
            cpol[aϵi] = R*CurrentAssets[aϵi] + w*ϵGrid[ϵi] - aprime[aϵi]
        end
    end
    return cpol, aprime
end

apol_test = collect(LinRange(10.0, 36.0, 10))
aGrid_test = vcat(collect(LinRange(1e-9, 18.0, 5)), collect(LinRange(1e-9, 18.0, 5)))

AiyagariModel_test, pol0_test = AiyagariEGM()
cpol_test = zeros(eltype(pol0),5*2)
new_cpol, aprime = get_cEGM(apol_test, aGrid_test, AiyagariModel_test, cpol_test)


function EulerBackEGM(pol::AbstractArray,
    AiyagariModel::AiyagariModel,
    cpol::AbstractArray,
    abar::AbstractArray)

    @unpack params,na,nϵ,aGridl,P_ϵ,ϵGrid = AiyagariModel
    @unpack β,η,μ = params
    cj = get_cEGM(pol,aGridl,AiyagariModel,cpol)

    U(c) = (((c^η)*(0.5^(1-η)))^(1-μ))/(1-μ); # Utility function, for now, assume labor is exogenous
    Uc(c) = ForwardDiff.derivative(U, c); # Marginal utility 
    Uc_inv(y) = NewtonRoot(c -> Uc(c) - y, 1); # Inverse of marginal utility

    Uc_cj = Uc.(cj);
    Uc_cj = reshape(Uc_cj, na, nϵ)

    EUc = (P_ϵ * Uc_cj')'
    EUc = reshape(EUc, na*nϵ)
    Upc = R*β*EUc

    cbar = Uc_inv.(Upc)

    # Compute abar
    for ai = 1:na
        for ϵi = 1:nϵ
            aϵi = (ϵi-1)*na+ai
            abar[aϵi] = (aGridl[aϵi] + cbar[aϵi] - w*ϵGrid[ϵi])/R
        end
    end

    return cbar, abar
end

## Test Code ##
AA0,pol0 = AiyagariEGM()
cmat = zeros(eltype(pol0),5*2)
cbar, abar = EulerBackEGM(pol0, AA0, cmat, amat)


