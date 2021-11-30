## Packages
using ForwardDiff
using UnPack
using Plots
using LinearAlgebra

## Files
include("Tauchen.jl")
include("NewtonRoot.jl")

################################### Model types #########################
mutable struct AiyagariParameters{T <: Real}
    η::T
    μ::T
    β::T
    δ::T
    θ::T
    ρ::T
    σ::T
    μ_ϵ::T
end

mutable struct AiyagariModel{T <: Real, I <: Integer}
    r::T
    w::T
    params::AiyagariParameters{T}
    aGrid::Array{T,1} # Policy grid
    aGridl::Array{T,1} # Grid for a with different ϵ stacked up
    na::I
    ϵGrid::Array{T,1} # Labor shock grid
    nϵ::I
    P_ϵ::Array{T,2} # Transition matrix for labor shock process
end

function AiyagariEGM(
    r::T,
    η::T = 0.3,
    μ::T = 1.5,
    β::T = 0.98,
    δ::T = 0.075,
    θ::T = 0.3,
    ρ::T = 0.6,
    σ::T = 0.3,
    μ_ϵ::T = 0.0,
    amin::T = 0.0,
    amax::T = 40.0,
    na::I = 10,
    nϵ::I = 5) where {T <: Real, I <: Integer}

    ########### Parameters
    params = AiyagariParameters(η,μ,β,δ,θ,ρ,σ,μ_ϵ);

    ########### Implied wage
    w = (1-θ)*((r+δ)/θ)^(θ/(θ-1));

    ########### Policy grid
    aGrid = collect(LinRange(sqrt(amin), sqrt(amax), na));
    aGrid = aGrid.^2;
    aGridl = repeat(aGrid, nϵ)

    ########### Transition
    ϵGrid, P_ϵ = Tauchen(μ_ϵ, ρ, σ, nϵ);
    ϵGrid = exp.(collect(ϵGrid));

    ########### Initial guess for consumption 
    cGuess = r*aGridl + w*repeat(ϵGrid, inner = na)
    
    return AiyagariModel(r,w,params,aGrid,aGridl,na,ϵGrid,nϵ,P_ϵ), cGuess
end

function interpC(model::AiyagariModel,
                 cbar::AbstractArray,
                 abar::AbstractArray,
                 a_tmr::T,
                 ϵ::T) where {T <: Real}

    @unpack r,w,na = model

    if a_tmr <= abar[1]
        cbar_interp = (1+r)*a_tmr + w*ϵ;
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

    @unpack r,w,na,nϵ,aGridl,ϵGrid = model

    cbar = reshape(cbar,na,nϵ);
    abar = reshape(abar,na,nϵ);

    for ϵi = 1:nϵ
        for ai = 1:na
            aϵi = (ϵi-1)*na+ai
            cbar_interp[aϵi] = interpC(model,cbar[:,ϵi],abar[:,ϵi],aGridl[aϵi],ϵGrid[ϵi]) 
        end
    end

    return cbar_interp
end


function EGM(model::AiyagariModel,
             cpol::AbstractArray,
             abar::AbstractArray)

    @unpack r,w,params,na,nϵ,aGridl,P_ϵ,ϵGrid = model
    @unpack β,η,μ = params

    U(c) = (((c^η)*(0.5^(1-η)))^(1-μ))/(1-μ); # Utility function, for now, assume labor is exogenous
    Uc(c) = ForwardDiff.derivative(U, c); # Marginal utility 
    Uc_inv(y) = NewtonRoot(c -> Uc(c) - y, 1); # Inverse of marginal utility

    Uc_cj = Uc.(cpol)
    Uc_cj = reshape(Uc_cj, na, nϵ);

    EUc = (P_ϵ*Uc_cj')';
    EUc = reshape(EUc, na*nϵ);
    cbar = Uc_inv.((1+r)*β*EUc);

    # Compute abar
    for ai = 1:na
        for ϵi = 1:nϵ
            aϵi = (ϵi-1)*na+ai
            abar[aϵi] = (cbar[aϵi]+aGridl[aϵi]-w*ϵGrid[ϵi])/(1+r)
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

    @unpack r,w,na,aGridl,nϵ,ϵGrid = model

    for i = 1:10000
        cNext = EGM(model,cCurrent,abar)
        if (i-1) % 50 == 0
            test = maximum(abs.(cCurrent - cNext));
            #println("iteration: $i"," ")
            if test < tol
                #println("Solved in $i iterations")
                break
            end
        end
        cCurrent = copy(cNext);
    end

    apol = zeros(na*nϵ)
    for ai = 1:na
        for ϵi = 1:nϵ
            aϵi = (ϵi-1)*na+ai
            apol[aϵi] = (1+r)*aGridl[aϵi] + w*ϵGrid[ϵi] - cCurrent[aϵi] 
        end
    end

    return cCurrent, apol
end

function MakeTransMatEGM(model::AiyagariModel,
                         apol::AbstractArray,
                         Qmat)
    
    @unpack na,nϵ,aGridl,P_ϵ = model
    apol = reshape(apol,na,nϵ);   
    aGridl = reshape(aGridl,na,nϵ);                  
    for i = 1:na
        for j = 1:nϵ
            aStar = apol[i,j];
            l = searchsortedlast(aGridl[:,j],aStar);

            (l > 0 && l < na) ? l = l :
                (l == na) ? l = na-1 :
                    l = 1

            p = (aStar-aGridl[l,j])/(aGridl[l+1,j]-aGridl[l,j]);
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

function StationaryDistributionEGM(Qmat,λCurrent,tol = 1e-10)
    for i = 1:10000
        λnext = Qmat*λCurrent
        if (i-1) % 50 == 0
            test = maximum(abs.(λCurrent - λnext));
            #println("iteration: $i"," ")
            if test < tol
                #println("Solved in $i iterations")
                break
            end
        end
        λCurrent = copy(λnext);
    end

    return λCurrent
end

function equilibriumEGM(model::AiyagariModel,
                        cpol::AbstractArray,
                        abar::AbstractArray,
                        tol = 1e-5,maxr = 50) 

    @unpack params,na,nϵ,aGridl,ϵGrid = model
    @unpack β,δ,θ = params

    Ar = 0.0;
    
    λ_init = zeros(na*nϵ) .+ 1/(na*nϵ);
    ur,lr = 1/β - 1,-δ
    

    for rit = 1:maxr
        cstar, apol = SolveEGM(model,cpol,abar);

        Qmat = zeros(na*nϵ, na*nϵ);
        Qmat = MakeTransMatEGM(model,apol,Qmat);
        λ = StationaryDistributionEGM(Qmat,λ_init);

        Ar = dot(aGridl, λ);
        Nbar = dot(λ, repeat(ϵGrid, inner = na));
        K = ((model.r+δ)/(θ*Nbar^(1-θ)))^(1/(θ-1));
    
        println("This is iteration $rit, and r = $(model.r)")

        if (Ar > K)
            ur = model.r;
            model.r = 1.0/2.0*(lr+ur);
        else
            lr = model.r;
            model.r = 1.0/2.0*(lr+ur);
        end
        println("Bond Supply: ",Ar," ","Bond Demand: ",K)

        if abs(Ar - K) < tol
            println("Market clear!")
            return cstar, apol, λ, Ar, K, model.r
            break
        end
    end

    return println("Markets did not clear")
end