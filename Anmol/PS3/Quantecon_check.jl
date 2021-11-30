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
    β::T
    σ::T
    A::T 
    N::T 
    θ::T 
    δ::T 
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
    σ::T = 1.0,
    β::T = 0.96,
    A::T = 1.0,
    N::T = 1.0,
    θ::T = 0.33,
    δ::T = 0.05,
    amin::T = 1e-10,
    amax::T = 18.0,
    na::I = 200,
    nϵ::I = 2) where {T <: Real, I <: Integer}

    ########### Parameters
    params = AiyagariParameters(β,σ,A,N,θ,δ);
    w = A * (1 - θ) * (A * θ / (r + δ)) ^ (θ / (1 - θ));

    ########### Policy grid
    aGrid = collect(LinRange(sqrt(amin), sqrt(amax), na));
    aGrid = aGrid.^2;
    aGridl = repeat(aGrid, nϵ)

    ########### Transition
    ϵGrid = [0.1;1.0];
    P_ϵ = [0.9 0.1; 0.1 0.9];

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
    @unpack β = params

    U(c) = log(c); # Utility function, for now, assume labor is exogenous
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

##### Test code
# Check for policy functions
r0 = 0.03;
AA0, cpol0 = AiyagariEGM(r0);
abar0 = zeros(length(cpol0));
cpol_EGM, apol_EGM = SolveEGM(AA0,cpol0,abar0);

plot(AA0.aGrid, AA0.aGrid, label = "", color = :black, linestyle = :dash)
plot!(AA0.aGrid, apol_EGM[1:200,:], label = "ϵ = $(round(AA0.ϵGrid[1], digits = 3))")
plot!(AA0.aGrid, apol_EGM[201:400,:], label = "ϵ = $(round(AA0.ϵGrid[2], digits = 3))")

# Check aggregate demand and aggregate supply of capital
r_vals = LinRange(0.005,0.04,20)
Ar_vals = zeros(length(r_vals))
Nbar_vals = zeros(length(r_vals))
K_vals = zeros(length(r_vals))

for i = 1:length(r_vals)
    r0 = r_vals[i];
    AA0, cpol0 = AiyagariEGM(r0);
    abar0 = zeros(length(cpol0));
    cpol_EGM, apol_EGM = SolveEGM(AA0,cpol0,abar0);

    λ_init_test = zeros(AA0.na*AA0.nϵ) .+ 1/(AA0.na*AA0.nϵ);
    Qmat_test = zeros(AA0.na*AA0.nϵ,AA0.na*AA0.nϵ);
    Qmat_test = MakeTransMatEGM(AA0,apol_EGM,Qmat_test);
    λ_test = StationaryDistributionEGM(Qmat_test,λ_init_test)
    
    Ar_vals[i] = dot(λ_test, AA0.aGridl)
    Nbar_vals[i] = dot(λ_test, repeat(AA0.ϵGrid, inner = AA0.na));
    K_vals[i] = ((r0+AA0.params.δ)/(AA0.params.θ*Nbar_vals[i]^(1-AA0.params.θ)))^(1/(AA0.params.θ-1));
end

plot(Ar_vals,r_vals, label = "supply",lw = 2, alpha = 0.6,xlim = (2, 14), ylim = (0.0, 0.1))
plot!(K_vals,r_vals,label = "demand",lw = 2, alpha = 0.6,xlim = (2, 14), ylim = (0.0, 0.1))