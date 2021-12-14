## Packages
using NLsolve
using Plots
using Parameters

## Files
include("Tauchen.jl")
include("NewtonRoot.jl")

## Parameters
Model = @with_kw (θ = 0.35, # capital share
    δ = 0.0464, # depreciation rate
    γ_z = 0.016, # growth rate of productivity
    γ_n = 0.015, # growth rate of population
    β = 0.9722, # discount rate
    β_hat = β*(1+γ_n), # detrended discount rate
    ψ = 2.24, # labor coefficient
    ρ = 0.2,# autocorr of productivity process
    σ = 0.5, # stand. dev. for productivity process
    nz = 5, # number of states for productivity
    μ_z = 0.0, # mean of productivity process
    zGrid = exp.(Tauchen(μ_z,ρ,σ,nz)[1]), # grid for productivity process
    P_z = Tauchen(μ_z,ρ,σ,nz)[2]) # transition matrix


## Get the Steady-State level of capital from a Nonlinear solver
function getSS(md)
    @unpack θ,ψ,δ,β_hat,γ_n,γ_z = md

    function ee!(eq,x)
        k=(x[1])
        h=(x[2])
        eq[1]=β_hat*(θ*k^(θ-1)*h^(1-θ)+1-δ)-(1+γ_n)*(1+γ_z)
        eq[2]=(-(ψ/(1-h)))+((1-θ)*(k^θ)*(h^(-θ)))/((k^θ)*(h^(1-θ))-(1+γ_n)*(1+γ_z)*k+(1-δ)*k)
    end

    S = nlsolve(ee!, [0.5,0.5],ftol = :1.0e-9, method = :trust_region , autoscale = true);
    kss = S.zero[1];
    hss = S.zero[2];
    lss = 1-hss;
    css = (kss^θ)*((exp(0)*hss)^(1-θ))-(1+γ_n)*(1+γ_z)*kss+(1-δ)*kss;

    return kss,hss,lss,css
end

# Solve h from intratemporal equation over the grids
function geth(md,nk,kGrid)

    @unpack θ,ψ,γ_n,γ_z,δ,nz,zGrid = md
    
    h_star = zeros(nk,nk,nz);

    for i = 1:nk
        for j = 1:nk
            for k = 1:nz
                intra(h) = (ψ/(1-θ))*((kGrid[i]^θ)*((zGrid[k]*h)^(1-θ))-(1+γ_n)*(1+γ_z)*kGrid[j]+(1-δ)*kGrid[i])+((h-1)*(kGrid[i]^θ)*(zGrid[k]^(1-θ))*(h^(-θ)))
                h = NewtonRoot(h -> intra(h),0.5)
                if h > 1
                    h_star[i,j,k] = 0.9999;
                else
                    h_star[i,j,k] = h;
                end
            end
        end
    end

    return h_star

end


# Precalculate return value
function getrt(md,nk,kGrid)

    @unpack θ,ψ,γ_n,γ_z,δ,nz,zGrid = md

    h_star = geth(md,nk,kGrid)
    rt = zeros(nk,nk,nz);

    for i = 1:nk
        for j = 1:nk
            for k = 1:nz
                c = (kGrid[i]^θ)*((zGrid[k]*h_star[i,j,k])^(1-θ))-(1+γ_n)*(1+γ_z)*kGrid[j]+(1-δ)*kGrid[i];
                if c <= 0
                    rt[i,j,k] = -100.0;
                else
                    rt[i,j,k] = log(c)+ψ*log(1-h_star[i,j,k]);
                end
            end
        end
    end

    return rt,h_star
    
end

# Main loop
function VFI(md,nk,kGrid)
    @unpack θ,ψ,γ_n,γ_z,δ,nz,zGrid,β_hat,P_z = md
    
    #Preallocate memories
    V_old = zeros(nk,nz);
    V_new = zeros(nk,nz);
    argmax = Array{Int64,2}(undef,nk,nz);

    kpol = zeros(nk,nz);
    hpol = zeros(nk,nz);
    cpol = zeros(nk,nz);

    maxDiff = 10.0; 
    tol = 1.0e-10; 
    iter = 0;

    rt,h_star = getrt(md,nk,kGrid);

    while maxDiff > tol
        iter = iter+1;
        println("Current iteration: $iter")
        
        EV = V_old*P_z';
        for i = 1:nk, k = 1:nz
            V_new[i,k],argmax[i,k] = findmax(rt[i,:,k].+β_hat*EV[:,k]);
        end
        
        maxDiff = maximum(abs.(V_new-V_old));
        println("maxDiff = $maxDiff")
        V_old = copy(V_new); 
    end

    for i = 1:nk,j = 1:nz
        kpol[i,j] = kGrid[argmax[i,j]];
        hpol[i,j] = h_star[i,argmax[i,j],j];
        cpol[i,j] = (kGrid[i]^θ)*(zGrid[j]*hpol[i,j])^(1-θ)-(1+γ_n)*(1+γ_z)*kpol[i,j]+(1-δ)*kGrid[i];
    end

    return V_new,kpol,hpol,cpol
end