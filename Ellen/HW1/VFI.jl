## Packages
using NLsolve
using Plots

## Files
include("Tauchen.jl")
include("NewtonRoot.jl")

## Parameters
θ = 0.35; # capital share
δ = 0.0464; # depreciation rate
γ_z = 0.016; # growth rate of productivity
γ_n = 0.015; # growth rate of population
β = 0.9722; # discount rate
β_hat = β*(1+γ_n); # detrended discount rate
ψ = 2.24; # labor coefficient
nz = 2; # number of states for productivity
σ = 0.5; # stand. dev. for productivity process
ρ = 0.2; # autocorr of productivity process
μ_z = 0.0; # mean of productivity process
zGrid = exp.(Tauchen(μ_z,ρ,σ,nz)[1]); # grid for productivity process
P_z = Tauchen(μ_z,ρ,σ,nz)[2]; # transition matrix

## Get the Steady-State level of capital from a Nonlinear solver
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
# css 

## Construct capital grid
nk = 1000; # number of capital grid
kmin = 0.5*kss;
kmax = 1.5*kss;
kGrid = LinRange(kmin,kmax,nk);

# Solve h from intratemporal equation over the grids
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

# Precalculate return value
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

## Main loop
V_old = zeros(nk,nz);
#Fill an initial guess
for j = 1:nz
    for i = 1:nk
        V_old[i, j] = (log((kGrid[i]^θ)*((zGrid[j]*hss)^(1-θ))-δ*kGrid[i])+ψ*log(1-hss))/(1-β_hat);
    end
end

V_new = zeros(nk,nz);
argmax = Array{Int64,2}(undef,nk,nz);

kpol = zeros(nk,nz);
hpol = zeros(nk,nz);
cpol = zeros(nk,nz);

maxDiff = 10.0; 
tol = 1.0e-10; 
iter = 0;

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

    for i = 1:nk,j = 1:nz
        kpol[i,j] = kGrid[argmax[i,j]];
        hpol[i,j] = h_star[i,argmax[i,j],j];
        cpol[i,j] = (kGrid[i]^θ)*(zGrid[j]*hpol[i,j])^(1-θ)-(1+γ_n)*(1+γ_z)*kpol[i,j]+(1-δ)*kGrid[i];
    end
end

plot(kGrid,V_new[:,1])
plot!(kGrid,V_new[:,2])
 
plot(kGrid,kGrid,label = "", color = :black, linestyle = :dash)
plot!(kGrid,kpol[:,1],label = "low",legend = :topleft)
plot!(kGrid,kpol[:,2],label = "high",legend = :topleft)

plot(kGrid,hpol[:,1],label = "low",legend = :topleft)
plot!(kGrid,hpol[:,2],label = "high",legend = :topleft)

plot(kGrid,cpol[:,1],label = "low",legend = :topleft)
plot!(kGrid,cpol[:,2],label = "high",legend = :topleft)

