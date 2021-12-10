## Packages
using NLsolve
using Parameters
using BenchmarkTools

## Files
include("Tauchen.jl")

## Model parameters
Model = @with_kw (θ = 0.35, # capital share
    δ = 0.0464, # depreciation rate
    γ_z = 0.016, # growth rate of productivity
    γ_n = 0.015, # growth rate of population
    β = 0.9722, # discount rate
    β_hat = β*(1+γ_n), # detrended discount rate
    ψ = 2.24, # labor coefficient
    nz = 2, # number of states for productivity
    σ = 0.5, # stand. dev. for productivity process
    ρ = 0.2, # autocorr of productivity process
    μ_z = 0.0, # mean of productivity process
    zGrid = exp.(Tauchen(μ_z,ρ,σ,nz)[1]), # grid for productivity process
    P_z = Tauchen(μ_z,ρ,σ,nz)[2]) # transition matrix

md = Model()

## Get the Steady-State level of capital from a Nonlinear solver
function ee!(eq,x;model=md)
    @unpack θ,δ,γ_z,γ_n,β_hat,ψ = model
    k=(x[1])
    h=(x[2])
    eq[1]=β_hat*(θ*k^(θ-1)*h^(1-θ)+1-δ)-(1+γ_n)*(1+γ_z)
    eq[2]=(-ψ/(1-h))+((1-θ)*k^θ*h^(-θ))/(k^θ*h^(1-θ)+(1-δ)*k-(1+γ_n)*(1+γ_z)*k)
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
