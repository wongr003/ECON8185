#### Aiyagari Model ####

## Packages 

## Grid for A and E
include("Tauchen.jl")

a_min = 1e-10;
a_max = 18.0;
a_size = 5;
a_vals = range(a_min, a_max, length = a_size);

ρ = 0.6;
σ = 0.3;
μ = 0.0;
n = 2;
ϵ_vals, P_ϵ = Tauchen(μ, ρ, σ, n);