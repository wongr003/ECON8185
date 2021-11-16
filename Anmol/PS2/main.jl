## Packages
using ForwardDiff, LinearAlgebra, Plots

## Parameters
β = 0.99;
σ = 2;
γ = 1;

## Utility functions
u(x) = x^(1 - σ) / (1 - σ);
v(x) = x^(1 + 1/γ) / (1 + 1/γ);
U(c, n) = u(c) - v(n);

## Differentials
Uc(c) = ForwardDiff.derivative(u, c);
Ucc(c) = ForwardDiff.derivative(Uc, c);

Un(n) = -ForwardDiff.derivative(v, n);
Unn(n) = ForwardDiff.derivative(Un, n);

## Discretizes g and θ
include("Tauchen.jl")
g, Pg = Tauchen(0.95, (1.2/15)^2, 5, (1 - 0.95) * log(0.15));
θ, Pθ = Tauchen(0.95, (2/400)^2, 5, (1 - 0.95) * log(1));
s = vec(collect(Iterators.product(θ, g)));
Ps = kron(Pg, Pθ)

include("NewtonRoot.jl");

## This function calculates Ramsey Allocation, given Φ and initial state s0 
function RamseyAllocation(Φ::Real, s0::Tuple)

    function time1_allocation(s::Tuple)
        log_θ, log_g = s[1], s[2];
        θ, g = exp(log_θ), exp(log_g);
        n(c::Real) = (g + c) / θ;
        res1(c::Real) = (1 + Φ) * (Uc(c) + Un(n(c))) + Φ * (c * Ucc(c) + n(c) * Unn(n(c)));
        c = NewtonRoot(c -> res1(c), 0.5);

        return c, n(c)
    end

    c_time1 = [time1_allocation(s[i])[1] for i in 1:length(s)];
    n_time1 = [time1_allocation(s[i])[2] for i in 1:length(s)];

    function time0_allocation(s::Tuple)
        log_θ, log_g = s[1], s[2];
        θ, g = exp(log_θ), exp(log_g);
        n(c::Real) = (g + c) / θ;
        b0(c::Real) = 4 * θ * n(c);
        res0(c::Real) = (1 + Φ) * (Uc(c) + Un(n(c))) + Φ * (c * Ucc(c) + n(c) * Unn(n(c))) - Φ * Ucc(c) * b0(c)
        c = NewtonRoot(c -> res0(c), 0.5)

        return c, n(c)
    end

    c_time0, n_time0 = time0_allocation(s0)[1], time0_allocation(s0)[2];

    return c_time1, n_time1, c_time0, n_time0

end

## Updating the value of Φ:
c_time1, n_time1, c_time0, n_time0 = RamseyAllocation(0.12615792845184043, s[1])

sum_mat = inv(1.0I - β * Ps) * (Uc.(c_time1) .* c_time1 .+ Un.(n_time1) .* n_time1);
implementCons = Uc(c_time0) * (4 * exp(s[1][1]) * n_time0) - Uc(c_time0) * c_time0 - Un(n_time0) * n_time0 - β * sum([Ps[1,i] * sum_mat[i] for i in 1:length(sum_mat)]);

b_time1 = sum_mat ./ Uc.(c_time1);

τ_time1 = 1.0 .+ Un.(n_time1) ./ Uc.(c_time1);

## Root finding method for Φ

function search_Φ(Φ::Real, s0::Tuple)
    c_time1, n_time1, c_time0, n_time0 = RamseyAllocation(Φ, s0);
    
    i_s0 = findfirst(isequal(s0), s);

    sum_mat = inv(1.0I - β * Ps) * (Uc.(c_time1) .* c_time1 .+ Un.(n_time1) .* n_time1); #this sum is independent of s0

    implementCons = Uc(c_time0) * (4 * exp(s0[1]) * n_time0) - Uc(c_time0) * c_time0 - Un(n_time0) * n_time0 - β * sum([Ps[i_s0,i] * sum_mat[i] for i in 1:length(sum_mat)]);

    return implementCons

end

Φ_est = NewtonRoot(x -> search_Φ(x, s[1]), 0.1) # 0.12615792845184043

#----------------- SIMULATING RAMSEY POLICY WITH STATE CONTINGENT DEBT ----------------------#
include("Finite_Markov_Chains.jl")
state_tran = mc_sample_path(Ps, init = 1, sample_size = 1000);

c_sim = [c_time1[i] for i in state_tran];
n_sim = [n_time1[i] for i in state_tran];
b_sim = [b_time1[i] for i in state_tran];
τ_sim = [τ_time1[i] for i in state_tran];

plot(c_sim, label = "Simulated path C")
plot(n_sim, label = "Simulated path N")
plot(b_sim, label = "Simulated path b")
plot(τ_sim, label = "Simulated path taxes")