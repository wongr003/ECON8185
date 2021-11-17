# This module solves the Ramsey problem for the quasi linear case
# We'll try to replicate figures from quant econ https://python-advanced.quantecon.org/opt_tax_recur.html#top

#----------------- CHECKING WITH QUANTECON ---------------------------------------------------#
## Parameters
β = 0.9;
σ = 2;
γ = 2;

## Utility functions
u(x) = x^(1 - σ) / (1 - σ);
v(x) = x^(1 + γ) / (1 + γ);
U(c, n) = u(c) - v(n);

## Differentials
Uc(c) = ForwardDiff.derivative(u, c);
Ucc(c) = ForwardDiff.derivative(Uc, c);

Un(n) = -ForwardDiff.derivative(v, n);
Unn(n) = ForwardDiff.derivative(Un, n);

## Defining states
g = [0.1; 0.1; 0.1; 0.1; 0.2; 0.1];
θ = ones(length(g));
s = [(1.0, g[i]) for i in 1:6];
Ps = [0.0 1.0 0.0 0.0 0.0 0.0;
      0.0 0.0 1.0 0.0 0.0 0.0;
      0.0 0.0 0.0 0.5 0.5 0.0;
      0.0 0.0 0.0 0.0 0.0 1.0;
      0.0 0.0 0.0 0.0 0.0 1.0;
      0.0 0.0 0.0 0.0 0.0 1.0];

include("NewtonRoot.jl");

## This function calculates Ramsey Allocation, given Φ and initial state s0 
function RamseyAllocation(Φ::Real, s0::Tuple)

    function time1_allocation(s::Tuple)
        θ, g = s[1], s[2];
        n(c::Real) = (g + c) / θ;
        res1(c::Real) = (1 + Φ) * (Uc(c) * θ + Un(n(c))) + Φ * (c * Ucc(c) * θ + n(c) * Unn(n(c)));
        c = NewtonRoot(c -> res1(c), 0.5);

        return c, n(c)
    end

    c_time1 = [time1_allocation(s[i])[1] for i in 1:length(s)];
    n_time1 = [time1_allocation(s[i])[2] for i in 1:length(s)];

    function time0_allocation(s::Tuple)
        θ, g = s[1], s[2];
        n(c::Real) = (g + c) / θ;
        b0(c::Real) = 1.0;
        res0(c::Real) = (1 + Φ) * (Uc(c) * θ + Un(n(c))) + Φ * (c * Ucc(c) * θ + n(c) * Unn(n(c))) - Φ * Ucc(c) * b0(c) * θ
        c = NewtonRoot(c -> res0(c), 0.5)

        return c, n(c)
    end

    c_time0, n_time0 = time0_allocation(s0)[1], time0_allocation(s0)[2];

    return c_time1, n_time1, c_time0, n_time0

end

c_time1, n_time1, c_time0, n_time0 = RamseyAllocation(0.06175628494006816, s[1])

sum_mat = inv(1.0I - β * Ps) * (Uc.(c_time1) .* c_time1 .+ Un.(n_time1) .* n_time1);

ResImC = Uc(c_time0) - Uc(c_time0) * c_time0 - Un(n_time0) * n_time0 - β * sum([Ps[1,i]*sum_mat[i] for i in 1:length(sum_mat)]);

b_time1 = sum_mat ./ Uc.(c_time1);
b_time0 = 1.0;

τ_time1 = [1.0 + Un(n_time1[i]) / Uc(c_time1[i]) for i in 1:length(s)];




function search_Φ(Φ::Real, s0::Tuple)
    c_time1, n_time1, c_time0, n_time0 = RamseyAllocation(Φ, s0);
    
    i_s0 = findfirst(isequal(s0), s);

    sum_mat = inv(1.0I - β * Ps) * (Uc.(c_time1) .* c_time1 .+ Un.(n_time1) .* n_time1); #this sum is independent of s0

    implementCons = Uc(c_time0) - Uc(c_time0) * c_time0 - Un(n_time0) * n_time0 - β * sum([Ps[i_s0,i] * sum_mat[i] for i in 1:length(sum_mat)]);

    return implementCons

end

Φ_est = NewtonRoot(x -> search_Φ(x, s[1]), 0.1); # 0.06175628494006816

τ_time0 = 0.0;

## Plotting
sHist_h = [2, 3, 4, 6, 6, 6];
sHist_l = [2, 3, 5, 6, 6, 6];

c_sim_h = [c_time1[i] for i in sHist_h];
c_sim_l = [c_time1[i] for i in sHist_l];

n_sim_h = [n_time1[i] for i in sHist_h];
n_sim_l = [n_time1[i] for i in sHist_l];

b_sim_h = [b_time1[i] for i in sHist_h];
b_sim_l = [b_time1[i] for i in sHist_l];

τ_sim_h = [τ_time1[i] for i in sHist_h];
τ_sim_l = [τ_time1[i] for i in sHist_l];

c_plot_h = vcat(c_time0, c_sim_h)
c_plot_l = vcat(c_time0, c_sim_l)

n_plot_h = vcat(n_time0, n_sim_h)
n_plot_l = vcat(n_time0, n_sim_l)

b_plot_h = vcat(b_time0, b_sim_h)
b_plot_l = vcat(b_time0, b_sim_l)

τ_plot_h = vcat(τ_time0, τ_sim_h)
τ_plot_l = vcat(τ_time0, τ_sim_l)


plot(c_plot_h, label="g high")
plot!(c_plot_l, label = "g low")

plot(n_plot_h, label="g high")
plot!(n_plot_l, label = "g low")

plot(b_plot_l, label="g high")
plot!(b_plot_h, label = "g low")

plot(τ_plot_l, label="g high")
plot!(τ_plot_h, label = "g low")