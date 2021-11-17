# This subroutine discretize an AR(1) process with Tauchen (86)
# λ' = (1 - ρ)μ_λ + ρλ + ϵ, where ϵ ~ N(0, (σ_ϵ)^2)
# Inputs:
# Outputs:

using Distributions

function Tauchen(μ_λ, ρ, σ_ϵ, n; m = 3)
    σ_λ = σ_ϵ/(1-ρ^2)^(1/2);

    λ_min = μ_λ - m*σ_λ;
    λ_max = μ_λ + m*σ_λ;
    λ_vals = range(λ_min, λ_max, length = n); # can also use LinRange(λ_min, λ_max, n)

    w = λ_vals[2] - λ_vals[1];

    d = Normal();
    P = zeros(n,n);

    for i in 1:n
        for j = 1:n
            if j == 1
                P[i,j] = cdf(d, (λ_vals[1] + w/2 - (1-ρ)*μ_λ - ρ*λ_vals[i])/σ_ϵ);
            elseif j == n
                P[i,j] = 1 - cdf(d, (λ_vals[n] - w/2 - (1-ρ)*μ_λ - ρ*λ_vals[i])/σ_ϵ);
            else
                P[i,j] = cdf(d, (λ_vals[j] + w/2 - (1-ρ)*μ_λ - ρ*λ_vals[i])/σ_ϵ) - cdf(d, (λ_vals[j] - w/2 - (1-ρ)*μ_λ - ρ*λ_vals[i])/σ_ϵ);
            end

        end
    end
    return λ_vals, P
end

