#### Aiyagari Model ####

## Packages 
using ForwardDiff
using Plots

## Parameters
η = 0.3;
μ = 1.5;
β = 0.98;
ρ = 0.6;
σ = 0.3;
μ_ϵ = 0.0;

## Subroutine
include("Tauchen.jl")
include("NewtonRoot.jl")
include("Lin_interpolate.jl")

## Grid for A and E
a_min = 1e-10; 
a_max = 18.0;

ϵ_size = 2;
ϵ_vals, P_ϵ = Tauchen(μ_ϵ, ρ, σ, ϵ_size);
ϵ_vals = exp.(ϵ_vals);

## Guess r and w (for now)
r = 0.01;
w = 1.0;

## EGM
a1_size = 5; # size of a'
a1_vals = range(a_min, a_max, length = a1_size);  # Grid for a'
s_EGM = collect(Iterators.product(a1_vals, ϵ_vals)); # matrix of states A' x E for endogenous grid method

l = 0.5; # for now, assume labor is exogenous
U(c::Real) = (((c^η)*(l^(1-η)))^(1-μ))/(1-μ); # Utility function
Uc(c) = ForwardDiff.derivative(U, c); # Marginal utility 
Uc_inv(y) = NewtonRoot(c -> Uc(c) - y, 1); # Inverse of marginal utility

cj = zeros(a1_size, ϵ_size) # Preallocate memory
[cj[i,j] = (1+r) * s_EGM[i,j][1] + w * s_EGM[i,j][2] for i in 1:a1_size, j in 1:ϵ_size] # Guess of c
distance = 10;
tol = 10^(-5);
maxiter = 10000;
iter = 1;

while distance >= tol && iter <= maxiter

    cjnext = copy(cj);

    cbar = Uc_inv.(β .* (1 + r) * (P_ϵ*Uc.(cjnext)')');
    abar = zeros(a1_size, ϵ_size);
    [abar[i,j] = (cbar[i,j] + s_EGM[i,j][1] - w*s_EGM[i,j][2])/(1+r) for i in 1:a1_size, j in 1:ϵ_size];

    for i in 1:a1_size, j in 1:ϵ_size
        a1 = s_EGM[i,j][1];
        ϵ = s_EGM[i,j][2];
        if a1 <= abar[1, j]
            cj[i,j] = (1+r)*a1 + w*ϵ;
        else 
            abar_vec = abar[:, j];
            abar_low_index = searchsortedlast(abar_vec, a1);
            if abar_low_index == size(abar_vec)[1]
                abar_high_index = abar_low_index;
                abar_low_index = abar_low_index - 1;     
            else
                abar_high_index = abar_low_index + 1;
            end

            abar_low = abar[abar_low_index, j];
            abar_high = abar[abar_high_index, j];
            cbar_low = cbar[abar_low_index, j];
            cbar_high = cbar[abar_high_index, j];
            

            cj[i,j] = Lin_interpolate(cbar_low, cbar_high, abar_low, abar_high, a1);
        end
    end

    distance = maximum(abs.(cj - cjnext));
    iter = iter + 1;
    println("This is iteration $iter");

end

policy_a = zeros(a1_size, ϵ_size);
[policy_a[i,j] = (1+r)*s_EGM[i,j][1] + w*s_EGM[i,j][2] - cj[i,j] for i in 1:a1_size, j in 1:ϵ_size];


## Let's plot
plot(a1_vals, a1_vals, label = "", color = :black, linestyle = :dash)
plot!(a1_vals, policy_a[:, 1], label = "ϵ = $(round(ϵ_vals[1], digits = 3))")
plot!(a1_vals, policy_a[:, 2], label = "ϵ = $(round(ϵ_vals[2], digits = 3))")


## Stationary equlibrium
λjnext = ones(a1_size, ϵ_size)/(a1_size); # Uniform over a only
distance = 10;
tol = 10^(-5);
maxiter = 10000;
iter = 1;

while distance >= tol && iter <= maxiter
    λj = copy(λjnext);
    

    #### Update first state of a1
    for j in 1:ϵ_size
        a_first = a1_vals[1];
        a_second = a1_vals[2];
        for k in 1:ϵ_size, l in 1:a1_size
            astar = policy_a[l,k]
            if astar >= a_first && astar < a_second
                λjnext[1,j] = λjnext[1,j] + P_ϵ[k,j]*((a_second-astar)/(a_second-a_first))*λj[l,k];
            else
                λjnext[1,j] = λjnext[1,j]
                
            end
        end
    end

    #### Update last state of a1
    for j in 1:ϵ_size
        a_last = a1_vals[end];
        a_before_last = a1_vals[end-1];
        for k in 1:ϵ_size, l in 1:a1_size
            astar = policy_a[l,k]
            if astar >= a_before_last && astar < a_last
                λjnext[end,j] = λjnext[end,j] + P_ϵ[k,j]*((astar-a_before_last)/(a_last-a_before_last))*λj[l,k];
                #println("I'm here first and astar = $astar")
            else
                λjnext[end,j] = λjnext[end,j]
                #println("I'm here second and astar = $astar")
            end
        end
    end
            

    #### Update rest of the a1 states
    for i in 2:a1_size-1, j in 1:ϵ_size
        ak = a1_vals[i];
        akP = a1_vals[i-1]; 
        akN = a1_vals[i+1];
        ϵ1 = ϵ_vals[j];

        for k in 1:ϵ_size, l in 1:a1_size
            astar = policy_a[l,k]
            if astar >= akP && astar < ak
                λjnext[i,j] = λjnext[i,j] + P_ϵ[k,j]*((astar-akP)/(ak-akP))*λj[l,k];
            elseif astar >= ak && astar < akN
                λjnext[i,j] = λjnext[i,j] + P_ϵ[k,j]*((akN-astar)/(akN-ak))*λj[l,k];
            else
                λjnext[i,j] = λjnext[i,j];
            end
        end
    end

    distance = maximum(abs.(λj - λjnext));
    iter = iter + 1;
    println("This is iteration $iter");

end



endow = [1.0;2.5]
lamw = 0.6

using LinearAlgebra
EmpTrans = [1.0-lamw σ ;lamw 1.0-σ]
dis = LinearAlgebra.eigen(EmpTrans)
mini = argmin(abs.(dis.values .- 1.0)) 
stdist = abs.(dis.vectors[:,mini]) / sum(abs.(dis.vectors[:,mini]))
lbar = dot(stdist,endow)
states = endow/lbar

@assert sum(EmpTrans[:,1]) == 1.0

guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

