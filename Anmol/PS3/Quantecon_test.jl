#### Aiyagari Model (Test with Quantecon parameters) ####

## Packages 
using ForwardDiff
using Interpolations
using Plots

## Parameters
β = 0.96;
σ = 1.0;


## Subroutine
include("Tauchen.jl")
include("NewtonRoot.jl")
include("Lin_interpolate.jl")

## Grid for A and E

a_min = 1e-10; # Set a_min to this value to ensure that guess for consumption is not negative
a_max = 20.0;
a_size = 200;
a_vals = range(a_min, a_max, length = a_size);

ϵ_size = 2;
ϵ_vals = range(0.1, 1.0, length = ϵ_size);
P_ϵ = [0.9 0.1; 0.1 0.9];

## Guess r and w (for now)
r = 0.03;
w = 0.956;

## EGM
a1_size = 200; # size of a'
a1_vals = range(a_min, a_max, length = a1_size);  # Grid for a'
s_EGM = collect(Iterators.product(a1_vals, ϵ_vals)); # matrix of states A' x E for endogenous grid method

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
                abar_high_index = abar_low_index
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

aj = zeros(a1_size, ϵ_size)
[aj[i,j] = (1+r)*s_EGM[i,j][1] + w*s_EGM[i,j][2] - cj[i,j] for i in 1:a1_size, j in 1:ϵ_size]


## Let's plot
plot(a1_vals, a1_vals, label = "", color = :black, linestyle = :dash)
plot!(a1_vals, aj[:, 1], label = "ϵ = $(round(ϵ_vals[1], digits = 3))")
plot!(a1_vals, aj[:, 2], label = "ϵ = $(round(ϵ_vals[2], digits = 3))")




