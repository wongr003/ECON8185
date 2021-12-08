## Packages
using LinearAlgebra
using Plots

## Parameters
α = 0.333;
δ = 1.0;
β = 0.975;
A = 1.05;
nk = 1000; # number of capital grids
tol = 1e-5;
maxDiff = 10.0;
iter = 0;
maxiter = 5000;

## Steady state of capital
kss = ((1/β+δ-1)/(α*A))^(1/(α-1));

## Capital grids
kmin = 0.5*kss;
kmax = 1.5*kss;
k_grid = LinRange(kmin, kmax,nk);
kp_grid = LinRange(kmin, kmax,nk);

# VFI
V_old = zeros(nk);
V_new = zeros(nk);
kpol = zeros(nk);

while maxDiff >= tol && iter < maxiter
    iter = iter+1;
    println("Current iteration is $iter")

    temp = zeros(nk, nk);
    V_old = V_new;

    # Take care of negative consumption
    for i = 1:nk
        for j = 1:nk
            c = A*(k_grid[i]^α)+(1-δ)*k_grid[i]-kp_grid[j];
            if c < 0
                temp[i,j] = -10.0;
            else
                temp[i,j] = log(c)+β*V_old[j];
            end
        end
    end

    # Find max and argmax
    V_new,argmax = findmax(temp,dims=2);

    for i = 1:nk
        kpol[i] = k_grid[argmax[i][2]];
    end

    maxDiff = maximum(abs.(V_new-V_old));
    println("maxDiff = $maxDiff")
end

plot(k_grid,k_grid,label = "", color = :black, linestyle = :dash)
plot!(k_grid,kpol,label = "VFI")
plot!(k_grid,α*β*A*k_grid.^α,label = "true")