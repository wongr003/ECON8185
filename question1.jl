using Plots

# Question 1
# Defining parameters
θ = 0.25;
β = 0.9;
δ = 1;
A = (1 - β * (1 - δ)) / (θ * β); # This will normalize the SS to 1

# Defining the element nodes between [0,2]. Their distance is increasing exponentially since it is known
# that the consumption functiono is less linear close to 0. 
K = zeros(15);
for i = 2:length(K)
    K[i] = K[i - 1] + 0.0005 * exp(0.574 * (i - 2))
end

include("finite_elements_functions.jl")

# Capital policy function derived from budget constraint
polk(k, α) = min(max(eps(), A * k.^θ + (1 - δ) * k - cn(k, α, K)), K[end]) # min/max are needed to avoid NaNs and other numerical instabilities

function residual(k, α, K)
    R = cn(k, α, K) / cn(polk(k, α), α, K) * β * (A * θ * polk(k, α)^(θ - 1) + 1 - δ) - 1;
    return R
end

#Setting initial conditions
initial =  ones(length(K)-1) .* range(0.35, stop = 3.5, length = length(K)-1);

#Here we start the minimization procedure we want to find α:= argmin mini(α)

#Solver stuff:
#lower and upper bound of the parameters:
lower = zeros(length(initial));
upper = Inf*ones(length(initial));
#Optimizer method is BFGS, see Judd's book page 114 for an explanation:
inner_optimizer = BFGS();

#Solver:
bla = optimize(mini,lower,upper,initial, Fminbox(inner_optimizer))
#Parameters
α = vcat(0,bla.minimizer)