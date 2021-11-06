## Packages
using CSV, DataFrames, Plots, Distributions, Random, Optim, LinearAlgebra

#### Question 1:

## Load the data
Ydata = CSV.read("data/realGDP.csv", DataFrame, types = [String, Float64]) # Specify types of each column
names(Ydata) # Get column names
rename!(Ydata, "    Gross domestic product " => :GDP) # Rename GDP

## Calculate growth rate of GDP
num_data = size(Ydata)[1];
GDP_growth = repeat([reshape([0.0], 1, 1)], num_data);
[GDP_growth[i] = [100.0] * ((reshape([log(Ydata[i, 2])], 1, 1)) - (reshape([log(Ydata[i - 1, 2])], 1, 1))) for i in 2:num_data]
vec_GDP_growth = [GDP_growth[i][1, 1] for i in 1:num_data]
## Plot the growth rate
plot(Ydata[:, 1], vec_GDP_growth, xlabel = "Year", label = "GDP Growth (%)")
savefig("figs/GDP_growth.png")

#-----------------------------------------------------------------------------------------------------------------#

#### Question 2

Random.seed!(456)

# Parameters
σ_ϵ = 0.5;
σ_ν = 0.5;
n = length(GDP_growth);

# Getting Matrices to apply kalman filter
T = reshape([1.0], 1, 1);
Z = reshape([1.0], 1, 1);
Q = reshape([σ_ν^2], 1, 1);
H = reshape([σ_ϵ^2], 1, 1);
μ0 = reshape([2.0], 1, 1);

# Applying Kalman Filter on real data
include("KalmanFilter.jl")
μhat, v, yhat, P, F, G, K  = KalmanFilter(T, Z, Q, H, n, μ0, GDP_growth)

# Generate a "test" sample, 100 data points
sample_size = 100;

μ_1 = 2.0;
μ_sim = repeat([reshape([μ_1], 1, 1)], sample_size)
[μ_sim[i] = μ_sim[i-1] + [σ_ν * rand(Normal(0.0, 1.0))] for i in 2:sample_size];
sample_y = repeat([zeros(size(GDP_growth[1]))], sample_size);
[sample_y[i] = μ_sim[i] + [σ_ν * rand(Normal(0.0, 1.0))] for i in 1:sample_size];

# Applying Kalman Filter on simulated data
μhat_sim  = KalmanFilter(T, Z, Q, H, sample_size, μ0, sample_y)[1]
σ_μ = sqrt.([KalmanFilter(T, Z, Q, H, sample_size, μ0, sample_y)[4][i + 1][1, 1] for i in 1:sample_size - 1]);


# Plot
vec_μ_sim = [μ_sim[i][1, 1] for i in 1:sample_size]
vec_μhat_sim = [μhat_sim[i + 1][1, 1] for i in 1:sample_size - 1]
μhat_sim_UCI = vec_μhat_sim + 2 * σ_μ;
μhat_sim_LCI = vec_μhat_sim - 2 * σ_μ;

plot(vec_μ_sim[1:end - 1], xlabel = "time", ylabel = "μ", label = "Simulated values", legend=:topleft)
plot!(vec_μhat_sim, label = "Estimated μ from Kalman Filter")
plot!(μhat_sim_UCI, label = "μ + 2*σ")
plot!(μhat_sim_LCI, label = "μ - 2*σ")
savefig("figs/Q2b.png")

#-----------------------------------------------------------------------------------------------------------------#

#### Question 3
## Writing the likelhood function
function log_L_GDP(θ::Vector)  # θ = [log(σ_ϵ), log(σ_ν)]
    σ_ν1 = exp(θ[1])
    σ_ϵ1 = exp(θ[2])
    Q1 = reshape([σ_ν1^2], 1, 1)
    H1 = reshape([σ_ϵ1^2], 1, 1)

    v1 = [KalmanFilter(T, Z, Q1, H1, n, μ0, GDP_growth)[2][i][1] for i in 1:n]
    F1 = [KalmanFilter(T, Z, Q1, H1, n, μ0, GDP_growth)[5][i][1] for i in 1:n]
    Ft = abs.(det.(vec(F1))) # How about this?

    return sum([-0.5 * log(Ft[i]) - 0.5 * v1[i]' * inv(Ft[i]) * v1[i] for i in 1:n])
end


## Optimization
res = optimize(θ -> -log_L_GDP(θ), [-2.0, -2.0])
σ_ν_hat, σ_ϵ_hat = exp(Optim.minimizer(res)[1]), exp(Optim.minimizer(res)[2]) # (0.1717, 2.2575)

## Plotting
σ_ν = 0.1717;
σ_ϵ = 2.2575;

n = length(GDP_growth);

T = reshape([1.0], 1, 1);
Z = reshape([1.0], 1, 1);
Q = reshape([σ_ν^2], 1, 1);
H = reshape([σ_ϵ^2], 1, 1);
μ0 = reshape([2.0], 1, 1);

μ_hat  = KalmanFilter(T, Z, Q, H, n, μ0, GDP_growth)[1]
vec_μ_hat = [μ_hat[i][1, 1] for i in 1:n]

plot(Ydata[:, 1], vec_μ_hat, xlabel = "Year", label = "Hidden State")
savefig("figs/Q3b")

#-----------------------------------------------------------------------------------------------------------------#

#### Question 4