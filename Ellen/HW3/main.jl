# This is the main output file for PS3 Question 1
# We write down matrices T, Z, Q, H of each process 
# α_{t} = T * α_{t-1} + η_{t} ~ N(0, Q)
# y_{t} = Z * α_{t} + ϵ_{t} ~ N(0, H)

## Packages
using Random
using Distributions
using Plots
using Optim
using ForwardDiff
using LinearAlgebra

## Files
include("KalmanFilter.jl")

rng = Random.seed!(1234);
n = 100; # number of periods

############################## Part 1 (AR(1)) #############################
## Question 1.1: AR(1) process (x_{t} = ρ * x_{t-1} + ϵ_{t})
# Parameters
ρ_AR1 = 0.6;
σ_AR1 = 0.5;

# Simulate AR(1) process
y_AR1 = repeat([reshape([0.0], 1, 1)], n); # [0.0] is a Vector{Float64}, 'reshape' changes it to Matrix{Float64}
[y_AR1[i] = [ρ_AR1] * y_AR1[i - 1] + σ_AR1 * randn(rng, 1) for i in 2:n];

## Getting Matrices to apply kalman filter
T_AR1 = reshape([ρ_AR1], 1, 1);
Z_AR1 = reshape([1.0], 1, 1);
Q_AR1 = reshape([σ_AR1^2], 1, 1);
H_AR1 = reshape([0.0], 1, 1);
α0_AR1 = reshape(mean(y_AR1), 1, 1);

## Applying kalman_filter
αhat_AR1, v_AR1, yhat_AR1, P_AR1, F_AR1, G_AR1, K_AR1  = KalmanFilter(T_AR1, Z_AR1, Q_AR1, H_AR1, n, α0_AR1, y_AR1)

## Plots
vec_y_AR1 = [y_AR1[i][1, 1] for i in 1:n]
vec_yhat_AR1 = [yhat_AR1[i][1, 1] for i in 2:n]

plot(vec_y_AR1, xlabel = " Time ", label =" Simulated AR(1) ", legend=:bottomright)
plot!(vec_yhat_AR1, label = "Estimate from Kalman Filter")
savefig("figs/AR1.png")

## Writing the likelhood function
# For the AR1 process we need to estimate two Parameters ρ and σ:
function log_L_AR1(θ::Vector)  # θ = [ρ, log_σ]
    ρ1 = θ[1]
    σ1 = exp(θ[2]) 
    T1 = reshape([ρ1], 1, 1)
    Q1 = reshape([σ1^2], 1, 1)

    v1 = [KalmanFilter(T1, Z_AR1, Q1, H_AR1, n, α0_AR1, y_AR1)[2][i][1] for i in 1:n]
    F1 = [KalmanFilter(T1, Z_AR1, Q1, H_AR1, n, α0_AR1, y_AR1)[5][i][1] for i in 1:n]
    Ft = abs.(det.(vec(F1))) 

    return sum([-0.5 * log(Ft[i]) - 0.5 * v1[i]' * inv(Ft[i]) * v1[i] for i in 1:n])
end

## Optimization
res_AR1 = optimize(θ->-log_L_AR1(θ), [-0.6, -0.69])
ρ_hat_AR1, σ_hat_AR1 = Optim.minimizer(res_AR1)[1], exp(Optim.minimizer(res_AR1)[2]) # (0.61, 0.48)

############################## Part 2 (AR(2)) #############################
## Question 1.2: AR(2) process (x_{t} = ρ1 * x_{t-1} + ρ2 * x_{t-2} + ϵ_{t})
# Parameters
ρ1_AR2 = 0.3
ρ2_AR2 = 0.4
σ_AR2 = 0.5

# Simulate AR(2) process
y_AR2 = repeat([reshape([0.0], 1, 1)], n);
[y_AR2[i] = [ρ1_AR2] * y_AR2[i - 1] + [ρ2_AR2] * y_AR2[i - 2] + σ_AR2 * randn(rng, 1) for i in 3:n];

## Getting Matrices to apply kalman filter
T_AR2 = [ρ1_AR2 ρ2_AR2; 1.0 0.0];
Z_AR2 = [1.0 0.0];
Q_AR2 = [σ_AR2^2 0.0; 0.0 0.0];
H_AR2 = reshape([0.0], 1, 1);
α0_AR2 = [mean(y_AR2); mean(y_AR2)];

αhat_AR2, v_AR2, yhat_AR2, P_AR2, F_AR2, G_AR2, K_AR2 = KalmanFilter(T_AR2, Z_AR2, Q_AR2, H_AR2, n, α0_AR2, y_AR2)

vec_y_AR2 = [y_AR2[i][1, 1] for i in 3:n]
vec_y_hat_AR2 = [yhat_AR2[i][1, 1] for i in 3:n]

plot(vec_y_AR2, xlabel = " Time ", label =" Simulated AR(2) ", legend=:bottomright)
plot!(vec_y_hat_AR2, label = "Estimate from Kalman Filter")
savefig("figs/AR2.png")

## Writing the likelhood function
# For the AR2 process we need to estimate three Parameters ρ1, ρ2, and σ:
function log_L_AR2(θ::Vector)  # θ = [ρ1, ρ2, log_σ]
    ρ1 = θ[1]
    ρ2 = θ[2]
    σ1 = exp(θ[3]) 
    T1 = [ρ1 ρ2; 1.0 0.0]
    Q1 = [σ1^2 0.0; 0.0 0.0]

    v1 = [KalmanFilter(T1, Z_AR2, Q1, H_AR2, n, α0_AR2, y_AR2)[2][i][1] for i in 1:n]
    F1 = [KalmanFilter(T1, Z_AR2, Q1, H_AR2, n, α0_AR2, y_AR2)[5][i][1] for i in 1:n]
    Ft = abs.(det.(vec(F1))) 

    return sum([-0.5 * log(Ft[i]) - 0.5 * v1[i]' * inv(Ft[i]) * v1[i] for i in 3:n])
end

## Optimization
res_AR2 = optimize(θ->-log_L_AR2(θ), [0.3, 0.7, -0.6])
ρ1_hat_AR2,ρ2_hat_AR2,σ_hat_AR2 = Optim.minimizer(res_AR2)[1],Optim.minimizer(res_AR2)[2],exp(Optim.minimizer(res_AR2)[3]) 
# (0.26, 0.48, 0.54)

############################## Part 3 (MA(1)) #############################
## Question 1.3: MA(1) process (x_{t} = ϵ_{t} + ρ * ϵ_{t-1})
# Parameters
ρ_MA1 = 0.7
σ_MA1 = 0.8

# Simulate MA(1) process
ϵ_MA1 = rand(Normal(0, σ_MA1), n)
y_MA1 = repeat([reshape([0.0], 1, 1)], n);
[y_MA1[i] = reshape([ϵ_MA1[i]], 1,1) + [ρ_MA1]*ϵ_MA1[i-1] for i in 2:n] 

## Getting Matrices to apply kalman filter
T_MA1 = [0.0 0.0; 1.0 0.0]
Z_MA1 = [1.0 ρ_MA1]
Q_MA1 = [σ_MA1^2 0.0; 0.0 0.0]
H_MA1 = reshape([0.0], 1, 1)

α0_MA1 = reshape([0.0; 0.0],2,1);
αhat_MA1, v_MA1, yhat_MA1, P_MA1, F_MA1, G_MA1, K_MA1 = KalmanFilter(T_MA1, Z_MA1, Q_MA1, H_MA1, n, α0_MA1, y_MA1)

vec_y_MA1 = [y_MA1[i][1, 1] for i in 2:n]
vec_y_hat_MA1= [yhat_MA1[i][1, 1] for i in 2:n]

plot(vec_y_MA1, xlabel = " Time ", label =" Simulated MA(1) ", legend=:bottomright)
plot!(vec_y_hat_MA1, label = "Estimate from Kalman Filter")
savefig("figs/MA1.png")

## Writing the likelhood function
# For the MA1 process we need to estimate two Parameters ρ and σ:
function log_L_MA1(θ::Vector)  # θ = [ρ, log_σ]
    ρ1 = θ[1]
    σ1 = exp(θ[2]) 
    Z1 = [1.0 ρ1]
    Q1 = [σ1^2 0.0; 0.0 0.0]

    v1 = [KalmanFilter(T_MA1, Z1, Q1, H_MA1, n, α0_MA1, y_MA1)[2][i][1] for i in 1:n]
    F1 = [KalmanFilter(T_MA1, Z1, Q1, H_MA1, n, α0_MA1, y_MA1)[5][i][1] for i in 1:n]
    Ft = abs.(det.(vec(F1))) 

    return sum([-0.5 * log(Ft[i]) - 0.5 * v1[i]' * inv(Ft[i]) * v1[i] for i in 2:n])
end

## Optimization
res_MA1 = optimize(θ->-log_L_MA1(θ), [0.6, -0.6])
ρ_hat_MA1,σ_hat_MA1 = Optim.minimizer(res_MA1)[1],exp(Optim.minimizer(res_MA1)[2]) # (0.61, 0.73)

############################## Part 4 (Random Walk) #############################
## Question 1.4: Random Walk process 
# Parameters
σ_ϵ = 0.8
σ_η = 0.6

# Simulate MA(1) process
ϵ_RW = rand(Normal(0, σ_ϵ), n)
η_RW = rand(Normal(0, σ_η), n)

μ_RW = repeat([reshape([0.0], 1, 1)], n)
[μ_RW[i] = μ_RW[i-1] + reshape([η_RW[i]], 1,1) for i in 2:n]

y_RW = repeat([reshape([0.0], 1, 1)], n)
[y_RW[i] = μ_RW[i] + reshape([ϵ_RW[i]], 1,1) for i in 1:n] 

## Getting Matrices to apply kalman filter
T_RW = reshape([1.0], 1, 1)
Z_RW = copy(T_RW)
Q_RW = reshape([σ_η^2], 1, 1)
H_RW = reshape([σ_ϵ^2], 1, 1)

α0_RW = reshape([0.0],1,1);
αhat_RW, v_RW, yhat_RW, P_RW, F_RW, G_RW, K_RW = KalmanFilter(T_RW, Z_RW, Q_RW, H_RW, n, α0_RW, y_RW)

vec_y_RW = [y_RW[i][1, 1] for i in 2:n]
vec_y_hat_RW= [yhat_RW[i][1, 1] for i in 2:n]

plot(vec_y_RW, xlabel = " Time ", label =" Simulated Random Walk ", legend=:bottomright)
plot!(vec_y_hat_RW, label = "Estimate from Kalman Filter")
savefig("figs/RW.png")

## Writing the likelhood function
# For the Random Walk process we need to estimate two Parameters σ_ϵ and σ_η:
function log_L_RW(θ::Vector)  # θ = [log_σ_ϵ, log_σ_η]
    σ_ϵ = exp(θ[1])
    σ_η = exp(θ[2])
    Q1 = reshape([σ_η^2], 1, 1)
    H1 = reshape([σ_ϵ^2], 1, 1)

    v1 = [KalmanFilter(T_RW, Z_RW, Q1, H1, n, α0_RW, y_RW)[2][i][1] for i in 1:n]
    F1 = [KalmanFilter(T_RW, Z_RW, Q1, H1, n, α0_RW, y_RW)[5][i][1] for i in 1:n]
    Ft = abs.(det.(vec(F1))) 

    return sum([-0.5 * log(Ft[i]) - 0.5 * v1[i]' * inv(Ft[i]) * v1[i] for i in 2:n])
end

## Optimization
res_RW = optimize(θ->-log_L_RW(θ), [-0.3, -0.6])
σ_ϵ_hat, σ_η_hat= exp(Optim.minimizer(res_RW)[1]), exp(Optim.minimizer(res_RW)[2]) # (0.70, 0.65)