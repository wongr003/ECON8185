## Packages
using NLsolve
using SymPy
using ForwardDiff
using Plots

## Files
include("NewtonRoot.jl")
include("Riccati.jl")

## Parameters
θ = 0.35; # capital share
δ = 0.0464; # depreciation rate
γ_z = 0.016; # growth rate of productivity
γ_n = 0.015; # growth rate of population
β = 0.9722; # discount rate
β_hat = β*(1+γ_n); # detrended discount rate
ψ = 2.24; # labor coefficient
ρ = 0.2; # autocorr of productivity process

## Get the Steady-State level of capital from a Nonlinear solver
function ee!(eq,x)
    k=(x[1])
    h=(x[2])
    eq[1]=β_hat*(θ*k^(θ-1)*h^(1-θ)+1-δ)-(1+γ_n)*(1+γ_z)
    eq[2]=(-(ψ/(1-h)))+((1-θ)*(k^θ)*(h^(-θ)))/((k^θ)*(h^(1-θ))-(1+γ_n)*(1+γ_z)*k+(1-δ)*k)
end

S = nlsolve(ee!, [0.5,0.5],ftol = :1.0e-9, method = :trust_region , autoscale = true);
kss = S.zero[1];
hss = S.zero[2];
lss = 1-hss;
css = (kss^θ)*((exp(0)*hss)^(1-θ))-(1+γ_n)*(1+γ_z)*kss+(1-δ)*kss;

# Construct gradient and hessian at SS
f(x::Vector) = log((x[1]^θ)*((exp(x[2])*x[5])^(1-θ))-(1+γ_n)*(1+γ_z)*x[4]+(1-δ)*x[1])+ψ*log(1-x[5])
z_bar = [kss,0.0,1.0,kss,hss]
grad = ForwardDiff.gradient(f,z_bar)
hess = ForwardDiff.hessian(f,z_bar)

# Apply Kydland and Prescott's method
e = [0;0;1;0;0];

M = e.*(f(z_bar)-grad'*z_bar.+(0.5.*z_bar'*hess*z_bar))*e' +
    0.5*(grad*e'-e*z_bar'*hess-hess*z_bar*e'+e*grad') +
    0.5*hess;

# Translating M into the matrices we need:
Q = M[1:3,1:3]
W = M[1:3,4:5]
R = M[4:5,4:5]

A = [0 0 0; 0 ρ 0; 0 0 1];
B = [1 0; 0 0; 0 0];
C = [0; 1; 0];

#Mapping to the problem without discounting (1 VARIABLES ARE ~ IN LECTURE NOTES)
A_tld = sqrt(β_hat)*(A-B*(R\W'));
B_tld = sqrt(β_hat)*B;
Q_tld = Q-W*(R\W');

tol = 1.0e-10;
Pn, Fn = Riccati(A_tld, B_tld, R, Q_tld, tol);
F = Fn + R\W';
P = Pn;

# Constructing Policy function
nk = 100;
kGrid = LinRange(0.5*kss, 1.5*kss, nk);

# Low shock
pol_L = zeros(2, 1, 100);
for i = 1:nk
    pol_L[:,:,i] = -F*[kGrid[i]; -0.5; 1];
end

kpol_L = zeros(1,nk);
hpol_L = zeros(1,nk);

for i = 1:nk
    kpol_L[1,i] = pol_L[1,:,i][1];
    hpol_L[1,i] = pol_L[2,:,i][1];
end

# High shock
pol_H = zeros(2, 1, 100);
for i = 1:nk
    pol_H[:,:,i] = -F*[kGrid[i]; 0.5; 1];
end

kpol_H = zeros(1,nk);
hpol_H = zeros(1,nk);

for i = 1:nk
    kpol_H[1,i] = pol_H[1,:,i][1];
    hpol_H[1,i] = pol_H[2,:,i][1];
end

# SS shock
pol_SS = zeros(2, 1, 100);
for i = 1:nk
    pol_SS[:,:,i] = -F*[kGrid[i]; 0.0; 1];
end

kpol_SS = zeros(1,nk);
hpol_SS = zeros(1,nk);

for i = 1:nk
    kpol_SS[1,i] = pol_SS[1,:,i][1];
    hpol_SS[1,i] = pol_SS[2,:,i][1];
end

plot(kGrid,kGrid,label = "", color = :black, linestyle = :dash)
plot!(kGrid,vec(kpol_L),label = "Low",legend=:topleft)
plot!(kGrid,vec(kpol_H),label = "High")
plot!(kGrid,vec(kpol_SS),label = "SS")

plot(kGrid,vec(hpol_L),label = "Low",legend=:topright)
plot!(kGrid,vec(hpol_H),label = "High")
plot!(kGrid,vec(hpol_SS),label = "SS")