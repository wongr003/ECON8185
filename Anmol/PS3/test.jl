include("functions.jl")
include("NewtonRoot.jl")

## Test Code ##

# Check policy functions for c and a
r0 = 0.015;
AA0, cpol0 = AiyagariEGM(r0);
abar0 = zeros(length(cpol0));
cpol_EGM, apol_EGM = SolveEGM(AA0,cpol0,abar0);

plot(AA0.aGrid, AA0.aGrid, label = "", color = :black, linestyle = :dash)
plot!(AA0.aGrid, apol_EGM[1:10,:], label = "ϵ = $(round(AA0.ϵGrid[1], digits = 3))")
plot!(AA0.aGrid, apol_EGM[41:50,:], label = "ϵ = $(round(AA0.ϵGrid[5], digits = 3))")

# Check stationary distribution
λ_init_test = zeros(AA0.na*AA0.nϵ) .+ 1/(AA0.na*AA0.nϵ);
Qmat_test = zeros(AA0.na*AA0.nϵ,AA0.na*AA0.nϵ);
Qmat_test = MakeTransMatEGM(AA0,apol_EGM,Qmat_test);
λ_test = StationaryDistributionEGM(Qmat_test,λ_init_test)

sum(λ_test) # all elements sum to one
dot(λ_test, AA0.aGridl) # aggregate asset
dot(λ_test, apol_EGM)
dot(λ_test, repeat(AA0.ϵGrid, inner = AA0.na)) # aggregate labor

# Check equilibrium of the model
equilibriumEGM(AA0,cpol0,abar0) # Still get negative r at equilibrium


# Check aggregate demand and aggregate supply of capital
r_vals = LinRange(0.0, 1/0.98-1,100)
Ar_vals = zeros(length(r_vals))
Nbar_vals = zeros(length(r_vals))
K_vals = zeros(length(r_vals))

for i = 1:length(r_vals)
    r0 = r_vals[i];
    AA0, cpol0 = AiyagariEGM(r0);
    abar0 = zeros(length(cpol0));
    cpol_EGM, apol_EGM = SolveEGM(AA0,cpol0,abar0);

    λ_init_test = zeros(AA0.na*AA0.nϵ) .+ 1/(AA0.na*AA0.nϵ);
    Qmat_test = zeros(AA0.na*AA0.nϵ,AA0.na*AA0.nϵ);
    Qmat_test = MakeTransMatEGM(AA0,apol_EGM,Qmat_test);
    λ_test = StationaryDistributionEGM(Qmat_test,λ_init_test)
    
    Ar_vals[i] = dot(λ_test, AA0.aGridl)
    Nbar_vals[i] = dot(λ_test, repeat(AA0.ϵGrid, inner = AA0.na));
    K_vals[i] = ((r0+AA0.params.δ)/(AA0.params.θ*Nbar_vals[i]^(1-AA0.params.θ)))^(1/(AA0.params.θ-1));
end

plot(S_vals,r_vals)
plot!(K_vals,r_vals)


# Incorporate elastic labor supply
using ForwardDiff
U(c,l) = (((c^0.3)*((l)^(1-0.3)))^(1-1.5))/(1-1.5);
Uc0(c,l) = ForwardDiff.derivative(c -> U(c,l),c);
Uc = l -> Uc0(c,l)
Ul0(c,l) = ForwardDiff.derivative(l -> U(c,l), l);
Ul = l -> Ul0(c,l)

c = 2.0
NewtonRoot(l -> Ul(l)/Uc(l) - 2.0, 1)


#F(c,l) = c^2*l^2

#Fc(c,l) = ForwardDiff.derivative(c -> F(c,l), c)
#Fc_new = l -> Fc(c,l) # Fc with only function of l
#Fl(c,l) = ForwardDiff.derivative(l -> F(c,l), l)
#Fl_new = l -> Fl(c,l)

#c = 1.5
#NewtonRoot(l -> Fl_new(l)/Fc_new(l) - 2.0, 1)