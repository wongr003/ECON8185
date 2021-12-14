## Packages
using NLsolve
using Parameters
using BenchmarkTools
using ForwardDiff

## Files
include("NewtonRoot.jl")
include("Riccati.jl")

######################## Preliminary ###################################
# Construct gradient and hessian at SS
function getM(md,kss,hss)
    @unpack θ,ψ,δ,γ_n,γ_z,ρ,β_hat = md

    f(x::Vector) = log((x[1]^θ)*((exp(x[2])*x[5])^(1-θ))-(1+γ_n)*(1+γ_z)*x[4]+(1-δ)*x[1])+ψ*log(1-x[5]);
    z_bar = [kss,0.0,1.0,kss,hss];
    grad = ForwardDiff.gradient(f,z_bar);
    hess = ForwardDiff.hessian(f,z_bar);

    # Apply Kydland and Prescott's method
    e = [0;0;1;0;0];

    M = e.*(f(z_bar)-grad'*z_bar.+(0.5.*z_bar'*hess*z_bar))*e' +
        0.5*(grad*e'-e*z_bar'*hess-hess*z_bar*e'+e*grad') +
        0.5*hess;

    # Translating M into the matrices we need:
    Q = M[1:3,1:3];
    W = M[1:3,4:5];
    R = M[4:5,4:5];

    A = [0 0 0; 0 ρ 0; 0 0 1];
    B = [1 0; 0 0; 0 0];
    C = [0; 1; 0];

    #Mapping to the problem without discounting (1 VARIABLES ARE ~ IN LECTURE NOTES)
    A_tld = sqrt(β_hat)*(A-B*(R\W'));
    B_tld = sqrt(β_hat)*B;
    Q_tld = Q-W*(R\W');

    return A_tld,B_tld,Q_tld,R,W
end


################### LQ Approximation ##################################
function LQ(A_tld,B_tld,R,Q_tld;tol=1.0e-10)

    Pn, Fn = Riccati(A_tld, B_tld, R, Q_tld, tol);
    F = Fn + R\W';
    P = Pn;

    return P,F
end

################### Vaughan ##################################
function Vaughan(A_tld,B_tld,Q_tld,R,W)

    L = size(A_tld)[1]
    H = [inv(A_tld)  (A_tld\B_tld)*(R\B_tld');
        Q_tld/A_tld Q_tld*(A_tld\B_tld)*(R\B_tld')+A_tld'];

    V = eigen(H).vectors
    #Note that Julia puts the eigenvalues out of the unit circle in the bottom of the matrix,
    #while in the lecture notes they are at the top
    P = V[L+1:end,L+1:end]/(V[1:L,L+1:end]);
    F1 = (R+B_tld'*P*B_tld)\B_tld'*P*A_tld;
    F = F1+R\W';

    return P,F
end

