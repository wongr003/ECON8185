using NLsolve

include("functions_withG.jl")

hh = Household();
r_guess = 0.019061865234375003;
A_guess = 0.46077647226533514;
T_guess = 0.0932067257559209;
B_guess = 1.0;
cpol,apol,lpol = iterate_egm(hh,A_guess,0.1;r=r_guess) 

plot(hh.Amat[:,1], hh.Amat[:,1], label = "", color = :black, linestyle = :dash)
plot!(hh.Amat[:,1], apol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], apol[:,5], label = "ϵ = $(round(hh.Ymat[1,5], digits = 3))")

plot(hh.Amat[:,1], lpol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], lpol[:,2], label = "ϵ = $(round(hh.Ymat[1,2], digits = 3))")

r,K,N,w = market_clearing(hh,A_guess,r_guess,T_guess,B_guess)
plot_market_clearing(hh,A_guess,T_guess,B_guess)

function f!(F,x)
    r = x[1]
    A = x[2] 
    T = x[3]
    Y = 1.0
    θ = 0.3
    G = 0.2
    B = 1.0
    τ = 0.4
    r_eq,K,N,w = market_clearing(hh,A,r,T,B)
    F[1] = r_eq-r
    F[2] = Y-A*(K^θ)*(N^(1-θ))
    F[3] = G+T+r-τ*(w*N+r*(K+B))
end

SS = nlsolve(f!,[0.019,0.46,0.1])