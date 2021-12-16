using NLsolve

include("functions_withG.jl")

hh = Household();
r = 0.019;
A = 1.72;
T = 0.093;
ϕ = 342;
B = 1.0;

@unpack θ,δ,Amat,Ymat,na,nϵ,τ,γ,η,bc = hh
w = A_guess*(1-θ)*((r+δ)/(A*θ))^(θ/(θ-1));
cnext = @. r*Amat+w*Ymat+T

function getcBinding(a,c,ϵ)
    @. (1-τ)*w*ϵ*getl(c,ϵ,γ,ϕ,η,τ,w)+T+(1+(1-τ)*r)*a-bc-c;
end

cbinding = similar(Amat);

for i = 1:na
    for j = 1:nϵ
        cbinding[i,j] = NewtonRoot(c -> getcBinding(Amat[i,j],c,Ymat[i,j]),1.0);
    end
end





cpol,apol,lpol = egm(hh,T_guess,ϕ_guess,w=w,cnext = cnext,cbinding=cbinding,r=r,rnext=r)

cpol,apol,lpol = iterate_egm(hh,A_guess,T_guess,ϕ_guess;r=r_guess) 

plot(hh.Amat[1:10,1], hh.Amat[1:10,1], label = "", color = :black, linestyle = :dash)
plot!(hh.Amat[1:10,1], apol[1:10,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[1:10,1], apol[1:10,5], label = "ϵ = $(round(hh.Ymat[1,5], digits = 3))")

plot(hh.Amat[:,1], lpol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], lpol[:,2], label = "ϵ = $(round(hh.Ymat[1,2], digits = 3))")

r,K,N,w = market_clearing(hh,A_guess,r_guess,T_guess,B_guess,ϕ_guess)
plot_market_clearing(hh,A_guess,T_guess,B_guess,ϕ_guess)

function f!(F,x)
    r = x[1]
    A = x[2] 
    T = x[3]
    ϕ = x[4]
    Y = 1.0
    θ = 0.3
    G = 0.2
    B = 1.0
    τ = 0.4
    r_eq,K,N,w = market_clearing(hh,A,r,T,B,ϕ)
    F[1] = r_eq-r
    F[2] = Y-A*(K^θ)*(N^(1-θ))
    F[3] = G+T+r-τ*(w*N+r*(K+B))
    F[4] = N-0.28
end

SS = nlsolve(f!,[0.012,5.0,0.05,30.0])