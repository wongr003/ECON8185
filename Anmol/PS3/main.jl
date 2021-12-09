include("functions.jl")

hh = Household();
r_guess = 0.01;
cpol,apol = iterate_egm(hh;r=r_guess)

plot(hh.Amat[:,1], hh.Amat[:,1], label = "", color = :black, linestyle = :dash)
plot!(hh.Amat[:,1], apol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], apol[:,5], label = "ϵ = $(round(hh.Ymat[1,5], digits = 3))")

r = market_clearing(hh,A_guess,r = r_guess)
plot_market_clearing(hh)

