include("functions_endoL.jl")

hh = Household();
r_guess = 0.005;
A_guess = 0.45726596004725534;
cpol,apol,lpol = iterate_egm(hh,A_guess;r=r_guess)

plot(hh.Amat[:,1], hh.Amat[:,1], label = "", color = :black, linestyle = :dash)
plot!(hh.Amat[:,1], apol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], apol[:,5], label = "ϵ = $(round(hh.Ymat[1,5], digits = 3))")

plot(hh.Amat[:,1], lpol[:,1], label = "ϵ = $(round(hh.Ymat[1,1], digits = 3))")
plot!(hh.Amat[:,1], lpol[:,2], label = "ϵ = $(round(hh.Ymat[1,2], digits = 3))")

# Calibrating A such that Y = 1, A = 0.45726596004725534
dist_Y = 10;
while dist_Y > 10^-5
    r,Y = market_clearing(hh,A_guess,r=r_guess);
    dist_Y = abs(Y-1);
    A_guess = 0.5*A_guess+0.5*A_guess/Y;
    println("Y is $Y, new A is $A_guess")
end

market_clearing(hh,A_guess)
plot_market_clearing(hh,A_guess)