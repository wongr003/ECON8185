include("functions_withG.jl")

hh = Household();
r_guess = 0.0001;
A_guess = 0.45726596004725534;
cpol,apol,lpol = iterate_egm(hh,A_guess;r=r_guess) # Problem is cbinding is too high, so the new c vector is not sorted 