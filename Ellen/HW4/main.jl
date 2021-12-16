## Files
include("FEM.jl")

hh = construct_household()
θ_guess = guessθ(hh)

S = hh.nodes
Ψ = BasisMatrix(hh.basis, Expanded(), S, 0)
Ψ.vals[1].*θ_guess[:,1]

apol = funeval(θ_guess,hh.basis,S)