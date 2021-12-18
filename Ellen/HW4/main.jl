## Packages
using Plots

## Files
include("FEM_aiyagari.jl")
include("NewtonRoot_mult.jl")

############ Plot basis functions #################
plot(x -> ψ(x, [0, 1, 3, 6],1), xlim=(0, 6), label = "ψ_1(x)")
plot!(x -> ψ(x, [0, 1, 3, 6], 2), xlim=(0, 6), label = "ψ_2(x)")
plot!(x -> ψ(x, [0, 1, 3, 6], 3), xlim=(0, 6), label = "ψ_3(x)")
plot!(x -> ψ(x, [0, 1, 3, 6], 4), xlim=(0, 6), label = "ψ_4(x)")
savefig("figs/linear_basis.png")

###################### Aiyagari ###########################
hh = Household()
θ_init = [hh.aGrid; hh.aGrid .+ 0.5]
θ_est = NewtonRoot_mult(θ->G(hh,θ),θ_init)

plot(a->a,xlim = (0.0,6.1),label = "45 degree", color = :black, linestyle = :dash,legend =:topleft)
plot!(a->get_apol(hh,a,hh.lGrid[1],θ_est),xlim = (0.0,6.1),label = "low")
plot!(a->get_apol(hh,a,hh.lGrid[2],θ_est),xlim = (0.0,6.1),label = "high")
scatter!(hh.aGrid,θ_est[1:8],label = "coefficient of basis (Low)")
scatter!(hh.aGrid,θ_est[9:16],label = "coefficient of basis (High)")
savefig("figs/aiyagari.png")




