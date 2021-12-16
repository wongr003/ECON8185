## Packages
using Parameters
using BasisMatrices

Household = @with_kw (na, # Number of assets
    amax, # asset max
    aGrid, # asset grid
    basis, # basis type
    bs, # basis matrix, not expanded
    lGrid, # labor grid
    nl, # number of labor states
    nodes, # tensor product of aGrid, lGrid
    P_l = [0.5 0.5; 0.5 0.5], # transition matrix for labor
    β = 0.96, # discount factor
    μ = 3.0, # elasticity of substitution
    w = 1.0, # wage
    R = 1.02) # gross interest rate

function construct_household(na = 10,amax = 70.0)
    aGrid = LinRange(sqrt(0.01), sqrt(amax), na).^2;
    lGrid = [0.2,0.9];
    nl = length(lGrid);

    a_basis = Basis(SplineParams(aGrid,0,1));
    l_basis = Basis(SplineParams(lGrid,0,1));

    basis = Basis(a_basis,l_basis);

    s,(aGrid,lGrid) = nodes(basis);

    bs = BasisMatrix(basis,Direct());

    hh = Household(na=na,amax=amax,aGrid=aGrid,basis=basis,bs=bs,lGrid=lGrid,nl=nl,nodes=s)

    return hh
end

# Initial Guess for theta with vectorization
# of nodes = 16 (2*10)
function guessθ(hh)
    @unpack na,nl,R,w,aGrid,lGrid = hh
    θ = zeros(na*nl,);
    for i = 1:na
        for j = 1:nl
            ij = (j-1)*na+i
            θ[ij] = 0.9*(R*aGrid[i]+w*lGrid[j])
        end
    end
    return θ
end



aGrid = LinRange(sqrt(0.01), sqrt(70.0), 10).^2;
a_basis = Basis(SplineParams(aGrid,0,1));
S,aGrid = nodes(a_basis)
Φ = BasisMatrix(a_basis, Expanded(), S, 0)
Φ.vals[1]*ones(10).+1