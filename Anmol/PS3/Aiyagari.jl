using LinearAlgebra
using Parameters
using IterativeSolvers
using FastGaussQuadrature
using ForwardDiff
using QuantEcon
using Plots
using Arpack
using BenchmarkTools

################################### Model types #########################

uPrime(c,γ) = c.^(-γ)
uPrimeInv(up,γ) = up.^(-1.0/γ)

struct AiyagariParametersEGM{T <: Real}
    β::T
    α::T
    δ::T
    γ::T
    ρ::T
    σz::T #st. deviation of Z shock
    σ::T #job separation
    lamw::T #job finding prob
    Lbar::T
    amin::T
    Penalty::T
end

struct AiyagariModelEGM{T <: Real,I <: Integer}
    params::AiyagariParametersEGM{T}
    aGrid::Array{T,1} ##Policy grid
    aGridl::Array{T,1}
    na::I ##number of grid points in policy function
    dGrid::Array{T,1}
    nd::I ##number of grid points in distribution
    states::Array{T,1} ##earning states 
    ns::I ##number of states
    EmpTrans::Array{T,2}
end
mutable struct AggVarsEGM{S <: Real,T <: Real}
    R::S
    w::T
end

function PricesEGM(K,Z,params::AiyagariParametersEGM)
    @unpack β,α,δ,γ,ρ,σ,lamw,Lbar = params
    R = Z*α*(K/Lbar)^(α-1.0) + 1.0 - δ
    w = Z*(1.0-α)*(K/Lbar)^(1.0-α) 
    
    return AggVarsEGM(R,w)
end

function AiyagariEGM(
    K::T,
    β::T = 0.98,
    α::T = 0.4,
    δ::T = 0.02,
    γ::T = 2.0,
    ρ::T = 0.95,
    σz::T = 1.0,
    σ::T = 0.2,
    lamw::T = 0.6,
    Lbar::T = 1.0,
    amin::T = 1e-9,
    amax::T = 200.0,
    Penalty::T = 1000000000.0,
    na::I = 201,
    nd::I = 201,
    ns::I = 2,
    endow = [1.0;2.5]) where{T <: Real,I <: Integer}

    #############Params
    params = AiyagariParametersEGM(β,α,δ,γ,ρ,σz,σ,lamw,Lbar,amin,Penalty)
    AggVars = PricesEGM(K,1.0,params)
    @unpack R,w = AggVars

    ################## Policy grid
    function grid_fun(a_min,a_max,na, pexp)
        x = range(a_min,step=0.5,length=na)
        grid = a_min .+ (a_max-a_min)*(x.^pexp/maximum(x.^pexp))
        return grid
    end
    aGrid = grid_fun(amin,amax,na,4.0)
    ################### Distribution grid
    #dGrid = collect(range(aGrid[1],stop = aGrid[end],length = nd))
    #aGrid = collect(range(amin,stop = amax,length = na))
    dGrid=aGrid

    ################## Transition
    EmpTrans = [1.0-lamw σ ;lamw 1.0-σ]
    dis = LinearAlgebra.eigen(EmpTrans)
    mini = argmin(abs.(dis.values .- 1.0)) 
    stdist = abs.(dis.vectors[:,mini]) / sum(abs.(dis.vectors[:,mini]))
    lbar = dot(stdist,endow)
    states = endow/lbar

    @assert sum(EmpTrans[:,1]) == 1.0 ###sum to 1 across rows
    #summing across rows is nice as we don't need to transpose transition before taking eigenvalue
    
    guess = vcat(10.0 .+ aGrid,10.0 .+ aGrid)

    
    return AiyagariModelEGM(params,aGrid,vcat(aGrid,aGrid),na,dGrid,nd,states,ns,EmpTrans),guess,AggVars
end

function interpEGM(pol::AbstractArray,
                grid::AbstractArray,
                x::T,
                na::Integer) where{T <: Real}
    np = searchsortedlast(pol,x)

    ##Adjust indices if assets fall out of bounds
    (np > 0 && np < na) ? np = np : 
        (np == na) ? np = na-1 : 
            np = 1        
    #@show np
    ap_l,ap_h = pol[np],pol[np+1]
    a_l,a_h = grid[np], grid[np+1] 
    ap = a_l + (a_h-a_l)/(ap_h-ap_l)*(x-ap_l) 
    
    above =  ap > 0.0 
    return above*ap,np
end

function get_cEGM(pol::AbstractArray,
               Aggs::AggVarsEGM,
               CurrentAssets::AbstractArray,
               AiyagariModel::AiyagariModelEGM,
               cpol::AbstractArray) 
    
    @unpack aGrid,na,ns,states = AiyagariModel
    pol = reshape(pol,na,ns)
    for si = 1:ns
        for ai = 1:na
            asi = (si - 1)*na + ai
            cpol[asi] = Aggs.R*CurrentAssets[asi] + Aggs.w*states[si] - interpEGM(pol[:,si],aGrid,CurrentAssets[asi],na)[1]
        end
    end
    return cpol
end

function EulerBackEGM(pol::AbstractArray,
                   Aggs::AggVarsEGM,
                   Aggs_P::AggVarsEGM,
                   AiyagariModel::AiyagariModelEGM,
                   cpol::AbstractArray,
                   apol::AbstractArray)
    
    @unpack params,na,nd,ns,aGridl,EmpTrans,states = AiyagariModel
    @unpack γ,β = params

    R_P,w_P = Aggs_P.R,Aggs_P.w
    R,w = Aggs.R,Aggs.w
    
    cp = get_cEGM(pol,Aggs_P,aGridl,AiyagariModel,cpol)
    upcp = uPrime(cp,γ)
    Eupcp = copy(cpol)
    #Eupcp_sp = 0.0

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na + ai
            Eupcp_sp = 0.0
            for spi = 1:ns
                aspi = (spi-1)*na + ai
                Eupcp_sp += EmpTrans[spi,si]*upcp[aspi]
            end
            Eupcp[asi] = Eupcp_sp 
        end
    end

    upc = R_P*β*Eupcp

    c = uPrimeInv(upc,γ)

    for ai = 1:na
        for si = 1:ns
            asi = (si-1)*na+ai
            apol[asi] = (aGridl[asi] + c[asi] - w*states[si])/R
        end
    end

    return apol,c
end


function SolveEGM(pol::AbstractArray,
                  Aggs::AggVarsEGM,
                  AiyagariModel::AiyagariModelEGM,
                  cpol::AbstractArray,
                  apol::AbstractArray,tol = 1e-17)
    @unpack ns,na = AiyagariModel

    for i = 1:10000
        a = EulerBackEGM(pol,Aggs,Aggs,AiyagariModel,cpol,apol)[1]
        if (i-1) % 50 == 0
            test = abs.(a - pol)/(abs.(a) + abs.(pol))
            #println("iteration: ",i," ",maximum(test))
            if maximum(test) < tol
                println("Solved in ",i," ","iterations")
                break
            end
        end
        pol = copy(a)
    end
    return pol
end

K0 = 48.0
AA0, pol0, Aggs0 = AiyagariEGM(K0)
SolveEGM(pol0, Aggs0, AA0, cpol, apol)

na = 201
ns = 2
cpol = zeros(eltype(pol0),na*ns)
apol = cpol