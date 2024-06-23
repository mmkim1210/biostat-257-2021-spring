module LMM

using LinearAlgebra, DelimitedFiles, Random
using BenchmarkTools, CSV, DataFrames, Distributions, PrettyTables
using Ipopt, NLopt, MathOptInterface, MixedModels

export LmmObs,
    logl!

const MOI = MathOptInterface

struct LmmObs{T <: AbstractFloat}
    # data
    y          :: Vector{T}
    X          :: Matrix{T}
    Z          :: Matrix{T}
    # arrays for holding gradient
    ∇β         :: Vector{T}
    ∇σ²        :: Vector{T}
    ∇L         :: Matrix{T}    
    # working arrays
    yty        :: T
    xty        :: Vector{T}
    zty        :: Vector{T}
    storage_p  :: Vector{T}
    storage_q  :: Vector{T}
    xtx        :: Matrix{T}
    ztx        :: Matrix{T}
    ztz        :: Matrix{T}
    storage_qq :: Matrix{T}
    LM⁻¹LᵗZᵗZ     :: Matrix{T}
    ZᵗZLM⁻¹LᵗZᵗr  :: Vector{T}
    Zᵗr           :: Vector{T}
    ZᵗΩ⁻¹r        :: Vector{T}
end

"""
    LmmObs(y::Vector, X::Matrix, Z::Matrix)

Create an LMM datum of type `LmmObs`.
"""
function LmmObs(
        y::Vector{T}, 
        X::Matrix{T}, 
        Z::Matrix{T}
    ) where T <: AbstractFloat
    n, p, q       = size(X, 1), size(X, 2), size(Z, 2)    
    ∇β            = Vector{T}(undef, p)
    ∇σ²           = Vector{T}(undef, 1)
    ∇L            = Matrix{T}(undef, q, q)    
    yty           = abs2(norm(y))
    xty           = transpose(X) * y
    zty           = transpose(Z) * y    
    storage_p     = Vector{T}(undef, p)
    storage_q     = Vector{T}(undef, q)
    xtx           = transpose(X) * X
    ztx           = transpose(Z) * X
    ztz           = transpose(Z) * Z
    storage_qq    = similar(ztz)
    LM⁻¹LᵗZᵗZ     = Matrix{T}(undef, q, q)
    ZᵗZLM⁻¹LᵗZᵗr  = Vector{T}(undef, q)
    Zᵗr           = Vector{T}(undef, q)
    ZᵗΩ⁻¹r        = Vector{T}(undef, q)
    LmmObs(y, X, Z, ∇β, ∇σ², ∇L, 
        yty, xty, zty, storage_p, storage_q, 
        xtx, ztx, ztz, storage_qq,
        LM⁻¹LᵗZᵗZ, ZᵗZLM⁻¹LᵗZᵗr, Zᵗr, ZᵗΩ⁻¹r)
end

include("logl.jl")

end