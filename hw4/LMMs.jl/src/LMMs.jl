module LMMs

using LinearAlgebra, Ipopt, NLopt, MathOptInterface

export LmmObs,
    LmmModel,
    logl!,
    init_ls!,
    fit!,
    ◺,
    CopyMatrix

const MOI = MathOptInterface

struct LmmObs{T <: AbstractFloat}
    # data
    y             :: Vector{T}
    X             :: Matrix{T}
    Z             :: Matrix{T}
    # arrays for gradient
    ∇β            :: Vector{T}
    ∇σ²           :: Vector{T}
    ∇L            :: Matrix{T}
    # arrays for Hessian
    Hββ           :: Matrix{T}
    HLL           :: Matrix{T}
    Hσ²L          :: Matrix{T}
    Hσ²σ²         :: Vector{T}
    # working arrays
    yty           :: T
    xty           :: Vector{T}
    zty           :: Vector{T}
    storage_p     :: Vector{T}
    storage_q     :: Vector{T}
    xtx           :: Matrix{T}
    ztx           :: Matrix{T}
    ztz           :: Matrix{T}
    storage_qq_1  :: Matrix{T}
    LM⁻¹LᵗZᵗZ     :: Matrix{T}
    ZᵗZLM⁻¹LᵗZᵗr  :: Vector{T}
    Zᵗr           :: Vector{T}
    ZᵗΩ⁻¹r        :: Vector{T}
    storage_qp    :: Matrix{T}
    storage_pp    :: Matrix{T}
    storage_qq_2  :: Matrix{T}
    storage_qq_3  :: Matrix{T}
    storage_qq_4  :: Matrix{T}
    storage_qq_5  :: Matrix{T}
    storage_qq_6  :: Matrix{T}
    storage_qq2_1 :: Matrix{T}
    storage_qq2_2 :: Matrix{T}
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
    Hββ           = Matrix{T}(undef, p, p)  
    HLL           = Matrix{T}(undef, ◺(q), ◺(q))
    Hσ²L          = Matrix{T}(undef, q, q)
    Hσ²σ²         = Vector{T}(undef, 1)
    yty           = abs2(norm(y))
    xty           = transpose(X) * y
    zty           = transpose(Z) * y    
    storage_p     = Vector{T}(undef, p)
    storage_q     = Vector{T}(undef, q)
    xtx           = transpose(X) * X
    ztx           = transpose(Z) * X
    ztz           = transpose(Z) * Z
    storage_qq_1  = similar(ztz)
    LM⁻¹LᵗZᵗZ     = Matrix{T}(undef, q, q)
    ZᵗZLM⁻¹LᵗZᵗr  = Vector{T}(undef, q)
    Zᵗr           = Vector{T}(undef, q)
    ZᵗΩ⁻¹r        = Vector{T}(undef, q)
    storage_qp    = Matrix{T}(undef, q, p)
    storage_pp    = Matrix{T}(undef, p, p)
    storage_qq_2  = Matrix{T}(undef, q, q)
    storage_qq_3  = Matrix{T}(undef, q, q)
    storage_qq_4  = Matrix{T}(undef, q, q)
    storage_qq_5  = Matrix{T}(undef, q, q)
    storage_qq_6  = Matrix{T}(undef, q, q)
    storage_qq2_1 = Matrix{T}(undef, abs2(q), abs2(q))
    storage_qq2_2 = Matrix{T}(undef, abs2(q), abs2(q))
    LmmObs(y, X, Z, ∇β, ∇σ², ∇L, Hββ, HLL, Hσ²L, Hσ²σ²,
        yty, xty, zty, storage_p, storage_q, 
        xtx, ztx, ztz, storage_qq_1,
        LM⁻¹LᵗZᵗZ, ZᵗZLM⁻¹LᵗZᵗr, Zᵗr, ZᵗΩ⁻¹r,
        storage_qp, storage_pp, storage_qq_2, storage_qq_3,
        storage_qq_4, storage_qq_5, storage_qq_6,
        storage_qq2_1, storage_qq2_2)
end

struct LmmModel{T <: AbstractFloat} <: MOI.AbstractNLPEvaluator
    # data
    data        :: Vector{LmmObs{T}}
    # parameters
    β           :: Vector{T}
    L           :: Matrix{T}
    σ²          :: Vector{T}
    # arrays for gradient
    ∇β          :: Vector{T}
    ∇σ²         :: Vector{T}
    ∇L          :: Matrix{T}
    # arrays for Hessian
    Hββ         :: Matrix{T}
    HLL         :: Matrix{T}
    Hσ²L        :: Matrix{T}
    Hσ²σ²       :: Vector{T}
    # working arrays
    xty         :: Vector{T}
    ztr2        :: Vector{T}
    xtx         :: Matrix{T}
    ztz2        :: Matrix{T}
    storage_pp  :: Matrix{T}
    storage_qq2 :: Matrix{T}
    ztr         :: Vector{T}
    storage_qq  :: Matrix{T}
end

"""
    LmmModel(data::Vector{LmmObs})

Create an LMM model that contains data and parameters.
"""
function LmmModel(obsvec::Vector{LmmObs{T}}) where T <: AbstractFloat
    # dims
    p             = size(obsvec[1].X, 2)
    q             = size(obsvec[1].Z, 2)
    # parameters
    β             = Vector{T}(undef, p)
    L             = Matrix{T}(undef, q, q)
    σ²            = Vector{T}(undef, 1)    
    # gradients
    ∇β            = similar(β)    
    ∇σ²           = similar(σ²)
    ∇L            = similar(L)
    # Hessians
    Hββ           = Matrix{T}(undef, p, p)  
    HLL           = Matrix{T}(undef, ◺(q), ◺(q))
    Hσ²L          = Matrix{T}(undef, q, q)
    Hσ²σ²         = Vector{T}(undef, 1)
    # working arrays for initialization
    xty           = Vector{T}(undef, p)
    ztr2          = Vector{T}(undef, abs2(q))
    xtx           = Matrix{T}(undef, p, p)
    ztz2          = Matrix{T}(undef, abs2(q), abs2(q))
    storage_pp    = Matrix{T}(undef, p, p)
    storage_qq2   = Matrix{T}(undef, abs2(q), abs2(q))
    ztr           = Vector{T}(undef, q)
    storage_qq    = Matrix{T}(undef, q, q)
    LmmModel(obsvec, β, L, σ², ∇β, ∇σ², ∇L, Hββ, HLL, Hσ²L, Hσ²σ²,
        xty, ztr2, xtx, ztz2,
        storage_pp, storage_qq2, ztr, storage_qq)
end

include("logl.jl")
include("init.jl")
include("nlp.jl")
include("multivariate_calculus.jl")

end