"""
    logl!(obs::LmmObs, β, L, σ², needgrad = false)

Evaluate the log-likelihood of a single LMM datum at parameter values `β`, `L`, 
and `σ²`. If `needgrad == true`, then `obs.∇β`, `obs.∇L`, and `obs.σ²` are filled 
with the corresponding gradient.
"""
function logl!(
        obs      :: LmmObs{T}, 
        β        :: Vector{T}, 
        L        :: Matrix{T}, 
        σ²       :: T,
        needgrad :: Bool = true
    ) where T <: AbstractFloat
    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    ####################
    # Evaluate objective
    ####################    
    # form the q-by-q matrix: M = σ² * I + Lᵗ Zᵗ Z L
    copy!(obs.storage_qq, obs.ztz)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qq) # O(q^3)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.storage_qq) # O(q^3)
    @inbounds for j in 1:q
        obs.storage_qq[j, j] += σ²
    end
    # cholesky on M = σ² * I + Lᵗ Zᵗ Z L
    LAPACK.potrf!('U', obs.storage_qq) # O(q^3)
    # storage_q = (Mchol.U') \ (Lt * (Zt * res))
    BLAS.gemv!('N', T(-1), obs.ztx, β, T(1), copy!(obs.Zᵗr, obs.zty)) # O(pq)
    BLAS.trmv!('L', 'T', 'N', L, copy!(obs.storage_q, obs.Zᵗr)) # O(q^2)
    BLAS.trsv!('U', 'T', 'N', obs.storage_qq, obs.storage_q) # O(q^3)
    # l2 norm of residual vector
    copy!(obs.storage_p, obs.xty)
    rtr  = obs.yty +
    dot(β, BLAS.gemv!('N', T(1), obs.xtx, β, T(-2), obs.storage_p))
    # assemble pieces
    logl::T = n * log(2π) + (n - q) * log(σ²) # constant term
    @inbounds for j in 1:q
        logl += 2log(obs.storage_qq[j, j])
    end
    qf    = abs2(norm(obs.storage_q)) # quadratic form term
    logl += (rtr - qf) / σ² 
    logl /= -2
    ###################
    # Evaluate gradient
    ###################    
    if needgrad
        # TODO: fill ∇β, ∇σ², ∇L by gradients
        # compute ∇β
        copy!(obs.∇β, obs.xty) 
        BLAS.gemv!('N', T(-1), obs.xtx, β, T(1), obs.∇β)
        BLAS.trsv!('U', 'N', 'N', obs.storage_qq, obs.storage_q)
        BLAS.trmv!('L', 'N', 'N', L, obs.storage_q) # LM⁻¹LᵗZᵗr
        BLAS.gemv!('T', T(-1), obs.ztx, obs.storage_q, T(1), obs.∇β)
        obs.∇β ./= σ²
        # compute ∇σ² 
        obs.∇σ²[1] = - n 
        BLAS.gemm!('T', 'N', T(1), L, obs.ztz, T(0), obs.LM⁻¹LᵗZᵗZ)
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq, obs.LM⁻¹LᵗZᵗZ)
        BLAS.trsm!('L', 'U', 'N', 'N', T(1), obs.storage_qq, obs.LM⁻¹LᵗZᵗZ)
        BLAS.trmm!('L', 'L', 'N', 'N', T(1), L, obs.LM⁻¹LᵗZᵗZ)
        obs.∇σ²[1] += tr(obs.LM⁻¹LᵗZᵗZ)
        BLAS.gemv!('N', T(1), obs.ztz, obs.storage_q, T(0), obs.ZᵗZLM⁻¹LᵗZᵗr)
        obs.∇σ²[1] += (rtr - 2 * qf + dot(obs.storage_q, obs.ZᵗZLM⁻¹LᵗZᵗr)) / σ²
        obs.∇σ²[1] /= (2 * σ²)
        # compute ∇L
        copy!(obs.∇L, obs.ztz)
        BLAS.gemm!('N', 'N', T(1), obs.ztz, obs.LM⁻¹LᵗZᵗZ, T(-1), obs.∇L)
        obs.∇L ./= σ²
        obs.ZᵗΩ⁻¹r .= (obs.Zᵗr .- obs.ZᵗZLM⁻¹LᵗZᵗr) ./ σ²
        BLAS.ger!(T(1), obs.ZᵗΩ⁻¹r, obs.ZᵗΩ⁻¹r, obs.∇L)
        # L is multiplied later to reduce flops
    end
    logl
end
