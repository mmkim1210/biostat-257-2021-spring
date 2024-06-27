"""
    logl!(obs::LmmObs, β, L, σ², needgrad = true, needhess = false)

Evaluate the log-likelihood of a single LMM datum at parameter values `β`, `L`, 
and `σ²`. If `needgrad == true`, then `obs.∇β`, `obs.∇L`, and `obs.σ²` are filled 
with the corresponding gradient. If `needhess == true`, then `obs.Hββ`, `obs.HLL`,
`obs.Hσ²L`, and `obs.Hσ²σ²` are filled with the corresponding Hessian.
"""
function logl!(
        obs      :: LmmObs{T}, 
        β        :: Vector{T}, 
        L        :: Matrix{T}, 
        σ²       :: T,
        needgrad :: Bool = true,
        needhess :: Bool = false
    ) where T <: AbstractFloat
    n, p, q = size(obs.X, 1), size(obs.X, 2), size(obs.Z, 2)
    ####################
    # Evaluate objective
    ####################    
    # Ω = σ²In + ZLLᵗZᵗ, Ω⁻¹ = σ⁻²In - σ⁻²ZL(σ²Iq + LᵗZᵗZL)⁻¹LᵗZᵗ
    # form the q-by-q matrix: M = σ²Iq + LᵗZᵗZL
    # det(Ω) = det(σ⁻²M)det(Iq)det(σ²In)
    copy!(obs.storage_qq_1, obs.ztz)
    BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qq_1) # O(q^3)
    BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.storage_qq_1) # O(q^3)
    @inbounds for j in 1:q
        obs.storage_qq_1[j, j] += σ²
    end
    # cholesky on M
    LAPACK.potrf!('U', obs.storage_qq_1) # O(q^3)
    # storage_q = (Mchol.U') \ (Lt * (Zt * res))
    BLAS.gemv!('N', T(-1), obs.ztx, β, T(1), copy!(obs.Zᵗr, obs.zty)) # O(pq)
    BLAS.trmv!('L', 'T', 'N', L, copy!(obs.storage_q, obs.Zᵗr)) # O(q^2)
    BLAS.trsv!('U', 'T', 'N', obs.storage_qq_1, obs.storage_q) # O(q^3)
    # l2 norm of residual vector
    copy!(obs.storage_p, obs.xty)
    rtr  = obs.yty +
        dot(β, BLAS.gemv!('N', T(1), obs.xtx, β, T(-2), obs.storage_p))
    # assemble pieces
    logl::T = n * log(2π) + (n - q) * log(σ²) # constant term
    @inbounds for j in 1:q
        logl += 2log(obs.storage_qq_1[j, j])
    end
    qf    = abs2(norm(obs.storage_q)) # quadratic form term, rᵗZLM⁻¹LᵗZᵗr
    logl += (rtr - qf) / σ² 
    logl /= -2
    ###################
    # Evaluate gradient
    ###################    
    if needgrad
        # compute ∇β
        copy!(obs.∇β, obs.xty) 
        BLAS.gemv!('N', T(-1), obs.xtx, β, T(1), obs.∇β)
        BLAS.trsv!('U', 'N', 'N', obs.storage_qq_1, obs.storage_q)
        BLAS.trmv!('L', 'N', 'N', L, obs.storage_q) # LM⁻¹LᵗZᵗr
        BLAS.gemv!('T', T(-1), obs.ztx, obs.storage_q, T(1), obs.∇β)
        obs.∇β ./= σ²
        # compute ∇σ² 
        obs.∇σ²[1] = - n
        copy!(obs.LM⁻¹LᵗZᵗZ, obs.ztz)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.LM⁻¹LᵗZᵗZ)
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.LM⁻¹LᵗZᵗZ)
        BLAS.trsm!('L', 'U', 'N', 'N', T(1), obs.storage_qq_1, obs.LM⁻¹LᵗZᵗZ)
        BLAS.trmm!('L', 'L', 'N', 'N', T(1), L, obs.LM⁻¹LᵗZᵗZ)
        obs.∇σ²[1] += tr(obs.LM⁻¹LᵗZᵗZ) # tr(Ω⁻¹)
        BLAS.gemv!('N', T(1), obs.ztz, obs.storage_q, T(0), obs.ZᵗZLM⁻¹LᵗZᵗr)
        obs.∇σ²[1] += (rtr - 2 * qf + dot(obs.storage_q, obs.ZᵗZLM⁻¹LᵗZᵗr)) / σ² # dot(rᵗZLM⁻¹Lᵗ, ZᵗZLM⁻¹LᵗZᵗr)
        obs.∇σ²[1] /= (2 * σ²)
        # compute ∇L
        copy!(obs.∇L, obs.ztz)
        BLAS.gemm!('N', 'N', T(1), obs.ztz, obs.LM⁻¹LᵗZᵗZ, T(-1), obs.∇L) # ZᵗΩ⁻¹Z
        obs.∇L ./= σ²
        obs.ZᵗΩ⁻¹r .= (obs.Zᵗr .- obs.ZᵗZLM⁻¹LᵗZᵗr) ./ σ²
        BLAS.ger!(T(1), obs.ZᵗΩ⁻¹r, obs.ZᵗΩ⁻¹r, obs.∇L) # ZᵗΩ⁻¹rrᵗΩ⁻¹Z
        # L is multiplied later to reduce flops
    end
    ###################
    # Evaluate Hessian
    ###################    
    if needhess
        # compute Hββ
        copy!(obs.Hββ, obs.xtx)
        copy!(obs.storage_qp, obs.ztx)
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qp)
        BLAS.trsm!('L', 'U', 'T', 'N', T(1), obs.storage_qq_1, obs.storage_qp)
        BLAS.gemm!('T', 'N', T(1), obs.storage_qp, obs.storage_qp, T(0), obs.storage_pp) # XᵗZLM⁻¹LᵗZᵗX
        obs.Hββ .= (obs.Hββ .- obs.storage_pp) ./ (T(-1) * σ²)
        # compute Hσ²L
        copy!(obs.storage_qq_2, obs.ztz)
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.storage_qq_2) # ZᵗZL
        mul!(obs.storage_qq_3, obs.ztz, obs.LM⁻¹LᵗZᵗZ) # ZᵗZLM⁻¹LᵗZᵗZ
        copy!(obs.storage_qq_4, obs.storage_qq_3)
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.storage_qq_4) # ZᵗZLM⁻¹LᵗZᵗZL
        mul!(obs.storage_qq_5, obs.storage_qq_3, obs.LM⁻¹LᵗZᵗZ) # ZᵗZLM⁻¹LᵗZᵗZLM⁻¹LᵗZᵗZ
        copy!(obs.storage_qq_6, obs.storage_qq_5)
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), L, obs.storage_qq_6) # ZᵗZLM⁻¹LᵗZᵗZLM⁻¹LᵗZᵗZL
        obs.Hσ²L .= obs.storage_qq_2 .- 2 .* obs.storage_qq_4 .+ obs.storage_qq_6
        @inbounds for j in 1:(q - 1), i in (j + 1):q
            obs.Hσ²L[i, j] += obs.Hσ²L[j, i]
        end
        obs.Hσ²L ./= (T(-1) * abs2(σ²))
        # compute HLL
        obs.storage_qq_5 .= obs.storage_qq_2 .- obs.storage_qq_4
        kron!(obs.storage_qq2_1, transpose(obs.storage_qq_5), obs.storage_qq_5)
        obs.storage_qq_5 .= obs.ztz .- obs.storage_qq_3
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qq_2) # LᵗZᵗZL
        BLAS.trmm!('L', 'L', 'T', 'N', T(1), L, obs.storage_qq_4) # LᵗZᵗZLM⁻¹LᵗZᵗZL
        obs.storage_qq_3 .= obs.storage_qq_2 .- obs.storage_qq_4
        kron!(obs.storage_qq2_2, obs.storage_qq_3, obs.storage_qq_5)
        obs.storage_qq2_1 .+= obs.storage_qq2_2
        D = duplication(q)
        copy!(obs.HLL, transpose(D) * obs.storage_qq2_1 * D)
        obs.HLL ./= (T(-1) * abs2(σ²))
        # compute Hσ²σ²
        obs.Hσ²σ²[1] = n - 2 * tr(obs.LM⁻¹LᵗZᵗZ) + dot(transpose(obs.LM⁻¹LᵗZᵗZ), obs.LM⁻¹LᵗZᵗZ)
        obs.Hσ²σ²[1] /= (-2 * abs2(σ²))
    end
    logl
end

"""
    logl!(m::LmmModel, needgrad = true, needhess = false)

Evaluate the log-likelihood of an LMM model at parameter values `m.β`, `m.L`, 
and `m.σ²`. If `needgrad == true`, then `m.∇β`, `m.∇L`, and `m.σ²` are filled 
with the corresponding gradient. If `needhess == true`, then `obs.Hββ`, `obs.HLL`,
`obs.Hσ²L`, and `obs.Hσ²σ²` are filled with the corresponding Hessian.
"""
function logl!(
    m        :: LmmModel{T},
    needgrad :: Bool = true,
    needhess :: Bool = false
    ) where T <: AbstractFloat
    logl = zero(T)
    if needgrad
        fill!(m.∇β , 0)
        fill!(m.∇L , 0)
        fill!(m.∇σ², 0)        
    end
    @inbounds for i in 1:length(m.data)
        obs = m.data[i]
        logl += logl!(obs, m.β, m.L, m.σ²[1], needgrad, needhess)
        if needgrad
            BLAS.axpy!(T(1), obs.∇β, m.∇β)
            BLAS.axpy!(T(1), obs.∇L, m.∇L)
            m.∇σ²[1] += obs.∇σ²[1]
        end
        if needhess
            BLAS.axpy!(T(1), obs.Hββ, m.Hββ)
            BLAS.axpy!(T(1), obs.HLL, m.HLL)
            BLAS.axpy!(T(1), obs.Hσ²L, m.Hσ²L)
            m.Hσ²σ²[1] += obs.Hσ²σ²[1]
        end    
    end
    # obtain gradient wrt L: m.∇L = m.∇L * L
    if needgrad
        BLAS.trmm!('R', 'L', 'N', 'N', T(1), m.L, m.∇L)
    end
    logl
end