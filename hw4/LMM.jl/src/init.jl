
"""
    init_ls!(m::LmmModel)

Initialize parameters of a `LmmModel` object from the least squares estimate. 
`m.β`, `m.L`, and `m.σ²` are overwritten with the least squares estimates.
"""
function init_ls!(m::LmmModel{T}) where T <: AbstractFloat
    p, q = size(m.data[1].X, 2), size(m.data[1].Z, 2)
    # fill m.β, m.σ², m.L by LS estimates
    # initialize arrays
    n = T(0)
    m.xtx .= T(0)
    m.xty .= T(0)
    m.σ²[1] = 0
    @inbounds for i in 1:length(m.data)
        obs = m.data[i]
        n += size(obs.X, 1)
        m.xtx .+= obs.xtx
        m.xty .+= obs.xty
        m.σ²[1] += obs.yty
    end
    # compute m.β
    copy!(m.storage_pp, m.xtx)
    copy!(m.β, m.xty)
    LAPACK.potrf!('U', m.storage_pp)
    LAPACK.potrs!('U', m.storage_pp, m.β)
    # compute m.σ²
    m.σ²[1] += dot(m.β, BLAS.gemv!('N', T(1), m.xtx, m.β, T(-2), m.xty))
    m.σ²[1] /= n
    # compute m.L
    m.ztz2 .= T(0)
    m.storage_qq .= T(0)
    @inbounds for i in 1:length(m.data)
        obs = m.data[i]
        kron!(m.storage_qq2, obs.ztz, obs.ztz)
        BLAS.axpy!(T(1), m.storage_qq2, m.ztz2)
        copy!(m.ztr, obs.zty)
        BLAS.gemv!('N', T(-1), obs.ztx, m.β, T(1), m.ztr)
        BLAS.ger!(T(1), m.ztr, m.ztr, m.storage_qq)
    end
    m.ztr2 .= vec(m.storage_qq)
    LAPACK.potrf!('U', m.ztz2)
    LAPACK.potrs!('U', m.ztz2, m.ztr2)
    m.L .= reshape(m.ztr2, q, q)
    LAPACK.potrf!('L', m.L)
end