"""
    kron_axpy!(A, X, Y)

Overwrite `Y` with `A ⊗ X + Y`. Same as `Y += kron(A, X)`, but more memory efficient.
"""
@inline function kron_axpy!(
    A :: AbstractVecOrMat{T},
    X :: AbstractVecOrMat{T},
    Y :: AbstractVecOrMat{T}
    ) where T <: Real
    m, n = size(A, 1), size(A, 2)
    p, q = size(X, 1), size(X, 2)
    @assert size(Y, 1) == m * p
    @assert size(Y, 2) == n * q
    yidx = 0
    @inbounds for j in 1:n, l in 1:q, i in 1:m
        aij = A[i, j]
        for k in 1:p
            Y[yidx += 1] += aij * X[k, l]
        end
    end
    Y
end

"""
    vech(A::AbstractVecOrMat)

Return lower triangular part of `A`.
"""
vech(A::AbstractVecOrMat) = [A[i, j] for i in 1:size(A, 1), j in 1:size(A, 2) if i ≥ j]

function duplication(n::Int)
    D = zeros(Int, abs2(n), ◺(n))
    vechidx = 1
    for j in 1:n
        for i in j:n
            D[(j - 1) * n + i, vechidx] = 1
            D[(i - 1) * n + j, vechidx] = 1
            vechidx += 1
        end
    end
    D
end

function commutation(m::Int, n::Int)
    K = zeros(Int, m * n, m * n)
    colK = 1
    @inbounds for j in 1:n, i in 1:m
        rowK          = n * (i - 1) + j
        K[rowK, colK] = 1
        colK         += 1
    end
    K
end

commutation(m::Int) = commutation(m, m)