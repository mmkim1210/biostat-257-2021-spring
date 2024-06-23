"""
    fit!(m::LmmModel, solver=Ipopt.Optimizer())

Fit an `LmmModel` object by MLE using a nonlinear programming solver. Start point 
should be provided in `m.β`, `m.σ²`, `m.L`.
"""
function fit!(
        m :: LmmModel{T},
        solver = Ipopt.Optimizer()
    ) where T <: AbstractFloat
    p    = size(m.data[1].X, 2)
    q    = size(m.data[1].Z, 2)
    npar = p + ((q * (q + 1)) >> 1) + 1
    # prep the MOI
    MOI.empty!(solver)
    # set lower bounds and upper bounds of parameters
    # q diagonal entries of Cholesky factor L should be >= 0
    # σ² should be >= 0
    lb   = fill(0.0, q + 1)
    ub   = fill(Inf, q + 1)
    NLPBlock = MOI.NLPBlockData(MOI.NLPBoundsPair.(lb, ub), m, true)
    MOI.set(solver, MOI.NLPBlock(), NLPBlock)
    # start point
    params = MOI.add_variables(solver, npar)    
    par0   = Vector{T}(undef, npar)
    modelpar_to_optimpar!(par0, m)    
    for i in 1:npar
        MOI.set(solver, MOI.VariablePrimalStart(), params[i], par0[i])
    end
    MOI.set(solver, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    # optimize
    MOI.optimize!(solver)
    optstat = MOI.get(solver, MOI.TerminationStatus())
    optstat in (MOI.LOCALLY_SOLVED, MOI.ALMOST_LOCALLY_SOLVED) || 
        @warn("Optimization unsuccesful; got $optstat")
    # update parameters and refresh gradient
    xsol = [MOI.get(solver, MOI.VariablePrimal(), MOI.VariableIndex(i)) for i in 1:npar]
    optimpar_to_modelpar!(m, xsol)
    logl!(m, true)
    m
end

"""
    ◺(n::Integer)

Triangular number `n * (n + 1) / 2`.
"""
@inline ◺(n::Integer) = (n * (n + 1)) >> 1

"""
    modelpar_to_optimpar!(par, m)

Translate model parameters in `m` to optimization variables in `par`.
"""
function modelpar_to_optimpar!(
        par :: Vector,
        m   :: LmmModel
    )
    p = size(m.data[1].X, 2)
    q = size(m.data[1].Z, 2)
    # β
    copyto!(par, m.β)
    # L
    offset = p + 1
    @inbounds for j in 1:q, i in j:q
        par[offset] = m.L[i, j]
        offset += 1
    end
    # σ²
    par[end] = m.σ²[1]
    par
end

"""
    optimpar_to_modelpar!(m, par)

Translate optimization variables in `par` to the model parameters in `m`.
"""
function optimpar_to_modelpar!(
        m   :: LmmModel, 
        par :: Vector
    )
    p = size(m.data[1].X, 2)
    q = size(m.data[1].Z, 2)
    # β
    copyto!(m.β, 1, par, 1, p)
    # L
    fill!(m.L, 0)
    offset = p + 1
    @inbounds for j in 1:q, i in j:q
        m.L[i, j] = par[offset]
        offset   += 1
    end
    # σ²
    m.σ²[1] = par[end]    
    m
end

function MOI.initialize(
        m                  :: LmmModel, 
        requested_features :: Vector{Symbol}
    )
    for feat in requested_features
        if !(feat in MOI.features_available(m))
            error("Unsupported feature $feat")
        end
    end
end

MOI.features_available(m::LmmModel) = [:Grad, :Hess, :Jac]

function MOI.eval_objective(
        m   :: LmmModel, 
        par :: Vector
    )
    optimpar_to_modelpar!(m, par)
    logl!(m, false) # don't need gradient here
end

function MOI.eval_objective_gradient(
        m    :: LmmModel, 
        grad :: Vector, 
        par  :: Vector
    )
    p = size(m.data[1].X, 2)
    q = size(m.data[1].Z, 2)
    optimpar_to_modelpar!(m, par) 
    obj = logl!(m, true)
    # gradient wrt β
    copyto!(grad, m.∇β)
    # gradient wrt L
    offset = p + 1
    @inbounds for j in 1:q, i in j:q
        grad[offset] = m.∇L[i, j]
        offset += 1
    end
    # gradient with respect to σ²
    grad[end] = m.∇σ²[1]
    # return objective
    obj
end

function MOI.eval_constraint(m::LmmModel, g, par)
    p = size(m.data[1].X, 2)
    q = size(m.data[1].Z, 2)
    gidx   = 1
    offset = p + 1
    @inbounds for j in 1:q, i in j:q
        if i == j
            g[gidx] = par[offset]
            gidx   += 1
        end
        offset += 1
    end
    g[end] = par[end]
    g
end

function MOI.jacobian_structure(m::LmmModel)
    p    = size(m.data[1].X, 2)
    q    = size(m.data[1].Z, 2)
    row  = collect(1:(q + 1))
    col  = Int[]
    offset = p + 1
    for j in 1:q, i in j:q
        (i == j) && push!(col, offset)
        offset += 1
    end
    push!(col, offset)
    [(row[i], col[i]) for i in 1:length(row)]
end

MOI.eval_constraint_jacobian(m::LmmModel, J, par) = fill!(J, 1)

function MOI.hessian_lagrangian_structure(m::LmmModel)
    p    = size(m.data[1].X, 2)
    q    = size(m.data[1].Z, 2)    
    q◺   = ◺(q)
    # we work on the upper triangular part of the Hessian
    arr1 = Vector{Int}(undef, ◺(p) + ◺(q◺) + q◺ + 1)
    arr2 = Vector{Int}(undef, ◺(p) + ◺(q◺) + q◺ + 1)
    # Hββ block
    idx  = 1    
    for j in 1:p, i in 1:j
        arr1[idx] = i
        arr2[idx] = j
        idx      += 1
    end
    # HLL block
    for j in 1:q◺, i in 1:j
        arr1[idx] = p + i
        arr2[idx] = p + j
        idx      += 1
    end
    # HLσ² block
    for i in (p + 1):(p + q◺)
        arr1[idx] = i
        arr2[idx] = p + q◺ + 1
        idx      += 1
    end
    # Hσ²σ² block
    arr1[idx] = p + q◺ + 1
    arr2[idx] = p + q◺ + 1
    [(arr1[i], arr2[i]) for i in 1:length(arr1)]
end

function MOI.eval_hessian_lagrangian(
        m   :: LmmModel, 
        H   :: AbstractVector{T},
        par :: AbstractVector{T}, 
        σ   :: T, 
        μ   :: AbstractVector{T}
    ) where {T}    
    p  = size(m.data[1].X, 2)
    q  = size(m.data[1].Z, 2)    
    q◺ = ◺(q)
    optimpar_to_modelpar!(m, par)
    logl!(m, true, true)
    # Hββ block
    idx = 1    
    @inbounds for j in 1:p, i in 1:j
        H[idx] = m.Hββ[i, j]
        idx   += 1
    end
    # HLL block
    @inbounds for j in 1:q◺, i in 1:j
        H[idx] = m.HLL[i, j]
        idx   += 1
    end
    # HLσ² block
    @inbounds for j in 1:q, i in j:q
        H[idx] = m.Hσ²L[i, j]
        idx   += 1
    end
    # Hσ²σ² block
    H[idx] = m.Hσ²σ²[1, 1]
    lmul!(σ, H)
end