using LMMs, Random, Test, LinearAlgebra, Distributions, BenchmarkTools
using Ipopt, NLopt, MathOptInterface

const MOI = MathOptInterface

@testset "LmmObs" begin
    Random.seed!(257)
    # dimension
    n, p, q = 2000, 5, 3
    # predictors
    X  = [ones(n) randn(n, p - 1)]
    Z  = [ones(n) randn(n, q - 1)]
    # parameter values
    β  = [2.0; -1.0; rand(p - 2)]
    σ² = 1.5
    Σ  = fill(0.1, q, q) + 0.9I # compound symmetry 
    L  = Matrix(cholesky(Symmetric(Σ)).L)
    # generate y
    y  = X * β + Z * rand(MvNormal(Σ)) + sqrt(σ²) * randn(n)
    # form the LmmObs object
    obs = LmmObs(y, X, Z)
    @show logl = logl!(obs, β, L, σ², true, false)
    @show obs.∇β
    @show obs.∇σ²
    @show obs.∇L
    @assert abs(logl - (-3256.1793358058258)) < 1e-4
    @assert norm(obs.∇β - [0.26698108057144054, 41.61418337067327, 
            -34.34664962312689, 36.10898510707527, 27.913948208793144]) < 1e-4
    @assert norm(obs.∇L - 
        [-0.9464482950697888 0.057792444809492895 -0.30244127639188767; 
            0.057792444809492895 -1.00087164917123 0.2845116557144694; 
            -0.30244127639188767 0.2845116557144694 1.170040927259726]) < 1e-4
    @assert abs(obs.∇σ²[1] - (1.6283715138412163)) < 1e-4
    @benchmark logl!($obs, $β, $L, $σ², false, false)
    bm_objgrad = @benchmark logl!($obs, $β, $L, $σ², true, false)
    println(clamp(10 / (median(bm_objgrad).time / 1e3) * 10, 0, 10))
end

Random.seed!(257)
# dimension
m      = 1000 # number of individuals
ns     = rand(1500:2000, m) # numbers of observations per individual
p      = 5 # number of fixed effects, including intercept
q      = 3 # number of random effects, including intercept
obsvec = Vector{LmmObs{Float64}}(undef, m)
# true parameter values
βtrue  = [0.1; 6.5; -3.5; 1.0; 5; zeros(p - 5)]
σ²true = 1.5
σtrue  = sqrt(σ²true)
Σtrue  = Matrix(Diagonal([2.0; 1.2; 1.0; zeros(q - 3)]))
Ltrue  = Matrix(cholesky(Symmetric(Σtrue), Val(true), check = false).L)
# generate data
for i in 1:m
    # first column intercept, remaining entries iid std normal
    X = Matrix{Float64}(undef, ns[i], p)
    X[:, 1] .= 1
    @views Distributions.rand!(Normal(), X[:, 2:p])
    # first column intercept, remaining entries iid std normal
    Z = Matrix{Float64}(undef, ns[i], q)
    Z[:, 1] .= 1
    @views Distributions.rand!(Normal(), Z[:, 2:q])
    # generate y
    y = X * βtrue .+ Z * (Ltrue * randn(q)) .+ σtrue * randn(ns[i])
    # form a LmmObs instance
    obsvec[i] = LmmObs(y, X, Z)
end
# form a LmmModel instance
lmm = LmmModel(obsvec)
(isfile("lmm_data.csv") && filesize("lmm_data.csv") == 245369685) || 
open("lmm_data.csv", "w") do io
    p = size(lmm.data[1].X, 2)
    q = size(lmm.data[1].Z, 2)
    # print header
    print(io, "ID,Y,")
    for j in 1:(p-1)
        print(io, "X" * string(j) * ",")
    end
    for j in 1:(q-1)
        print(io, "Z" * string(j) * (j < q-1 ? "," : "\n"))
    end
    # print data
    for i in eachindex(lmm.data)
        obs = lmm.data[i]
        for j in 1:length(obs.y)
            # id
            print(io, i, ",")
            # Y
            print(io, obs.y[j], ",")
            # X data
            for k in 2:p
                print(io, obs.X[j, k], ",")
            end
            # Z data
            for k in 2:q-1
                print(io, obs.Z[j, k], ",")
            end
            print(io, obs.Z[j, q], "\n")
        end
    end
end
copy!(lmm.β, βtrue)
copy!(lmm.L, Ltrue)
lmm.σ²[1] = σ²true
@show obj = logl!(lmm, true, true)
@show lmm.∇β
@show lmm.∇σ²
@show lmm.∇L
@show lmm.Hββ
@show lmm.HLL
@show lmm.Hσ²L
@show lmm.Hσ²σ²

@testset "LmmModel" begin
    @assert abs(obj - (-2.840068438369969e6)) < 1e-4
    @assert norm(lmm.∇β - [41.0659167074073, 445.75120353972426, 
            157.0133992249258, -335.09977360733626, -895.6257448385899]) < 1e-4
    @assert norm(lmm.∇L - [-3.3982575935824837 31.32103842086001 26.73645089732865; 
            40.43528672997116 61.86377650461202 -75.37427770754684; 
            37.811051468724486 -82.56838431216435 -56.45992542754974]) < 1e-4
    @assert abs(lmm.∇σ²[1] - (-489.5361730382465)) < 1e-4
    bm_model = @benchmark logl!($lmm, true, false)
    clamp(10 / (median(bm_model).time / 1e6) * 10, 0, 10)
    clamp(10 - median(bm_model).memory / 100, 0, 10)
end

@testset "init" begin
    init_ls!(lmm)
    @show logl!(lmm)
    @show lmm.β
    @show lmm.σ²
    @show lmm.L
    (logl!(lmm) >  -3.3627e6) * 10
    bm_init = @benchmark init_ls!($lmm)
    clamp(1 / (median(bm_init).time / 1e6) * 10, 0, 10)
end

@testset "fit!" begin
    # initialize from least squares
    init_ls!(lmm)
    println("objective value at starting point: ", logl!(lmm)); println()
    # NLopt (LD_MMA) obj. val = -2.8400587866501966e6
    NLopt_solver = NLopt.Optimizer()
    MOI.set(NLopt_solver, MOI.RawOptimizerAttribute("algorithm"), :LD_MMA)
    @time fit!(lmm, NLopt_solver)

    println("objective value at solution: $(logl!(lmm)))")
    println("solution values:")
    @show lmm.β
    @show lmm.σ²
    @show lmm.L * transpose(lmm.L)
    println("gradient @ solution:")
    @show lmm.∇β
    @show lmm.∇σ²
    @show lmm.∇L
    @show norm([lmm.∇β; vec(LowerTriangular(lmm.∇L)); lmm.∇σ²])
    # objective at solution should be close enough to the optimal
    @assert logl!(lmm) > -2.840059e6
    # gradient at solution should be small enough
    @assert norm([lmm.∇β; vec(LowerTriangular(lmm.∇L)); lmm.∇σ²]) < 0.1
    solver = Ipopt.Optimizer()
    MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), 0)
    bm_mma = @benchmark fit!($lmm, $(solver)) setup=(init_ls!(lmm))
    # NLopt_solver = NLopt.Optimizer()
    # MOI.set(NLopt_solver, MOI.RawOptimizerAttribute("algorithm"), :LD_MMA)
    # bm_mma = @benchmark fit!($lmm, $(NLopt_solver)) setup = (init_ls!(lmm))
    clamp(1 / (median(bm_mma).time / 1e9) * 10, 0, 10)

    # vector of solvers to compare
    solvers = ["NLopt (LN_COBYLA, gradient free)", "NLopt (LD_MMA, gradient-based)", 
        "Ipopt (L-BFGS)", "Ipopt (use FIM)"]

    function setup_solver(s::String)
        if s == "NLopt (LN_COBYLA, gradient free)"
            solver = NLopt.Optimizer()
            MOI.set(solver, MOI.RawOptimizerAttribute("algorithm"), :LN_COBYLA)
        elseif s == "NLopt (LD_MMA, gradient-based)"
            solver = NLopt.Optimizer()
            MOI.set(solver, MOI.RawOptimizerAttribute("algorithm"), :LD_MMA)
        elseif s == "Ipopt (L-BFGS)"
            solver = Ipopt.Optimizer()
            MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), 0)
            MOI.set(solver, MOI.RawOptimizerAttribute("hessian_approximation"), "limited-memory")
            MOI.set(solver, MOI.RawOptimizerAttribute("tol"), 1e-6)
        elseif s == "Ipopt (use FIM)"
            # Ipopt (use Hessian) obj val = -2.8400587866468e6
            solver = Ipopt.Optimizer()
            MOI.set(solver, MOI.RawOptimizerAttribute("print_level"), 0)        
        else
            error("unrecognized solver $s")
        end
        solver
    end
    # # containers for results
    # runtime = zeros(length(solvers))
    # objvals = zeros(length(solvers))
    # gradnrm = zeros(length(solvers))

    # for i in 1:length(solvers)
    #     solver = setup_solver(solvers[i])
    #     bm = @benchmark fit!($lmm, $solver) setup = (init_ls!(lmm))
    #     runtime[i] = median(bm).time / 1e9
    #     objvals[i] = logl!(lmm, true, false)
    #     gradnrm[i] = norm([lmm.∇β; vec(LowerTriangular(lmm.∇L)); lmm.∇σ²])
    # end

    # # display results
    # pretty_table(
    #     hcat(solvers, runtime, objvals, gradnrm),
    #     header = ["Solver", "Runtime", "Log-Like", "Gradiant Norm"],
    #     formatters = (ft_printf("%5.2f", 2), ft_printf("%8.8f", 3:4))
    #     )
end