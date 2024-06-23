using LMM, Random, Test, LinearAlgebra, Distributions

@testset "LMM.jl" begin
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
    @show logl = logl!(obs, β, L, σ², true)
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
end
