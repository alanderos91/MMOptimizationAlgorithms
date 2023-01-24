#
#   Load environment
#
import Pkg; Pkg.activate("./scripts"); Pkg.status()

using CSV, DataFrames
using LinearAlgebra, Statistics
using MMOptimizationAlgorithms

const MMOA = MMOptimizationAlgorithms
const OPTIONS = set_options(;
    maxiter=10^4,
    maxrhov=100,
    gtol=1e-4,
    dtol=1e-6,
    rtol=1e-8,
    rhof=geometric_progression(2.0),
    nesterov=10,
    rho_max=Inf,
)

sample_cov(X) = 1 / size(X, 1) * X'X

normF(A) = norm(A, 2) / sqrt(size(A, 2))

function shrinkage_cov(X)
    n = size(X, 1)
    S = sample_cov(X)
    m = tr(S)
    d2 = normF(S - m*I)^2
    b2bar = 1/(n^2) * sum(normF(x*x' - S)^2 for x in eachrow(X))
    b2 = min(b2bar, d2)
    a2 = d2 - b2
    return b2/d2 * m * I + a2/d2 * S
end

function annualize(r)
    n = length(r)
    v = one(eltype(r))
    for i in eachindex(r)
        v *= (1+r[i])
    end
    v = v^(1/n) - 1
end

n_years = 4
weeks_per_year = 52
days_per_week = 5
days_per_year = weeks_per_year * days_per_week
window = 2

df = CSV.read(joinpath("data", "S&P500-asset_returns.csv"), DataFrame; header=false)
n_observations, n_assets = size(df)

# Sample data to extract enough observations for rolling window.
i_start = n_observations - (n_years+1)*weeks_per_year + 1
i_stop = n_observations
data = Matrix(df[i_start:i_stop, :])

# Compute average within rolling windows.
X = [zeros(weeks_per_year, n_assets) for _ in 1:n_years]
for k in eachindex(X)
    for j in 1:window
        start = weeks_per_year * (k + (j-1) - 1) + 1
        stop = weeks_per_year * (k + (j-1))
        axpy!(1/window, view(data, start:stop, :), X[k])
    end
end

# Annualize the returns.
r = [map(r -> annualize(r), eachcol(Xi)) for Xi in X]
R = reshape(vcat(r...), n_assets, n_years)

# Estimate covariance matrices for each year.
C = [shrinkage_cov(Xi) for Xi in X]
@assert all(isposdef, C)

expected_annual_return = 10.0 # 5% growth per year
xi_init = 1.0
xi_term = (1 + expected_annual_return/100)^n_years

println("""
[ S&P 500 Portfolio Optimization ]

Data from:

    Bruni R, Cesarone F, Scozzani A, Tardella F (2016)
    \"Real-world datasets for portfolio selection and solutions of some
      stochastic dominance portfolio models\".
    Data in Brief. 8: 858-862.

    Number of assets:  $(n_assets)
    Number of periods: $(n_years) (years)
    
    Expected annual return: $(expected_annual_return) %
    Initial wealth: $(xi_init)
    Target wealth:  $(xi_term)
""")

a = 2/3
result = @time MMOA.portfolio_optimization(MML(), C, R, xi_init, xi_term;
    options=OPTIONS,
    callback=VerboseCallback(10),
    tau=(1e3, 1e3),
    alpha=(a/2, a/2, 1-a),
)

W = result.projected

risk = sum(dot(w, C[j], w) for (j, w) in enumerate(eachcol(W)))
realized_wealth = dot(1 .+ R[:,end], W[:,end])
n_short = count(<(0), W)
n_long = count(>(0), W)
n_zeros = count(isequal(0), W)

println("""
    Total short positions: $(n_short)
    Total long positiions: $(n_long)
    Sparsity:              $(round(n_zeros / (prod(size(W))) * 100, digits=2)) %

    Risk:                  $(risk)
    Initial wealth:        $(sum(W[:,1]))
    Final wealth:          $(realized_wealth)
""")

@show extrema(W)
@show norm(result.residual)

CSV.write("/home/alanderos/Desktop/assets.csv", DataFrame(W, :auto), header=false)
CSV.write("/home/alanderos/Desktop/diffs.csv", DataFrame(result.differences, :auto), header=false)