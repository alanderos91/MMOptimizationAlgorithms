#
#   Load environment
#
import Pkg; Pkg.activate("./scripts"); Pkg.status()

using DataStructures, MMOptimizationAlgorithms, Random
using CSV, DataFrames, LinearAlgebra, Statistics
using CairoMakie, LaTeXStrings

const MMOA = MMOptimizationAlgorithms

#
#   Algorithm Settings
#
const ALGORITHM = SD();
const OPTIONS = set_options(ALGORITHM;
    maxiter=10^4,
    maxrhov=100,
    gtol=1e-4,
    dtol=1e-6,
    rtol=0.0,
    rhof=geometric_progression(1.2),
    nesterov=0,
    rho_max=Inf,
)
const PATHF = OrderedDict(
        "warm start" => naive_update,
        "linear" => linear_update,
        "exponential" => exponential_update,
    )

#
#   HistoryCallback Settings
#
const CALLBACK = HistoryCallback()
const HISTORY = CALLBACK.data
MMOA.add_field!(CALLBACK, :rho, :iters, :objective, :gradient, :distance)

#
#   Regression Simulation Settings
#
const N = 10^3
const P = 2*N
const K = round(Int, 0.05*P)
const EXAMPLE = MMOA.simulate_sparse_regression(N, P, K; rng=Xoshiro(1111))

#
#   Portfolio Example
#
function create_portfolio_problem(n_years, expected_annual_return::Real=10.0)
    # Helper functions
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

    xi_init = 1.0
    xi_term = (1 + expected_annual_return/100)^n_years

    return C, R, xi_init, xi_term
end

const PORTFOLIO_EXAMPLE = create_portfolio_problem(4, 10.0)

#
#   Fused Lasso Settings
#
const R1 = 1e0
const R2 = 1e0

#
#   Plotting
#

struct IntegerTicks end

Makie.get_tickvalues(::IntegerTicks, vmin, vmax) = ceil(Int, vmin) : floor(Int, vmax)

function make_axis(f, title, xlabel, ylabel)
    Axis(f,
        title=title,
        xlabel=xlabel,
        xscale=log10,
        xticks=LogTicks(IntegerTicks()),
        xminorticksvisible=true,
        xminorticks=IntervalsBetween(9),
        ylabel=ylabel,
        yscale=log10,
        yticks=LogTicks(IntegerTicks()),
        yminorticksvisible=true,
        yminorticks=IntervalsBetween(9),
    )
end

function plot_history_data(example, r, h, field, xlabel, ylabel, bias=0.0)
    size_inches = (8, 6)
    size_pt = 72 .* size_inches
    fig = Figure(resolution=size_pt, fontsize=16)
    ax = make_axis(fig[1,1], example, xlabel, ylabel)
    # axzoom = make_axis(fig[2,1], example, xlabel, ylabel)
    for (label, data) in h
        x = eachindex(data[field])
        y = iszero(bias) ? data[field] : bias .+ data[field]

        xmax = maximum(x)
        ymin = minimum(y)

        # Full Figure
        # viewxmax = 1e1 .^ ceil(log10(xmax))
        # viewymin = 1e1 .^ floor(log10(ymin))
        # lines!(ax, x, y, label=label, linewidth=5,)
        # xlims!(ax, high=viewxmax)
        # ylims!(ax, low=viewymin)

        # Zoomed View
        viewxmax = 1e1 .^ ceil(log10(xmax))
        viewxmin = 1e3 # max(1.0, 1e1 .^ floor(log10(xmax) - 1))
        idx = findfirst(>(viewxmin), x)
        viewymin, viewymax = extrema(y[idx:end])
        viewymin = 1e1 .^ floor(log10(viewymin))
        viewymax = 1e1 .^ ceil(log10(viewymax))
        lines!(ax, x, y, label=label, linewidth=5,)
        xlims!(ax, low=viewxmin, high=viewxmax)
        ylims!(ax, low=viewymin, high=viewymax)
    end
    # fig[3,1] = Legend(fig, ax, "Extrapolation", framevisible=false, orientation=:horizontal)
    fig[1,2] = Legend(fig, ax, "Extrapolation", framevisible=false, orientation=:vertical)
    resize_to_layout!(fig)
    return fig
end

function plot_distances(example, r, h)
    plot_history_data(example, r, h, :distance, latexstring("Iteration", " ", L"n"), "Distance", 1e-6)
end

function plot_gradients(example, r, h)
    plot_history_data(example, r, h, :gradient, latexstring("Iteration", " ", L"n"), "Gradient")
end

function plot_objectives(example, r, h)
    plot_history_data(example, r, h, :objective, latexstring("Iteration", " ", L"n"), "Objective")
end

#
#   Examples
#

function run_fused_lasso()
    global OPTIONS, ALGORITHM, EXAMPLE, CALLBACK, HISTORY, PATHF
    global R1, R2
    y, X, _ = EXAMPLE

    r, h = OrderedDict{String,Any}(), OrderedDict{String,Any}()
    for (label, pathf) in PATHF
        @info "Running Fused Lasso / $(label)..."
        result = @time fused_lasso(ALGORITHM, y, X, R1, R2;
            options=OPTIONS,
            callback=CALLBACK,
            pathf=pathf,
        )
        r[label] = result
        h[label] = deepcopy(HISTORY)
        foreach(keyval -> empty!(last(keyval)), HISTORY)
    end
    return r, h
end

function run_sparse_regression()
    global OPTIONS, ALGORITHM, EXAMPLE, CALLBACK, HISTORY, PATHF
    global K
    y, X, _ = EXAMPLE

    r, h = OrderedDict{String,Any}(), OrderedDict{String,Any}()
    for (label, pathf) in PATHF
        @info "Running Sparse Regression / $(label)..."
        result = @time sparse_regression(ALGORITHM, y, X, K;
            options=OPTIONS,
            callback=CALLBACK,
            pathf=pathf,
        )
        r[label] = result
        h[label] = deepcopy(HISTORY)
        foreach(keyval -> empty!(last(keyval)), HISTORY)
    end
    return r, h
end

function run_portfolio_optimization()
    global OPTIONS, PORTFOLIO_EXAMPLE, CALLBACK, HISTORY, PATHF
    C, R, xi_init, xi_term = PORTFOLIO_EXAMPLE
    a = 2/3

    r, h = OrderedDict{String,Any}(), OrderedDict{String,Any}()
    for (label, pathf) in PATHF
        @info "Running Portfolio Optimization / $(label)..."
        result = @time MMOA.portfolio_optimization(MMAL(), C, R, xi_init, xi_term;
            options=OPTIONS,
            callback=CALLBACK,
            tau=(1e3, 1e3),
            alpha=(a/2, a/2, 1-a),
            pathf=pathf,
        )
        r[label] = result
        h[label] = deepcopy(HISTORY)
        foreach(keyval -> empty!(last(keyval)), HISTORY)
    end
    return r, h
end

#
#   Main Program
#

function main(outdir, examples)
    !ispath(outdir) && mkpath(outdir)
    ext = "pdf"
    for (example, runf) in examples
        result, history = runf()
        
        @info "Plotting distances..."
        fig1 = @time plot_distances(example, result, history)
        save(
            joinpath(outdir, "$(example)-distances.$(ext)"), fig1, pt_per_unit=1,
        )

        @info "Ploting gradients..."
        fig2 = @time plot_gradients(example, result, history)
        save(
            joinpath(outdir, "$(example)-gradients.$(ext)"), fig2, pt_per_unit=1,
        )

        @info "Ploting objectives..."
        fig3 = @time plot_objectives(example, result, history)
        save(
            joinpath(outdir, "$(example)-objectives.$(ext)"), fig3, pt_per_unit=1,
        )
    end
    return 0
end

outdir = nothing
examples = []

for (k, arg) in enumerate(ARGS)
    if k == 1
        global outdir = arg
    elseif arg == "SR"
        push!(examples, ("Sparse Regression", run_sparse_regression))
    elseif arg == "FL"
        push!(examples, ("Fused Lasso", run_fused_lasso))
    elseif arg == "PO"
        push!(examples, ("Portfolio Optimization", run_portfolio_optimization))
    else
        error("Unknown example code $(arg)")
    end
end

main(outdir, examples)
