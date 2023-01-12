#
#   Load environment
#
import Pkg; Pkg.activate("./scripts"); Pkg.status()

using DataStructures, MMOptimizationAlgorithms, Random
using CairoMakie

const MMOA = MMOptimizationAlgorithms

#
#   Algorithm Settings
#
const ALGORITHM = SD();
const OPTIONS = set_options(ALGORITHM;
    maxiter=10^3,
    maxrhov=100,
    gtol=1e-5,
    dtol=1e-5,
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
#   Simulation Settings
#
const N = 10^3
const P = 2*N
const K = round(Int, 0.05*P)
const EXAMPLE = MMOA.simulate_sparse_regression(N, P, K; rng=Xoshiro(1111))

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
    fig = Figure(resolution=size_pt, fontsize=12)
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
    plot_history_data(example, r, h, :distance, "iteration", "distance", 1e-6)
end

function plot_gradients(example, r, h)
    plot_history_data(example, r, h, :gradient, "iteration", "gradient")
end

function plot_objectives(example, r, h)
    plot_history_data(example, r, h, :objective, "iteration", "objective")
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
    else
        error("Unknown example code $(arg)")
    end
end

main(outdir, examples)
