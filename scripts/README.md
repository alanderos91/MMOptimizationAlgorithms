# Instructions for Scripts

These instructions assume that the top-level directory is the package directory `MMOptimizationAlgorithms`, with the following package structure:

```text
MMOptimizationAlgorithms
├── data
├── results
├── scripts
├── src
│   ├── problems
│   └── projections
└── test
```

Each script automatically loads the correct environment catalogued in `scripts/Project.toml` and `scripts/Manifest.toml`.

Output is saved to the `results` directory.

Scripts assume Julia 1.7.3 is used. Newer versions are likely to also work. If you use `juliaup` to manage multiple versions, replace `julia` with `julia +1.7.3` in the Bash commands below.

## Examples

### Portfolio Optimization

```bash
julia -t 2 scripts/portfolio_optimization.jl
```

The command above runs three algorithms:

- a *Cholesky-based* version that operates on the full Hessian,
- an *accelerated* version that exploits Hessian structure, and
- a *block* version that updates 2 parameter blocks in parallel.

Note that the parallel algorithm requires a machine with at least 2 physical cores to be effective.

Results are saved to CSV files.

### Path Following

```bash
julia -t 8 scripts/path_following.jl ./results FL PO
```

The command above runs path following on the **fused lasso (FL)** and **portfolio optimization (PO)** problems using 8 Julia threads. The command can be executed without the `-t 8` option, but it is documented here to reflect the computing environment used in our results.

Results are saved as figures (PDF).

### Styblinski-Tang

```bash
julia -t 8 scripts/styblinski-tang.jl  
```

The command above runs the **Styblinski-Tang** example comparing an adaptive trust region algorithm against Newton's method with step-halving. The command can be executed without the `-t 8` option, but it is documented here to reflect the computing environment used in our results.

The random seed is used to select a starting point as described in the text. Both methods should converge to the global minimum. The trust region method should converge in fewer iterations in a monotonic fashion whereas Newton's method with step-halving has a dramatic jump far away from the global minimum.

Results are printed to the console.

### Poisson Regression

```bash
julia -t 8 scripts/poisson_regression.jl  
```

The command above runs the **Poisson regression** example which compares log-likelihood models under the canonical link and alternative links with change points at $p=1$, $p=2$, and $p=3$. The command can be executed without the `-t 8` option, but it is documented here to reflect the computing environment used in our results.

Data is simulated so that the Poisson assumption is reasonable, as described in the manuscript. This means the random seed was selected to avoid dispersion.

Results are saved as figures (PDF).

## Computing Environment Information

Output from `versioninfo()`:

```text
Julia Version 1.7.3
Commit 742b9abb4d (2022-05-06 12:58 UTC)
Platform Info:
  OS: Linux (x86_64-pc-linux-gnu)
  CPU: Intel(R) Core(TM) i9-10900KF CPU @ 3.70GHz
  WORD_SIZE: 64
  LIBM: libopenlibm
  LLVM: libLLVM-12.0.1 (ORCJIT, skylake)
```

Output from `]status` in `MMOptimizationAlgorithms` and `MMOptimizationAlgorithms/scripts` environments:

```text
     Project MMOptimizationAlgorithms v0.1.0
      Status `~/Projects/MMAlgorithmsPNAS/Code/Project.toml`
  [b4f34e82] Distances v0.10.7
  [f6369f11] ForwardDiff v0.10.30
  [86223c79] Graphs v1.7.2
  [c8e1da08] IterTools v1.4.0
  [ba0b0d4f] Krylov v0.9.0
  [f517fe37] Polyester v0.6.15
  [21216c6a] Preferences v1.3.0
  [860ef19b] StableRNGs v1.0.0
  [2913bbd2] StatsBase v0.33.17
  [3a884ed6] UnPack v1.0.2
  [37e2e46d] LinearAlgebra
  [de0858da] Printf
  [9a3f8284] Random
  [2f01184e] SparseArrays
  [10745b16] Statistics
```

```text
      Status `~/Projects/MMAlgorithmsPNAS/Code/scripts/Project.toml`
  [336ed68f] CSV v0.10.9 ⚲
  [13f3f980] CairoMakie v0.10.0 ⚲
  [a93c6f00] DataFrames v1.4.4 ⚲
  [864edb3b] DataStructures v0.18.13 ⚲
  [31c24e10] Distributions v0.25.79 ⚲
  [b964fa9f] LaTeXStrings v1.3.0 ⚲
  [aedcfd60] MMOptimizationAlgorithms v0.1.0 `~/Projects/MMAlgorithmsPNAS/Code` ⚲
  [f2b01f46] Roots v2.0.8 ⚲
  [276daf66] SpecialFunctions v2.1.7 ⚲
  [2913bbd2] StatsBase v0.33.21 ⚲
  [bd369af6] Tables v1.10.0 ⚲
```
