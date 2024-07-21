module MPSCFD

using ITensors
using LinearAlgebra
using SparseArrays
using FFTW
using Plots
using MAT

using KrylovKit: linsolve

include("./parameters.jl")
include("./operator_sum.jl")
include("./mapping.jl")
include("./trial_solution.jl")
include("./setup.jl")
include("./finite_differences.jl")
include("./ladder_operator.jl")
include("./mpo.jl")
include("./cached_contractions.jl")
include("./rhs.jl")
include("./variational.jl")
include("./naviar_stokes.jl")
include("./load_references.jl")
include("./visualization.jl")

end # module MPSCFD
