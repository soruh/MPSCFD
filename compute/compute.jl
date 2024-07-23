_start_time = time()

using Printf
# using Distributed
# addprocs(4)

function format_time(t)::String
    (minutes, seconds) = fldmod(t, 60)
    (hours, minutes) = fldmod(minutes, 60)
    @sprintf("%02d:%02d:%02d", hours, minutes, seconds)
end

time_elapsed()::String = format_time(time() - _start_time)

let
    println("$(time_elapsed()) | preparing environment...")
    using Pkg
    # Pkg.respect_sysimage_versions(false)
    # Pkg.activate(; temp=true)
    Pkg.develop(url="../MPSCFD.jl")
end

let
    println("$(time_elapsed()) | loading packages...")

    using Statistics
    using MKL
    using LinearAlgebra

    BLAS.set_num_threads(8)

    using HDF5
    using ITensors

    using MPSCFD
    # using AMDGPU
end

println("$(time_elapsed()) | BLAS config: $(BLAS.get_config())")

if length(ARGS) < 1
    error("missing config file parameter")
end
config_file = ARGS[1]
println("$(time_elapsed()) | loading config $config_file...")
include(config_file)

max_it(keys) = maximum(map(x -> parse(Int64, x[1]), filter(x -> !isnothing(x), map(x -> match(r"it=(\d+),x", x), keys))))

it_start = try
    h5open(filename, "r") do file
        max_it(keys(file))
    end
catch
    0
end

# do_projections(values::Vector{Tuple{MPSCFD.CachedPartialContraction,ITensor}})::Vector{ITensor} = pmap(((P, T),) -> noprime(MPSCFD.product(P, T)), values)

function do_load_setup()::MPSCFD.SolverState
    println("$(time_elapsed()) | loading setup...")
    ux, uy = MPSCFD.load_reference("../reference/data/IncompNs2D/NsIncomp2DFieldOutL10C118Re200000mt0.mat")

    dx = jet_params.L_box / size(ux)[1]
    # r = range(0.0, jet_params.L_box; length=N + 1)[1:end-1]

    ux, uy = MPSCFD.do_mapping.((ux, uy))
    uy = replaceinds!(uy, inds(uy), inds(ux))

    ux = MPS(ux, inds(ux); maxdim=χ)
    uy = MPS(uy, inds(uy); maxdim=χ)

    ops = MPSCFD.Operators(ux, uy, dx, params)
    MPSCFD.SolverState(; ops, ux, uy)
end

function do_compute_setup()::MPSCFD.SolverState
    println("$(time_elapsed()) | computing setup for M=$M, χ=$χ...")
    MPSCFD.setup_jet(M, jet_params, params)
end

solver_state = if it_start == 0 || precompile

    if precompile
        @time do_load_setup()
        @time do_compute_setup()
    end

    solver_state = load_setup ? do_load_setup() : do_compute_setup()

    M = length(solver_state.ux)
    N = 2^M

    h5open(filename, "w") do file
        write(file, "it=0,x", solver_state.ux)
        write(file, "it=0,y", solver_state.uy)
    end

    solver_state
else
    println("$(time_elapsed()) | starting in $filename at iteration $it_start...")

    h5open(filename, "r") do file
        ux = read(file, "it=$it_start,x", MPS)
        uy = read(file, "it=$it_start,y", MPS)

        M = length(ux)
        N = 2^M

        dx = jet_params.L_box / N
        ops = MPSCFD.Operators(ux, uy, dx, params)
        MPSCFD.SolverState(; ops, ux, uy)
    end
end

#roc64(T::ITensor)::ITensor = ITensor(AMDGPU.ROCArray{Float64}(Array(T, inds(T)...)), inds(T))
#move = roc

mpo_linkdims_d = maxlinkdim(solver_state.ops.dx)
mpo_linkdims_dd = maxlinkdim(solver_state.ops.ddx)
mpo_linkdims_mix = maximum(maxlinkdim.([solver_state.ops.dxdx, solver_state.ops.dydy, solver_state.ops.dxdy, solver_state.ops.dydx]))

@assert maxlinkdim(solver_state.ops.dy) == mpo_linkdims_d
@assert maxlinkdim(solver_state.ops.ddy) == mpo_linkdims_dd

@assert maxlinkdim(solver_state.ux) == maxlinkdim(solver_state.uy)

print("$(time_elapsed()) | solver state size (M=$M, χ=$χ, linkdim=$(maxlinkdim(solver_state.ux)), mpo_linkdim=$mpo_linkdims_d/$mpo_linkdims_dd/$mpo_linkdims_mix): $(Base.format_bytes(Base.summarysize(solver_state)))")
flush(stdout)
# MPSCFD.move_tensors!(solver_state, move)
println(" -> $(Base.format_bytes(Base.summarysize(solver_state)))")

print("\x1b[?25l") # turn cursor off

t_start = (it_start + 1) * Δt
ts = t_start:Δt:T+Δt

iteration_times::Vector{Float64} = [0.0]

for (it2, t) in enumerate(ts)

    time_remaining = mean(iteration_times) * (length(ts) - it2)

    it = it_start + it2

    t = round(t; digits=ceil(Int64, 1 - log10(Δt)))

    start = time()

    print("\x1b[2K$(time_elapsed()) | $(format_time(time_remaining)) | starting timestep it=$it, t=$(@sprintf("%.4f", t))...\r")
    flush(stdout)

    convergence, rhs = MPSCFD.solve_timestep!(solver_state, params, ϵ=ϵ) # move=move

    h5open(filename, "cw") do file
        write(file, "it=$it,x", solver_state.ux)
        write(file, "it=$it,y", solver_state.uy)
    end

    numiter = sum(Iterators.map(((info, _),) -> info.numiter, Iterators.flatten(convergence)))
    numops = sum(Iterators.map(((info, _),) -> info.numops, Iterators.flatten(convergence)))
    rmax = ceil(Int64, log10(maximum(Iterators.map(((info, _),) -> info.normres, convergence[end]))))

    setup = sum(Iterators.map(((_, (start, setup_done, done)),) -> setup_done - start, Iterators.flatten(convergence)))
    elapsed = sum(Iterators.map(((_, (start, setup_done, done)),) -> done - start, Iterators.flatten(convergence)))

    setup = lpad(round(Int64, setup / elapsed * 100), 3, " ")
    elapsed = round(elapsed; digits=1)

    finish = time()

    total = round(finish - start; digits=1)
    rhs = round(rhs; digits=1)

    push!(iteration_times, total)
    if length(iteration_times) > 10 || it2 <= 2
        popfirst!(iteration_times)
    end

    println("\x1b[2K$(time_elapsed()) | $(format_time(time_remaining)) | [it=$it, t=$(@sprintf("%.4f", t))\t] timestep took $(total)s\t= iteration $(elapsed)s\t($setup% setup) + $(rhs)s\trhs. $(length(convergence))\tsweeps, $numiter\titerations and $numops\toperator applications. Maximum residual is E$rmax")
    # @show MPSCFD.time_spent_permuting(solver_state)

    if precompile
        if it >= 2
            break
        end
    end
end

println("\x1b[?25h") # turn cursor back on