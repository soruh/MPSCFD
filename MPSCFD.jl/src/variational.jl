struct Operators
    dx::MPO
    dy::MPO
    ddx::MPO
    ddy::MPO
    dxdx::MPO
    dydy::MPO
    dxdy::MPO
    dydx::MPO
end

function Operators(ux::MPS, uy::MPS, δx::Float64, params::Parameters; alg="naive")::Operators
    # we define y as the first dimension and x as the second dimension
    dx = central_differences(ux; dim=2, nth=1, order=params.order, dx=δx, maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg)
    dy = central_differences(uy; dim=1, nth=1, order=params.order, dx=δx, maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg)
    ddx = central_differences(ux; dim=2, nth=2, order=params.order, dx=δx, maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg)
    ddy = central_differences(uy; dim=1, nth=2, order=params.order, dx=δx, maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg)

    dxdx = apply(dx, dx; maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg=alg)
    dydy = apply(dy, dy; maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg=alg)
    dxdy = apply(dx, dy; maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg=alg)
    dydx = apply(dy, dx; maxdim=params.χ_mpo, cutoff=params.cutoff_mpo, alg=alg)
    @assert dxdy ≈ dydx
    dxdy = dydx

    Operators(
        dx,
        dy,
        ddx,
        ddy,
        dxdx,
        dydy,
        dxdy,
        dydx,
    )
end

struct CachedSolverContractions
    Pxx::CachedPartialContraction
    Pxy::CachedPartialContraction
    Pyx::CachedPartialContraction
    Pyy::CachedPartialContraction
end

mutable struct SolverState
    ops::Operators

    contractions::CachedSolverContractions

    ux::MPS
    uy::MPS

    rhs::Union{Nothing,Tuple{RhsTerms,RhsTerms}}
end

function CachedSolverContractions(ops::Operators)::CachedSolverContractions
    Pxx = CachedPartialContraction(ops.dxdx)
    Pxy = CachedPartialContraction(ops.dxdy)
    Pyx = CachedPartialContraction(ops.dydx)
    Pyy = CachedPartialContraction(ops.dydy)

    CachedSolverContractions(
        Pxx, Pxy,
        Pyx, Pyy,
    )
end

function SolverState(; ops::Operators, ux::MPS, uy::MPS)::SolverState
    SolverState(ops, CachedSolverContractions(ops), ux, uy, nothing)
end

# time_spent_permuting(contractions::CachedSolverContractions)::Float64 = sum(P -> P.time_permute, [contractions.Pxx, contractions.Pxy, contractions.Pyx, contractions.Pyy])
# time_spent_permuting(rhs::RhsTerms)::Float64 = sum(term -> term.P.time_permute, rhs.terms)
# time_spent_permuting(rhs::Tuple{RhsTerms,RhsTerms})::Float64 = time_spent_permuting(rhs[1]) + time_spent_permuting(rhs[2])
# time_spent_permuting(state::SolverState)::Float64 = time_spent_permuting(state.contractions) + (isnothing(state.rhs) ? 0.0 : time_spent_permuting(state.rhs))

# `solve` a single site by minimizing the penalty function `Θ` with respect to the tensor entries of the
# current bond `c`.
# This is done by solving dΘ/dc = 0.
# The solution of dΘ/dc = 0 can be put into the form O * α = β which is solved using `linsolve`.
function solve_single_site!(state::SolverState, params::Parameters, b::Int64)

    start = time()

    orthogonalize!(state.ux, b)
    orthogonalize!(state.uy, b)

    βx, βy = compute_β(state, b)

    βx = ITensors.permute(βx, inds(state.ux[b]); allow_alias=true)
    βy = ITensors.permute(βy, inds(state.uy[b]); allow_alias=true)

    O = make_linear_operator(state, params, b)

    α = SiteTrialSolution(state.ux[b], state.uy[b])
    β = SiteTrialSolution(βx, βy)

    setup_done = time()

    γ, info = linsolve(O, β, α; isposdef=true, issymmetric=true, params.linsolve...)

    state.ux[b] = γ.x
    state.uy[b] = γ.y

    done = time()

    info, (start, setup_done, done)
end

# iterate over each bond of the MPS forwards and backwards (ommiting the first bond on the backwards sweep)
# call solve_single_site! on each iteration.
function perform_sweep!(state::SolverState, params::Parameters; residuals=[])
    (; N) = get_params(state.ux)

    infos = []
    dir = true
    for b in Iterators.flatten([1:N, N-1:-1:2])
        if b == N
            dir = false
        end

        let
            print("\x1b[s ")
            dir_char = dir ? ">" : "<"

            print(dir_char)
            maxres = maximum(residuals)
            for i in 1:N
                v = 7 - round(Int64, -log10(residuals[i] / maxres))
                res_char = (v < 0) ? "·" : Char(0x2581 + v)
                s = (i == b) ? "\x1b[31m$res_char\x1b[0m" : res_char
                print(s)
            end
            print(dir_char)

            print("\x1b[u")
            flush(stdout)
        end

        info = solve_single_site!(state, params, b)
        push!(infos, info)
        residuals[b] = info[1].normres
    end
    infos
end

# repeatedly call `perform_sweep!` on the iteration state until the kinetic energy difference convergence crietion is met.
# The convergence criterion checks if the difference in kinetic enery caused by the sweep is less than the provided parameter ϵ
function sweep_to_convergence!(state::SolverState, params::Parameters, ϵ::Float64; name="")

    ΔEk_prev = nothing
    Ek_prev = nothing
    convergence = []
    i_sweep = 0

    residuals = ones(length(state.ux)) * 1e-40

    while true
        i_sweep += 1

        print("\x1b[2KΔEkin=")
        if isnothing(ΔEk_prev)
            print("?")
        else
            print("$(round(ΔEk_prev; sigdigits=2))")
        end

        print(" $(name)sweep $i_sweep...")
        flush(stdout)

        info = perform_sweep!(state, params; residuals=residuals)
        push!(convergence, info)

        Ek = inner(state.ux, state.ux) + inner(state.uy, state.uy)

        t_sweep = sum(Iterators.map(((_, t),) -> t[3] - t[1], info))

        print("done. sweep took $(round(t_sweep; digits=2))s\r")
        flush(stdout)

        if !isnothing(Ek_prev)
            ΔEk = abs(((Ek - Ek_prev) / Ek_prev))
            if ΔEk <= ϵ
                break
            end
            ΔEk_prev = ΔEk
        end

        Ek_prev = Ek

        # operator_applications = sum(Iterators.map(x -> x[1].numops, info)) / (2N - 2)
        # if operator_applications ≈ 1.0
        #     break
        # end
    end

    convergence
end

function solve_timestep!(state::SolverState, params::Parameters; ϵ=1e-15, euler=false)

    initial = deepcopy.((state.ux, state.uy),)

    if euler
        # euler time-step

        state.contractions = CachedSolverContractions(state.ops)

        start_rhs = time()
        state.rhs = make_rhs(state.ops, initial, initial, params)
        end_rhs = time()

        convergence = sweep_to_convergence!(state, params, ϵ; name="euler ")

        return convergence, end_rhs - start_rhs
    else
        # RK2 midpoint method time-step

        state.contractions = CachedSolverContractions(state.ops)

        # compute new rhs
        start_rhs_1 = time()
        state.rhs = make_rhs(state.ops, initial, initial, half_Δt(params))
        end_rhs_1 = time()

        convergence_1 = sweep_to_convergence!(state, half_Δt(params), ϵ; name="half ")

        midpoint = deepcopy.((state.ux, state.uy))
        # state.ux, state.uy = deepcopy.(initial)

        state.contractions = CachedSolverContractions(state.ops)

        start_rhs_2 = time()
        state.rhs = make_rhs(state.ops, initial, midpoint, params)
        end_rhs_2 = time()

        convergence_2 = sweep_to_convergence!(state, params, ϵ; name="full ")

        return vcat(convergence_1, convergence_2), end_rhs_1 - start_rhs_1 + end_rhs_2 - start_rhs_2
    end
end


function setup_jet(M::Int64, jet_params::JetParams, params::Parameters; alg="naive")::SolverState

    N = 2^M
    dx = jet_params.L_box / N
    r = range(0.0, jet_params.L_box; length=N + 1)[1:N]

    J = jet(jet_params, r, r)
    ux, uy = do_mapping.(J)
    uy = replaceinds!(uy, inds(uy), inds(ux))

    ux = MPS(ux, inds(ux); maxdim=params.χ, cutoff=params.cutoff)
    uy = MPS(uy, inds(uy); maxdim=params.χ, cutoff=params.cutoff)

    ops = Operators(ux, uy, dx, params; alg)
    SolverState(; ops=ops, ux=ux, uy=uy)
end
