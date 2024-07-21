
# construct the lienar operator O, that when applied to the trial solution tensors
function make_linear_operator(state::SolverState, params::Parameters, b::Int64)::Function
    orthogonalize!(state.ux, b)
    orthogonalize!(state.uy, b)

    (; N) = get_params(state.ux)


    # this is used in the paper
    # Δx = params.L * 2.0^-N
    # λ = Δx^2 * 10^4

    # this results from my dervivation
    λ = params.μ

    # prepare for computing d/dc <V_i|dx_i d_xj|V_j>
    position!(state.contractions.Pxx, state.ux, state.ux, b)
    position!(state.contractions.Pxy, state.ux, state.uy, b)
    position!(state.contractions.Pyx, state.uy, state.ux, b)
    position!(state.contractions.Pyy, state.uy, state.uy, b)

    function O(α::SiteTrialSolution)::SiteTrialSolution

        αx = α.x
        αy = α.y

        # compute H_ij = d/dc <V_i|dx_i d_xj|V_j>
        Hxx = noprime(product(state.contractions.Pxx, αx))
        Hxy = noprime(product(state.contractions.Pxy, αy))
        Hyx = noprime(product(state.contractions.Pyx, αx))
        Hyy = noprime(product(state.contractions.Pyy, αy))

        # Hxx, Hxy, Hyx, Hyy = pmap(((P, T),) -> noprime(product(P, T)), [
        #     (state.contractions.Pxx, αx),
        #     (state.contractions.Pxy, αy),
        #     (state.contractions.Pyx, αx),
        #     (state.contractions.Pyy, αy),
        # ])

        # compute H|α>
        Hx = Hxx + Hxy
        Hy = Hyy + Hyx

        # compute O|α>
        # with O = I - λ * H
        # O is positive semi definite because H is negative semidefinite
        resx = αx - λ * Hx
        resy = αy - λ * Hy

        resx = ITensors.permute(resx, inds(state.ux[b]); allow_alias=true)
        resy = ITensors.permute(resy, inds(state.uy[b]); allow_alias=true)

        SiteTrialSolution(resx, resy)
    end
end

# reverse mapping, apply function to unmapped fields, redo mapping
# function unmapped(f::Function, params...; kwargs...)::MPS
#     new_params = []
#     for p in params
#         if typeof(p) == MPS || typeof(p) == ITensor

#             T = if typeof(p) == MPS
#                 contract(p)
#             else
#                 p
#             end

#             push!(new_params, reverse_mapping(T))
#         else
#             push!(new_params, p)
#         end
#     end

#     T = do_mapping(f(new_params...))
#     MPS(T, inds(T); kwargs...)
# end

# # compute the equivalent of Had(u1)|u2>
# function Hd(u1::MPS, u2::MPS)::MPS
#     res = unmapped((a, b) -> a .* b, u1, u2)
#     for i in eachindex(u1)
#         replaceind!(res[i], siteinds(res)[i], siteinds(u1)[i])
#         replaceind!(res[i], siteinds(res)[i], siteinds(u1)[i])
#     end
#     res
# end

# contruct the terms to be summed up into the right hand side
function make_rhs(ops::Operators, initial::Tuple{MPS,MPS}, midpoint::Tuple{MPS,MPS}, params::Parameters; alg="naive")::Tuple{RhsTerms,RhsTerms}

    Tuple(Iterators.map(zip(initial, midpoint)) do (u_1, u_2)
        terms = []

        push!(terms, RhsTerm(1.0, u_1))

        push!(terms, RhsTerm(-params.Δt / 2, apply(hadamard_mpo(midpoint[1]), ops.dx; maxdim=params.χ_rhs, cutoff=params.cutoff_rhs, alg), u_2))
        push!(terms, RhsTerm(-params.Δt / 2, apply(hadamard_mpo(midpoint[2]), ops.dy; maxdim=params.χ_rhs, cutoff=params.cutoff_rhs, alg), u_2))

        push!(terms, RhsTerm(-params.Δt / 2, apply(ops.dx, hadamard_mpo(midpoint[1]); maxdim=params.χ_rhs, cutoff=params.cutoff_rhs, alg), u_2))
        push!(terms, RhsTerm(-params.Δt / 2, apply(ops.dy, hadamard_mpo(midpoint[2]); maxdim=params.χ_rhs, cutoff=params.cutoff_rhs, alg), u_2))

        push!(terms, RhsTerm(+params.Δt * params.ν, ops.ddx, u_2))
        push!(terms, RhsTerm(+params.Δt * params.ν, ops.ddy, u_2))

        RhsTerms(terms)
    end)
end


# NOTE: state.ux and state.uy must be orthogonalized at b
# compute the right hand side tensors for by adding up the tensors produced by partially contracting each term
# in `state.rhs` with the current trial solution
function compute_β(state::SolverState, b::Int64)::Tuple{ITensor,ITensor}
    Tuple(Iterators.map(zip(state.rhs, (state.ux, state.uy))) do (terms, u)
        compute_β(terms, u, b)
    end)
end


