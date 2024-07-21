# build a MPO from an operator sum
function operator_sum_mpo(mps::MPS, sum::LadderOperatorSum; alg="naive", cutoff=1E-13, maxdim=nothing)::MPO
    (; dim) = get_params(mps)

    id = 0.0

    operator_order = maximum(k -> k.power, keys(sum.terms))

    terms = map(1:dim) do _
        spzeros(Float64, operator_order, 2)
    end

    for (term, factor) in sum.terms

        if iszero(term.power)
            id += factor
            continue
        end

        d = term.dimension
        # convert false/true to 1, 2
        s = 1 + term.is_adjoint

        terms[d][term.power, s] = factor
    end

    res = id * MPO(siteinds(mps), "I")

    for d in 1:dim

        if SparseArrays.nnz(terms[d]) == 0
            # no elements in dimension. Don't contsruct ladder operator at all
            continue
        end

        # build the initial ladder operator
        Sp0 = RaisingOperator_mpo(mps, d)
        Sp = nothing

        # only build needed operators for this dimension
        dim_operator_order = maximum(SparseArrays.findnz(terms[d])[1])
        for p in 1:dim_operator_order
            fp, fm = terms[d][p, 2], terms[d][p, 1]

            # progressively multiply the ladder operator with the base operator
            # to achieve further shifts
            if isnothing(Sp)
                Sp = copy(Sp0)
            else
                Sp = apply(Sp, Sp0; alg, cutoff, maxdim)
            end

            # add operator for fp * (S+)^p, fm * (S-)^p
            if alg == "naive"
                res = add(res, fp * Sp, fm * transpose(Sp); alg="directsum")
                res = truncate!(res; maxdim, cutoff)
            else
                res = add(res, fp * Sp, fm * transpose(Sp); alg, maxdim, cutoff)
            end
        end
    end

    res
end

operator_sum_mpo(mps::MPS, sum::LadderOperator; alg="naive", cutoff=1E-13, maxdim=nothing)::MPO = operator_sum_mpo(mps, convert(LadderOperatorSum, sum); alg, cutoff, maxdim)

transpose(mpo::MPO)::MPO = swapprime(dag(mpo), 1 => 0)

function central_differences(mps::MPS; dim::Int64, nth::Int64, order::Int64, dx::Float64, alg="naive", cutoff=1E-13, maxdim=nothing)::MPO
    @assert dim >= 1 && dim <= get_params(mps).dim "difference operator dimension out of bounds"

    coefficients = central_difference_coefficients(nth, order)
    operator_sum = finite_difference_coefficients_to_opsum(dim, coefficients)
    operator_sum_mpo(mps, operator_sum / dx^nth; alg, cutoff, maxdim)
end

function central_difference_laplacian(mps::MPS; order::Int64, dx::Float64)::MPO
    (; dim) = get_params(mps)

    nth = 2

    coefficients = central_difference_coefficients(nth, order)
    operator_sum = mapreduce(d -> finite_difference_coefficients_to_opsum(d, coefficients), +, 1:dim)

    operator_sum_mpo(mps, operator_sum / dx^nth)
end

function hadamard_mpo(mps::MPS)::MPO
    tensors::Vector{ITensor} = []
    for i in 1:length(mps)
        T = mps[i]

        siteinds = filter(i -> !hastags(i, "Link"), inds(T))
        @assert length(siteinds) == 1
        site = siteinds[1]

        tmp = Index(dim(site), "tmp")

        T = replaceind(T, site, tmp)

        push!(tensors, T * delta(tmp, site, prime(site)))
    end
    MPO(tensors)
end