mutable struct CachedPartialContraction
    LR::Vector{ITensor}
    b::Int64
    H::MPO
end

Base.length(contraction::CachedPartialContraction) = length(contraction.H)

function Base.getindex(contraction::CachedPartialContraction, j::Int64)
    j < 1 && return ITensor(1)
    j > length(contraction) && return ITensor(1)
    return contraction.LR[j]
end

CachedPartialContraction(H::MPO)::CachedPartialContraction = CachedPartialContraction(Vector{ITensor}(undef, length(H)), -1, H)

function compute_lr!(contraction::CachedPartialContraction, psi_bra::MPS, psi_ket::MPS, range::AbstractRange{Int64})
    # println("compute_lr!($(collect(range)))")

    dir = step(range)
    for i in range

        L = contraction[i-dir]
        M = contraction.H[i]

        L = ((L * psi_ket[i]) * M) * prime(dag(psi_bra[i]))
        contraction.LR[i] = L

        if length(inds(L)) > 4
            @show inds(L)
            error("probably tried to permform partial contraction for mps with mismatching indices")
        end
    end
end

function position!(contraction::CachedPartialContraction, psi_bra::MPS, psi_ket::MPS, b::Int64)

    N = length(contraction.H)
    @assert b >= 1 && b <= N

    if contraction.b < 1
        compute_lr!(contraction, psi_bra, psi_ket, 1:b-1)
        compute_lr!(contraction, psi_bra, psi_ket, N:-1:b+1)
    elseif b == contraction.b
        return contraction
    else
        r = b > contraction.b ? (contraction.b:b-1) : (contraction.b:-1:b+1)
        compute_lr!(contraction, psi_bra, psi_ket, r)
    end

    contraction.b = b
    return contraction
end


# multiply cached contraction with T
function product(contraction::CachedPartialContraction, T::ITensor)::ITensor

    b = contraction.b
    L = contraction[b-1]
    R = contraction[b+1]
    M = contraction.H[b]

    A = L * T
    B = A * M
    C = B * R
    return C
end

#=
function reset_lr!(P::CachedPartialContraction)::CachedPartialContraction
    P.LR = Vector{ITensor}(undef, N)
    P.b = -1
    P
end

function apply_to_all(f::Function, state::CachedPartialContraction)::CachedPartialContraction

    LR = f.(state.LR)
    b = state.b
    N = state.N
    contract = state.contract

    CachedPartialContraction(LR, b, N, contract)
end


# permute indices of tensor A to allow for efficient contraction with tensors
# with the same index order as B
function permute_for_contraction(A::ITensor, B::ITensor)::ITensor

    sort_by = inds(B)
    common = commoninds(A, B)
    by = ind -> findfirst(sort_by, ind)
    is_link = ind -> hastags(ind, "Link")

    sort!(common; by=by)
    remainder = uniqueinds(A, B)
    sort!(remainder; by=is_link, rev=true)
    indices = vcat(common, remainder)
    reverse!(indices)

    # @show id.(inds(A)) .== id.(indices)

    ITensors.permute(A, indices; allow_alias=true)
end

# permute indices of tensors A and B to allow for efficient contraction with 
# each other, 
function permute_for_contraction_chain!(tensors::Vector{ITensor})::Vector{ITensor}

    for i in 1:length(tensors)

        @show i

        indices = collect(inds(tensors[i]))

        if i > 1
            @show inds(tensors[i-1])
        end

        @show indices


        sort!(indices, by=index -> begin

            a = if i > 1
                findfirst(inds(tensors[i-1]) .== index)
            else
                nothing
            end

            if isnothing(a)
                a = -1
            end

            @show a

            a
        end)
    end

    # tensors
end

function _contract_into(res::ITensor, tensors::Vector{ITensor})::ITensor
    for T in tensors
        if isnothing(res)
            res = T
        else
            res *= T
        end
    end
    res
end

# permute the internal representation of the contraction to allow for calls to `product` with a tensor of the
# same index order as T # to be more efficient
function optimize!(contraction::CachedPartialContraction, T::ITensor; link::Union{Nothing,ITensor}=nothing)::Union{Nothing,ITensor}

    # TODO
    return link

    tensors::Vector{ITensor} = []

    if contraction.b > 1
        push!(tensors, contraction.LR[contraction.b-1])
    end

    push!(tensors, T)

    if !isnothing(link)
        push!(tensors, link)
    end

    if contraction.b < contraction.N
        push!(tensors, contraction.LR[contraction.b+1])
    end

    tensors = permute_for_contraction_chain!(tensors)

    if contraction.b > 1
        contraction.LR[contraction.b-1] = popfirst!(tensors)
    end

    link = if isnothing(link)
        nothing
    else
        popfirst!(tensors)
    end

    T = popfirst!(tensors)

    if contraction.b < contraction.N
        contraction.LR[contraction.b+1] = popfirst!(tensors)
    end

    link
end

=#
