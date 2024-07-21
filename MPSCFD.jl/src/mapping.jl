# return the binary representation of `value` with `N` bits
binary_representation(value::Int64, N::Int64) = Iterators.map(i -> Bool((value >> i) & 1), reverse(0:N-1))

struct MappingParams
    N::Int64
    dim::Int64
end

function get_params(T::ITensor)::MappingParams
    N = length(size(T))
    site_dimension = size(T)[1]
    @assert all(size(T) .== site_dimension) "all indices of mapped field must have the same dimension"

    MappingParams(N, Int64(log2(site_dimension)))
end

function get_params(mps::MPS)::MappingParams
    N = length(mps)
    site_dimension = dim(siteinds(mps)[1])
    @assert all(dim.(siteinds(mps)) .== site_dimension) "all site indices of mapped field must have the same dimension"

    MappingParams(N, Int64(log2(site_dimension)))
end

# map a n-dimenstional cartesian index to a vector of site indices representing the position in the
# mapped tensor
function map_cartesian_index(r::CartesianIndex, N::Int64, ω::Vector{Int64})::Vector{Int64}
    K = length(r)
    fill!(ω, 0)

    # for each dimension
    for i in 1:K

        # for each length scale
        for (n, σ) in enumerate(binary_representation(r[i], N))
            # if σ[n] is set, set the bit corresponding to dimension i
            # (the i-th most significant bit) in the entry of ω corresponding
            # to length scale n
            ω[n] |= Int64(σ) << (K - i)

            # NOTE:
            # in zero based indexing we would want to set bit
            # (K-1) - i_z as the most significant is K-1
            # as we use one based indexing i = i_z + 1
            # so we set bit K - 1 - (i - 1) = K - i
        end
    end

    # add one to each index to convert to one based indexing
    ω .+= 1
    ω
end

map_cartesian_index(r::CartesianIndex, N::Int64)::Vector{Int64} = map_cartesian_index(r, N, zeros(Int64, N))

function do_mapping(data)::ITensor
    @assert count_ones(size(data)[1]) == 1 "tensor size must be a power of 2 >= 1 but is $(size(data)[1])"
    @assert all(size(data) .== size(data)[1]) "tensor must be the smae in each direction"
    dim = ndims(data)

    N = Int64(log2(size(data)[1]))

    indices = map(1:N) do i
        Index(2^dim, "index_$i")
    end

    U = zeros(Float64, ITensors.dim.(indices)...)
    ω = zeros(Int64, N)

    for i in CartesianIndices(data)

        # do setindex without any permutations, as the tensor
        # is created with indices in the correct order
        inds = map_cartesian_index(i, N, ω)
        U[inds...] = data[i]
    end

    ITensor(U, indices)
end

function do_mapping_mps(data)::MPS
    T = do_mapping(data)
    MPS(T, inds(T))
end

# A faster implementation of sub2ind than the one that is polyfilled in NDTensors.
function better_sub2ind(axes, inds)::Int64
    res = 1
    L = 1

    for (axis, index) in Iterators.zip(axes, inds)
        res += (index - 1) * L
        L *= length(axis)
    end

    res
end

reverse_mapping(mps::MPS) = reverse_mapping(contract(mps))
function reverse_mapping(T::ITensor)

    (; N, dim) = get_params(T)

    sizes = (2^N for _ in 1:dim)
    res = zeros(Float64, sizes...)

    indices = inds(T)

    p = NDTensors.getperm(inds(T), indices)
    ax = axes(NDTensors.tensor(T))
    ω = zeros(Int64, N)

    T̄ = NDTensors.expose(NDTensors.data(T))

    for i in CartesianIndices(res)
        permute_inputs = map_cartesian_index(i, N, ω)
        inds = map(i -> permute_inputs[i], p)
        res[i] = T̄[better_sub2ind(ax, inds)]
    end

    res
end
