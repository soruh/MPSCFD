# contract the mps stored in the .mat file into a ITensors tensor
function convert_to_itensors_mps(u)::MPS

    N = length(u["nodes"]["tensor"])

    @assert Int64(u["start"]) == 0
    @assert Int64(u["end"]) == N - 1

    @assert u["start_leg"] == "L"
    @assert u["end_leg"] == "R"

    @assert size(u["nodes"]["ids"]) == (1, N)
    ids = reshape(u["nodes"]["ids"], N)

    @assert size(u["nodes"]["indices"]) == (1, N)
    indices = map(reshape(u["nodes"]["indices"], N)) do inds
        reshape(Int64.(inds), 3)
    end

    @assert all(indices) do inds
        inds == [0, 1, 2]
    end

    connections = map(u["connections"]) do c
        !(c isa Matrix{Union{}})
    end
    @assert connections == (u["connections"] .== "RL")

    tensors = Vector{ITensor}()
    index_maps = []

    for i in 1:N

        T = u["nodes"]["tensor"][i]
        @assert T["elems_type"] == "values"

        # MatLab also uses CMO
        V = reshape(T["elems"]["vals"], Tuple(Int64.(T["dims"])))

        index_names = ids[i]

        @assert length(size(V)) == 3

        indices = []

        index_map = Dict()

        for d in 1:length(size(V))

            dim = size(V)[d]

            # create indicies tagged with the direction specified in the mat file
            i = Index(dim; tags=string(index_names[d]))

            # store indices by bond and direction
            index_map[index_names[d]] = i

            push!(indices, i)
        end

        T = ITensor(V, indices)

        push!(tensors, T)
        push!(index_maps, index_map)
    end


    # perform contractions specified in the mat file by converting the
    # specified indices to matching link indices
    for i in 1:N-1
        c = connections[i, :]
        j = findfirst(c)
        @assert isnothing(findfirst(c[j+1:end]))

        connection = u["connections"][i, j]

        # find indices to contract
        a = index_maps[i][connection[1]]
        b = index_maps[j][connection[2]]

        i_link = Index(dim(a); tags="Link,i=$i")

        tensors[i] *= delta(a, i_link)
        tensors[j] *= delta(i_link, b)
    end

    # remove the extra bonds on the left and right
    iL = findfirst(ind -> hastags(ind, "L"), inds(tensors[1]))
    iR = findfirst(ind -> hastags(ind, "R"), inds(tensors[end]))

    @assert !isnothing(iL)
    @assert !isnothing(iR)

    tensors[1] *= onehot(inds(tensors[1])[iL] => 1)
    tensors[end] *= onehot(inds(tensors[end])[iR] => 1)

    MPS(tensors)
end
function load_reference_mps(path::String)::Tuple{MPS,MPS}
    vars = matread(path)

    mpsx = convert_to_itensors_mps(vars["u1"])
    mpsy = convert_to_itensors_mps(vars["u2"])

    mpsx, mpsy
end

function load_reference(path::String)::Tuple{Matrix,Matrix}
    mpsx, mpsy = load_reference_mps(path)

    ux = reverse_mapping(contract(mpsx))
    uy = reverse_mapping(contract(mpsy))

    ux, uy
end
