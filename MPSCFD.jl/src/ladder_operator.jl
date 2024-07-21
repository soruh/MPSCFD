function _half_adder_matrix()
    T = zeros(Float64, 2, 2, 2, 2)

    for i in CartesianIndices(T)
        a, b, s, sp = Tuple(i) .- 1

        if (s ⊻ b == sp) && (s & b == a)
            T[i] = 1.0
        end
    end

    T
end

function RaisingOperator_mpo(mps::MPS, x::Int64)::MPO

    half_adder_tensor(; left::Index, right::Index, down::Index, up::Index)::ITensor = ITensor(_half_adder_matrix(), [left, right, up, down])

    (; N, dim) = MPSCFD.get_params(mps)
    @assert x in 1:dim "ladder operator direction out of bounds"

    left = link = Index(2, "Link,i=0")

    tensors = Vector{ITensor}()

    for i in 1:N

        # create combiner with one subindex for each bit
        subinds = map(1:dim) do d
            Index(2, "i=$i,d=$d")
        end
        C = combiner(subinds...)

        # reverse dimension index to match mapping
        reverse!(subinds)

        # make the combined index the site index at the current site
        replaceind!(C, combinedind(C), siteinds(mps)[i])

        new_link = Index(2, "Link,i=$i")
        T = half_adder_tensor(;
            left=link,
            right=new_link,
            up=subinds[x],
            down=prime(subinds[x]),
        )
        link = new_link

        # add identity for all indices we're not incrementing
        for d in 1:dim
            if d != x
                index = subinds[d]
                T *= delta(index, prime(index))
            end
        end

        # contract bit indices back to single site index
        T = (T * C) * prime(C)

        push!(tensors, T)
    end

    right = link

    # todo: get a better understanding for this
    tensors[1] *= ITensor([1.0, 1.0], left)
    tensors[end] *= onehot(right => 2)

    MPO(tensors)
end
