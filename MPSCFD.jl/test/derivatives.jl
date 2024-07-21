using MPSCFD: central_difference_coefficients, reverse_mapping, central_differences, do_mapping_mps, jet, JetParams
using ITensors: apply, norm
using FiniteDifferences: central_fdm, grad
using Test

function test_derivative_1d(N::Int64, order::Int64)::Float64

    nth = 1

    n_coeef = 2 * floor(Int64, nth + 1 / 2) - 1 + order
    p = (n_coeef - 1) ÷ 2

    cdm = central_fdm(n_coeef, nth; adapt=0, condition=1.0, factor=1.0)

    x = range(start=-2, stop=2, length=2^N)

    δx = step(x)

    prms = (α=1, β=1.1, γ=3, ω=4)
    fl(x) = trialfunction1d(x; prms...)
    dfl(x) = cdm(fl, x, δx)

    data = fl.(x)
    derivative = dfl.(x)

    ψf = do_mapping_mps(data)

    dx = central_differences(ψf; dim=1, nth=nth, order, dx=δx, alg="naive", maxdim=nothing, cutoff=0)
    res = reverse_mapping(apply(dx, ψf; alg="naive", maxdim=nothing, cutoff=0))

    differences = res .- derivative
    errors = differences[1+p:end-p]

    norm(errors)
end

function custom_derivative_with_coefficients(nth::Int64, order::Int64, dim::Int64, δx::Float64, data)
    cf = central_difference_coefficients(nth, order)
    map(CartesianIndices(data)) do index

        index = [Tuple(index)...]
        j = index[dim]

        #=
        lft = j - cf.center + 1
        rft = j - cf.center + length(cf.coeff)
        lft < 1 && return 0
        rft > size(data)[dim] && return 0
        =#

        mapreduce(+, 1:length(cf.coeff)) do k
            off = k - cf.center
            index[dim] = j + off

            # periodic boundary
            index[dim] = 1 + (index[dim] + size(data)[dim] - 1) % size(data)[dim]

            data[index...] * cf.coeff[k]
        end / δx^nth
    end
end

function test_derivative(data, dim::Int64; is_periodic=false)

    dims = length(size(data))
    N = size(data)[1]
    M = Int64(log2(N))
    @assert all(x -> x == N, size(data))

    order = 8
    nth = 1

    n_coeef = 2 * floor(Int64, nth + 1 / 2) - 1 + order
    p = (n_coeef - 1) ÷ 2

    xs = range(0.0, 1.0; length=N)
    δx = step(xs)

    #=
    derivative = map(x -> let
            g = grad(cdm, (x, unk) -> let
                    indices = collect(round.(Int64, x .* (N - 1)))

                    indices .+= N
                    indices .%= N
                    indices .+= 1

                    data[indices...]
                end, x, δx)
            g[1][dim]
        end, Iterators.product(ranges...))
    =#

    ψf = do_mapping_mps(data)

    derivative = custom_derivative_with_coefficients(nth, order, dim, δx, data)

    # println("switching dim: $dim -> $(1 + dims - dim)")
    # dim = 1 + dims - dim

    dx = central_differences(ψf; dim=dim, nth=nth, order, dx=δx, alg="naive", maxdim=nothing, cutoff=0)
    res = reverse_mapping(apply(dx, ψf; alg="naive", maxdim=nothing, cutoff=0))

    differences = res .- derivative

    errors = if is_periodic
        differences
    else
        differences[repeat([1+p:N-p], dims)...]
    end

    maximum(errors)
end

@testset "derivatives" begin

    M = 6

    for dims in 1:3

        function f(fs)
            xs = 4.0 .* fs .- 2.0

            prms(i) = (
                α=1.1 - i / 10,
                β=1.0 + i / 10,
                γ=4.0 - i,
                ω=2 + 2i,
            )
            sum(((i, x),) -> trialfunction1d(x; prms(i)...), enumerate(xs)) / dims
        end

        xs = range(0.0, 1.0; length=2^M + 1)[1:end-1]
        ranges = repeat([xs], dims)
        data = map(f, Iterators.product(ranges...))

        for dim in 1:dims
            @test test_derivative(data, dim) ≈ 0.0 atol = 1e-10
        end
    end

    jet_params = JetParams(1.0, 1 / 200, 0.4, 0.6, 1.0)
    r = range(0.0, 1.0, length=(2^M + 1))[1:end-1]
    jet_data = jet(jet_params, r, r)
    for dim in 1:2
        for i in 1:2
            @test test_derivative(jet_data[i], dim; is_periodic=true) ≈ 0.0 atol = 1e-12
        end
    end

    for N in 6:1:10
        for order in 2:2:16
            @test test_derivative_1d(N, order) ≈ 0.0 atol = 1e-9
        end
    end
end
