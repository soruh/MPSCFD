using MPSCFD: operator_sum_mpo, LadderOperator, RaisingOperator, do_mapping_mps, reverse_mapping
using ITensors: apply, norm, MPS

using Test

function test_shift(ψf::MPS, data, operator::LadderOperator, shift::Int64, dimensions::Int64, dimension::Int64)
    op = operator_sum_mpo(ψf, operator; cutoff=0)

    nrmψ = norm(ψf)
    ψf /= nrmψ

    ψf2 = apply(op, ψf; alg="naive", cutoff=0)
    f2 = reverse_mapping(ψf2) * nrmψ

    shifts = map(d -> d == dimension ? shift : 0, 1:dimensions)
    norm(f2 .- circshift(data, shifts))
end

@testset "ladder_operator" begin

    N = 5
    for dimensions in 1:3

        x = range(start=-2, stop=2, length=2^N)
        prms = (α=1, β=1.1, γ=3, ω=4)
        fl(x) = trialfunction1d(x; prms...)

        data = zeros(repeat([2^N], dimensions)...)

        for dimension in 1:dimensions

            for index in CartesianIndices(data)
                data[index] = fl(index[dimension])
            end

            ψf = do_mapping_mps(data)

            S = RaisingOperator(dimension)
            for p in -N:N
                O = p > 0 ? S : adjoint(S)
                @test test_shift(ψf, data, O^abs(p), -p, dimensions, dimension) ≈ 0.0 atol = 1e-11
            end
        end
    end
end
