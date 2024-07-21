using MPSCFD: do_mapping, reverse_mapping, hadamard_mpo
using ITensors: MPS, ITensor, apply, replaceinds!, inds, siteinds

using Test

function test_data_1d(x)::ITensor
    prms = (α=1, β=1.1, γ=3, ω=4)
    fl(x) = trialfunction1d(x; prms...)
    data = fl.(x)
    do_mapping(data)
end

function test_data_2d(r)::ITensor
    prms = (α=1, β=1.1, γ=3, ω=4)
    fl(x) = trialfunction1d(x; prms...)
    data = zeros(Float64, length(r), length(r))

    for i in CartesianIndices(data)
        data[i] = fl(r[i[1]] - 0.5) + fl(r[i[2]] + 0.5)
    end

    do_mapping(data)
end

@testset "mpo" begin

    M = 8
    N = 2^M
    χ = 128

    for test_data in [test_data_1d, test_data_2d]
        for alg in ["naive", "densitymatrix"]
            ψa = test_data(range(start=-2, stop=2, length=N))
            ψb = test_data(range(start=-1, stop=3, length=N))
            ψb = replaceinds!(ψb, inds(ψb), inds(ψa))

            ψa = MPS(ψa, inds(ψa); maxdim=χ)
            ψb = MPS(ψb, inds(ψb); maxdim=χ)

            mpo = hadamard_mpo(ψa)
            res = apply(mpo, ψb; maxdim=χ, alg)
            res_hadamard = reverse_mapping(res)
            res_expected = reverse_mapping(ψa) .* reverse_mapping(ψb)

            @test sum((res_hadamard .- res_expected) .^ 2) ≈ 0.0 atol = 1e-15 broken = (alg != "naive")
        end
    end
end
