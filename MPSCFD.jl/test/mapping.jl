using MPSCFD: binary_representation, do_mapping, do_mapping_mps, reverse_mapping, get_params
using ITensors: MPO, siteinds, apply, norm
using Test

function check_binary(v::Int64, n::Int64)
    BitVector(binary_representation(v, n)) == BitVector(map(c -> c != '0', collect(bitstring(v)[end-n+1:end])))
end

function mapping_is_bijective(dim, M)
    N = 2^M
    data = reshape(1:(N^dim), (N for _ in 1:dim)...)
    reverse_mapping(do_mapping(data)) == data
end


function test_apply_identity(N::Int64, alg::String)

    x = range(start=-2, stop=2, length=2^N)

    prms = (α=1, β=1.1, γ=3, ω=4)
    fl(x) = trialfunction1d(x; prms...)

    y = fl.(x)
    ψf = do_mapping_mps(y)

    params = get_params(ψf)

    @assert params.N == N
    @assert params.dim == 1

    nrmψ = norm(ψf)
    ψf /= nrmψ

    Id = MPO(siteinds(ψf), "I")

    ψf2 = apply(Id, ψf; alg=alg, cutoff=0)
    f2 = reverse_mapping(ψf2) * nrmψ

    norm(f2 .- y)
end

@testset "mapping" begin

    for n in 1:8
        @test all(v -> check_binary(v, n), 0:2^n-1)
    end

    @test mapping_is_bijective(1, 5)
    @test mapping_is_bijective(1, 10)
    @test mapping_is_bijective(2, 5)
    @test mapping_is_bijective(3, 5)
    @test mapping_is_bijective(4, 4)

    @test test_apply_identity(8, "naive") ≈ 0.0 atol = 1e-13
    @test test_apply_identity(8, "densitymatrix") ≈ 0.0 atol = 1e-13 broken = true
end
