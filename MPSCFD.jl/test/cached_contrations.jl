using MPSCFD: CachedPartialContraction, position!, product, do_mapping_mps, get_params
using ITensors: siteinds, orthogonalize!, prime, scalar, random_mps, op, op!, norm, MPO, apply, inner, orthogonalize!, delta, ITensor, contract, noprime

using Test

# TODO: 2D, compare to exact
@testset "cached_contrations" begin

    N = 8

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

    @test norm(ψf) ≈ 1.0 atol = 1e-14

    P = CachedPartialContraction(MPO(siteinds(ψf), "I"))

    for b in 1:N
        orthogonalize!(ψf, b)
        position!(P, ψf, ψf, b)
        n = scalar(product(P, ψf[b]) * prime(ψf[b]))
        @test n ≈ 1.0 atol = 1e-14
    end


    sites = siteinds("S=1/2", N)

    ψA = random_mps(sites; linkdims=64)
    ψB = random_mps(sites; linkdims=64)

    O = MPO(sites, map(1:N) do b
        if iseven(b)
            "Sz"
        else
            "Sx"
        end
    end)


    reference = inner(ψA, apply(O, ψB))

    P2 = CachedPartialContraction(O)

    for b in 1:N
        position!(P2, ψA, ψB, b)
        found = scalar(product(P2, ψB[b]) * prime(ψA[b]))

        @test reference - found ≈ 0 atol = eps(Float64)
    end

    for b in 1:N
        position!(P2, ψA, ψB, b)
        found = noprime(product(P2, ψB[b]))

        ψC = copy(ψA)
        ψC[b] = ITensor(1)

        expected = ITensor(1)
        tmp = apply(O, ψB)
        foreach(1:length(ψC)) do jj
            expected = expected * ψC[jj] * tmp[jj]
        end
        
        @test norm(expected - found) ≈ 0 atol = eps(Float64)
    end

end
