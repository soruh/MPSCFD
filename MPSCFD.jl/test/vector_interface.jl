import VectorInterface

using MPSCFD: SiteTrialSolution
using ITensors: random_itensor, Index, inds, combiner, ITensor, vector, norm

using Test

struct Method
    f::Function
    inplace::Union{Nothing,Int64}
    params::Vector{Type}
end

function site_trial_solution_as_vector(trsl::SiteTrialSolution, comb::ITensor)::Vector{Float64}
    vcat(
        vector(trsl.x * comb),
        vector(trsl.y * comb)
    )
end


function random_site_trial_solution(indices)::SiteTrialSolution
    SiteTrialSolution(random_itensor(indices), random_itensor(indices))
end

function test_method(comb::ITensor, indices, m::Method)

    params = map(m.params) do t
        if t == ITensor
            random_site_trial_solution(indices)
        elseif t == Float64
            10.0 * rand()
        else
            error("unexpected parameter type $t")
        end
    end

    test_method_params(m, comb, params...)
end

function test_method_params(m::Method, comb::ITensor, params...)

    orig_params = deepcopy.(params)

    reference_params = []

    for param in params
        if param isa SiteTrialSolution
            push!(reference_params, site_trial_solution_as_vector(param, comb))
        else
            push!(reference_params, deepcopy(param))
        end
    end

    reference = m.f(reference_params...)

    res = m.f(params...)

    for (i, param) in enumerate(params)
        if !isnothing(m.inplace) && m.inplace == i
            @test param != orig_params[i] || "$(m.f) overwrites inplace parameter $i"
            @test param == res || "$(m.f) inplace parameter $i is set to the result"
        else
            δ = if param isa SiteTrialSolution
                norm([
                    norm(param.x - orig_params[i].x),
                    norm(param.y - orig_params[i].y)
                ])
            else
                norm(param - orig_params[i])
            end

            iseq = if param isa SiteTrialSolution
                param.x == orig_params[i].x && param.y == orig_params[i].y
            else
                param == orig_params[i]
            end
            @test iseq || "$(m.f) param $i should not be modified δ=$δ"
        end
    end

    if res isa SiteTrialSolution
        δ = norm(site_trial_solution_as_vector(res, comb) - reference)
        @test δ < 1e-12 || "$(m.f) returns the correct result δ=$δ"
    else
        δ = norm(res - reference)
        @test δ < 1e-12 || "$(m.f) returns the correct result δ=$δ"
    end
end

@testset "vector_interface" begin

    indices = [Index(16, "link=1"), Index(16, "link=2"), Index(4, "Site,2")]
    comb = combiner(indices...)

    test_method(comb, indices, Method(VectorInterface.scale!!, 1, [ITensor, Float64]))
    test_method(comb, indices, Method(VectorInterface.scale!, 1, [ITensor, Float64]))
    test_method(comb, indices, Method(VectorInterface.scale, nothing, [ITensor, Float64]))

    test_method(comb, indices, Method(VectorInterface.scale!!, 1, [ITensor, ITensor, Float64]))
    test_method(comb, indices, Method(VectorInterface.scale!, 1, [ITensor, ITensor, Float64]))

    test_method(comb, indices, Method(VectorInterface.add!!, 1, [ITensor, ITensor, Float64, Float64]))
    test_method(comb, indices, Method(VectorInterface.add!, 1, [ITensor, ITensor, Float64, Float64]))
    test_method(comb, indices, Method(VectorInterface.add, nothing, [ITensor, ITensor, Float64, Float64]))

    test_method(comb, indices, Method(VectorInterface.zerovector!!, 1, [ITensor]))
    test_method(comb, indices, Method(VectorInterface.zerovector!, 1, [ITensor]))
    test_method(comb, indices, Method(VectorInterface.zerovector, nothing, [ITensor]))

    test_method(comb, indices, Method(VectorInterface.norm, nothing, [ITensor]))
    test_method(comb, indices, Method(VectorInterface.inner, nothing, [ITensor, ITensor]))
end
