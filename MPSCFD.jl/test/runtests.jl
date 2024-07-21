using Test

println("Running Tests....")

function trialfunction1d(x; α, β, γ, ω)
    sin(ω * x) * α * exp(-β * x) + tanh(x) / γ
end

# function d_trialfunction1d(x; α, β, γ, ω)
#     -α * β * exp(-β * x) * sin(ω * x) + α * ω * exp(-β * x) * cos(ω * x) + (4 * cosh(x)^2) / (γ * (cosh(2 * x) + 1)^2)
# end

@testset "MPSCFD.jl" begin
    @testset "$filename" for filename in [
        "ladder_operator.jl",
        "vector_interface.jl",
        "cached_contrations.jl",
        "derivatives.jl",
        "mpo.jl",
        "operator_sum.jl",
        "mapping.jl",
        "finite_differences.jl",
    ]
        println("Running $filename")
        include(filename)
    end
end
