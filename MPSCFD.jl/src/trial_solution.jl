
mutable struct SiteTrialSolution
    x::ITensor
    y::ITensor
end


import VectorInterface

# define VectorInterface methods to be able to treat SiteTrialSolution as a vector containing all elements of each tensor
# for use in KrylovKit linsolve

VectorInterface.inner(a::SiteTrialSolution, b::SiteTrialSolution)::Real = scalar(a.x * b.x + a.y * b.y)
function VectorInterface.norm(a::SiteTrialSolution, p::Real=2)::Real
    @assert p == 2
    sqrt(VectorInterface.inner(a, a))
end


VectorInterface.scale(a::SiteTrialSolution, α::Real)::SiteTrialSolution = SiteTrialSolution(α * a.x, α * a.y)
function VectorInterface.scale!(a::SiteTrialSolution, α::Real)::SiteTrialSolution
    a.x .*= α
    a.y .*= α
    a
end
VectorInterface.scale!!(a::SiteTrialSolution, α::Real)::SiteTrialSolution = VectorInterface.scale!(a, α)

function VectorInterface.scale!(a::SiteTrialSolution, b::SiteTrialSolution, α::Real)::SiteTrialSolution
    a.x .= α * b.x
    a.y .= α * b.y
    a
end
VectorInterface.scale!!(a::SiteTrialSolution, b::SiteTrialSolution, α::Real)::SiteTrialSolution = VectorInterface.scale!(a, b, α)

VectorInterface.scalartype(a::SiteTrialSolution) = eltype(a.x)

function VectorInterface.add(a::SiteTrialSolution, b::SiteTrialSolution, α::Real=1.0, β::Real=1.0)::SiteTrialSolution
    x = (β * a.x) + (α * b.x)
    y = (β * a.y) + (α * b.y)
    SiteTrialSolution(x, y)
end

function tensor_add!(a::ITensor, b::ITensor, α::Real, β::Real)::ITensor
    if typeof(ITensors.data(a)) == NDTensors.NoData || typeof(ITensors.data(b)) == NDTensors.NoData
        if typeof(ITensors.data(a)) == NDTensors.NoData
            a .= α .* b
        else
            ITensors.data(a) .= (β .* ITensors.data(a))
        end
    else
        ITensors.data(a) .= (β .* ITensors.data(a)) .+ (α .* ITensors.data(b))
    end

    a
end

function VectorInterface.add!(a::SiteTrialSolution, b::SiteTrialSolution, α::Real=1.0, β::Real=1.0)::SiteTrialSolution
    @assert inds(a.x) == inds(b.x) && inds(a.y) == inds(b.y)

    tensor_add!(a.x, b.x, α, β)
    tensor_add!(a.y, b.y, α, β)
    a
end
VectorInterface.add!!(a::SiteTrialSolution, b::SiteTrialSolution, α::Real=1.0, β::Real=1.0)::SiteTrialSolution = VectorInterface.add!(a, b, α, β)
function VectorInterface.zerovector(a::SiteTrialSolution, ::Type{Float64})::SiteTrialSolution
    VectorInterface.zerovector!(SiteTrialSolution(deepcopy(a.x), deepcopy(a.y)))
end
function VectorInterface.zerovector!(a::SiteTrialSolution)::SiteTrialSolution
    a.x *= 0.0
    a.y *= 0.0
    a
end
VectorInterface.zerovector!!(a::SiteTrialSolution)::SiteTrialSolution = VectorInterface.zerovector!(a)
