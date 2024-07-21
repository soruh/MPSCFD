# Struct for defining general parameters
struct Parameters
    χ::Int64
    χ_mpo::Int64 # MPO bond dimension
    χ_rhs::Int64 # Right Hand side bond dimension

    cutoff::Real
    cutoff_mpo::Real
    cutoff_rhs::Real

    Δt::Float64 # timestep size
    T::Float64 # end time
    μ::Float64 # penalty coefficient
    ν::Float64 # Kinematic Viscosity
    L::Float64 # characteristic length scale
    order::Int64 # order of central differences used
    linsolve::NamedTuple # keyword arguments passed to linsolve
end

half_Δt(params::Parameters)::Parameters = Parameters(
    params.χ, params.χ_mpo, params.χ_rhs,
    params.cutoff, params.cutoff_mpo, params.cutoff_rhs,
    params.Δt / 2.0,
    params.T, params.μ, params.ν, params.L, params.order, params.linsolve
)
