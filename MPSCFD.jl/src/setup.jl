struct JetParams
    u_0
    h
    y_min
    y_max
    L_box
end

J(params::JetParams, y::Real) = params.u_0 / 2.0 * (
    tanh((y - params.y_min) / params.h) -
    tanh((y - params.y_max) / params.h) -
    1.0
)

d_1(params::JetParams, x::Real, y::Real) =
    2.0 * params.L_box / params.h^2 * (
        (y - params.y_max) * exp(-(y - params.y_max)^2 / params.h^2) +
        (y - params.y_min) * exp(-(y - params.y_min)^2 / params.h^2)
    ) * (
        sin(8π * x / params.L_box) +
        sin(24π * x / params.L_box) +
        sin(6π * x / params.L_box)
    )


d_2(params::JetParams, x::Real, y::Real) =
    π * (
        exp(-(y - params.y_max)^2 / params.h^2) +
        exp(-(y - params.y_min)^2 / params.h^2)
    ) * (
        8 * cos(8π * x / params.L_box) +
        24 * cos(24π * x / params.L_box) +
        6 * cos(6π * x / params.L_box)
    )

A(params::JetParams, xs, ys) = maximum(Iterators.map(Iterators.product(xs, ys)) do (x, y)
    sqrt(d_1(params, x, y)^2 + d_2(params, x, y)^2)
end)

function jet(params::JetParams, xs, ys)

    δ = params.u_0 / (40.0 * A(params, xs, ys))

    vx = zeros(Float64, length(xs), length(ys))
    vy = zeros(Float64, length(xs), length(ys))

    for index in CartesianIndices(vx)
        x = xs[index[2]]
        y = ys[index[1]]

        vx[index] = δ * d_1(params, x, y) + J(params, y)
        vy[index] = δ * d_2(params, x, y)
    end

    vx, vy
end
