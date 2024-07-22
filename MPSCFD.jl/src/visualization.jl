
function _extend_field(u_bar, Nx)

    extended = zeros(Nx, length(u_bar))

    for x in 1:Nx
        extended[:, x] .= u_bar
    end

    extended
end

function compute_u_bar(u)
    Nx, Ny = size(u)

    u_bar = map(1:Ny) do i_y
        # since we want to compute 1/L * int_0^L we can set dx := 1 instead
        # this makes the integral u_bar(y) = 1/L * int_0^L u(x,y) dx = ∑_x=0^N u(x,y) / N
        sum(u[i_y, 1:Nx]) / Nx
    end

    _extend_field(u_bar, Nx)
end

function compute_reynolds_stress_tensor(ux, uy)

    ux_bar = MPSCFD.compute_u_bar(ux)
    uy_bar = MPSCFD.compute_u_bar(uy)

    ux_prime = ux .- ux_bar
    uy_prime = uy .- uy_bar

    u_prime = [ux_prime, uy_prime]

    τ_component(i, j) = Vector(MPSCFD.compute_u_bar(u_prime[i] .* u_prime[j])[:, 1])

    τ = Matrix{Vector}(undef, 2, 2)

    for i in CartesianIndices(τ)
        τ[i] = τ_component(i[1], i[2])
    end

    τ
end


function unmapped_derivative(x, dim::Int64, coefficients::FiniteDifferenceCoefficients)
    dx = zeros(size(x)...)
    for i in CartesianIndices(x)
        δx = 1 / size(x)[dim]
        dx[i] = sum(1:length(coefficients.coeff)) do j
            k = j - coefficients.center
            I = CartesianIndex(map(1:ndims(x)) do l
                if l == dim
                    return (i[l] + k + size(x)[l] - 1) % size(x)[l] + 1
                else
                    return i[l]
                end
            end...)
            coefficients.coeff[j] * x[I]
        end / δx
    end
    dx
end

function compute_vorticity(ux, uy; coefficients::FiniteDifferenceCoefficients=central_difference_coefficients(1, 8))
    # note, dimension one is y, dimension 2 is x

    unmapped_derivative(uy, 2, coefficients) - unmapped_derivative(ux, 1, coefficients)
end

function compute_vorticity_mps(ux::MPS, uy::MPS, dx::MPO, dy::MPO)::MPS
    apply(dx, uy) - apply(dy, ux)
end

function heat_map(data; kwargs...)
    m = max(-minimum(data), abs(maximum(data)))

    xmax, ymax = size(data)

    heatmap(data;
        aspect_ratio=1,
        color=:balance,
        clims=(-m, m),
        xlims=(0.5, xmax + 0.5),
        ylims=(0.2ymax, 0.8ymax),
        kwargs...
    )
end

function energy_cascade(x, y)
    Ṽx = norm.(fftshift(fft(x)) / prod(size(x)))
    Ṽy = norm.(fftshift(fft(y)) / prod(size(x)))

    Ẽx = Ṽx .* Ṽx
    Ẽy = Ṽy .* Ṽy

    N = size(x)[1]
    N_half = N ÷ 2

    box_radius = ceil(Int64, sqrt(N^2 + N^2) / 2.0) + 1

    Ek_x = zeros(Float64, box_radius) .+ eps(Float64)
    Ek_y = zeros(Float64, box_radius) .+ eps(Float64)

    for i in 0:N-1
        for j in 0:N-1
            wn = 1 + round(Int64, sqrt((i - N_half)^2 + (j - N_half)^2))

            Ek_x[wn] += Ẽx[1+i, 1+j]
            Ek_y[wn] += Ẽy[1+i, 1+j]
        end
    end

    (Ek_x .+ Ek_y) / 2.0
end

function color_map(; width=0.1, N=200)

    b = 0.50 - width / 2
    a = b * 5 / 9
    gradients = [
        ((0, a), ["#0ff", "#00f"]),
        ((a, b), ["#00f", "#40407f"]),
        ((1 - b, 1 - a), ["#7f4040", "#f00"]),
        ((1 - a, 1), ["#f00", "#ff0"]),
    ]

    colors = map(range(0, 1; length=N)) do v
        i = findfirst(gradients) do ((start, stop), _)
            v >= start && v <= stop
        end

        if isnothing(i)
            return "#fff"
        end

        start, stop = gradients[i][1]
        f = max(v - start, 0) / (stop - start)
        cgrad(gradients[i][2], [0.0, 1.0])[f]
    end

    cgrad(colors, range(0, 1; length=N + 1); categorical=true)
end