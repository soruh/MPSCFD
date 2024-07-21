### A Pluto.jl notebook ###
# v0.19.43

using Markdown
using InteractiveUtils

# ╔═╡ 9b39354d-7a54-4a21-9549-358e0f3ca518
let
    import Pkg
    Pkg.develop(url="../lib_new/MPSCFD.jl")
end

# ╔═╡ 1118a27d-fb7d-45c8-a7e3-866663c1e6e6
using ITensors

# ╔═╡ 47ce2bf4-9590-4400-ba1c-57e15a889420
using Plots

# ╔═╡ 5bf13147-9e30-4c9d-9d90-8ea5e4f86b13
using MAT

# ╔═╡ 84e456e5-e857-476c-9eb0-25f1a1884b40
using HDF5

# ╔═╡ 1fc396d9-2991-45c0-9cc6-e748dc209355
using Statistics

# ╔═╡ 2cd9a79c-fbf3-45bc-a0f7-a907fdf3527f
using Measurements

# ╔═╡ 39e73690-4c53-4669-8f7f-1c7aaf4e893b
using KrylovKit

# ╔═╡ 57d108d4-90e0-4b7e-aaa0-a0589c2c87f2
using ProgressLogging

# ╔═╡ ecb7e578-d0d0-41bc-8d84-7265362608f9
using VectorInterface

# ╔═╡ 83d6bb22-9e6a-4311-89b0-7c93006a7d1d
using MPSCFD

# ╔═╡ 1cea98af-1da7-4613-8c06-812b8a9186a5
using FFTW

# ╔═╡ c6a9021b-aa6a-4095-88ec-d1c4418d0240
# ╠═╡ disabled = true
#=╠═╡
last, Δ = h5open("res4.h5", "r") do file_a
	h5open("res5.h5", "r") do file_b

		t_max = max_t(keys(file_a))

		res = []

		last = nothing

		@progress for it in 0:t_max
			a = read_it(file_a, it)
			b = read_it(file_b, it)

			r = MPSCFD.reverse_mapping.(a), MPSCFD.reverse_mapping.(b)

			Δ = r[1] .- r[2]
			last = Δ, r

			mx = maximum(abs.(Δ[1]))
			my = maximum(abs.(Δ[2]))

			push!(res, (mx, my))
		end

		last, res
	end
end
  ╠═╡ =#

# ╔═╡ da99a110-40cc-49b8-84f7-7ee8fae9c806
#=╠═╡
hm(last[1][1])
  ╠═╡ =#

# ╔═╡ 36ef217d-d9a3-4b7b-a4d8-9dc5b0111020
#=╠═╡
hm(last[1][2])
  ╠═╡ =#

# ╔═╡ dec7576a-2b26-49fb-8829-3b35c06276ec
# ╠═╡ disabled = true
#=╠═╡
coefficients_1_8 = MPSCFD.central_difference_coefficients(1, 8)
  ╠═╡ =#

# ╔═╡ 150a7653-acd7-426b-a881-508d38f7d53e
filename = "res_74.h5"

# ╔═╡ 7deb1a03-ea4a-4286-9510-1c1b9e425858
max_t(keys) = maximum(map(x -> parse(Int64, x[1]), filter(x -> !isnothing(x), map(x -> match(r"it=(\d+),x", x), keys))))

# ╔═╡ 7d44fb2d-6aad-49c0-b34f-94d1f1fd7892
function read_it(file, it)

    ux = read(file, "it=$it,x", MPS)
    uy = read(file, "it=$it,y", MPS)

    return ux, uy
end

# ╔═╡ f3fb5773-3426-4309-9e87-cfc982ba67cc
history, Ekin = begin
    Ekin = []
    history = []

    h5open(filename, "r") do file
        t_max = max_t(keys(file))
        @info t_max
        @progress for it in 0:128:t_max
            ux, uy = read_it(file, it)

            sites = if isempty(history)
                siteinds(ux)
            else
                siteinds(history[1][1])
            end

            replace_siteinds!(ux, sites)
            replace_siteinds!(uy, sites)

            push!(Ekin, ITensors.inner(ux, ux) + ITensors.inner(uy, uy))
            push!(history, (ux, uy))
        end
    end

    history, Ekin
end;

# ╔═╡ 37ae0686-d0a8-41e6-a839-606d2eb4b9d6
τs = map(1:length(history)) do i
    MPSCFD.compute_reynolds_stress_tensor(
        MPSCFD.reverse_mapping(history[i][1]),
        MPSCFD.reverse_mapping(history[i][2])
    )
end

# ╔═╡ c0d9ff6e-aa26-476b-80bc-cd991d57b9a4
τ = hcat(Iterators.map(T -> T[1, 2], τs)...);

# ╔═╡ fe6a25c9-85a6-4658-a86e-8aced48af9c3
function plot_Ek(Ek)
    plot(1:length(Ek), Ek; xscale=:log10, yscale=:log10)
end

# ╔═╡ 129d8670-4059-4103-bce3-e36419353c91
function plot_Ek!(Ek)
    plot!(1:length(Ek), Ek; xscale=:log10, yscale=:log10)
end

# ╔═╡ 28be7053-64c5-416a-a85e-36e501fa0b2a
let
    plot_Ek(MPSCFD.energy_cascade(
        MPSCFD.reverse_mapping(history[end][1]),
        MPSCFD.reverse_mapping(history[end][2])
    ))
end

# ╔═╡ 5e91465d-ce2c-4130-90b3-44dbe25825b8
if length(history) > 1
    plot_Ek(abs.(MPSCFD.energy_cascade(
        MPSCFD.reverse_mapping(history[end][1]),
        MPSCFD.reverse_mapping(history[end][2])
    ) - MPSCFD.energy_cascade(
        MPSCFD.reverse_mapping(history[end-1][1]),
        MPSCFD.reverse_mapping(history[end-1][2])
    )) .+ eps())
end

# ╔═╡ 952db7c2-29ac-4fdd-a30d-bc93bc5b3eca
let
    plot()
    if length(history) > 1
        plot_Ek!(MPSCFD.energy_cascade(
            MPSCFD.reverse_mapping(history[end-1][1]),
            MPSCFD.reverse_mapping(history[end-1][2])
        ))
    end
    plot_Ek!(MPSCFD.energy_cascade(
        MPSCFD.reverse_mapping(history[end][1]),
        MPSCFD.reverse_mapping(history[end][2])
    ))
end

# ╔═╡ 0529a1b9-435b-4ceb-a5d5-c4305187a436
let
    δx = 1.0 * 2.0^(-length(history[1][1]))
    order = 8

    dx = MPSCFD.central_differences(history[1][1]; dim=2, nth=1, order=order, dx=δx)
    dy = MPSCFD.central_differences(history[1][2]; dim=1, nth=1, order=order, dx=δx)

    plot(map(1:length(history)) do i
        ux = history[i][1]
        uy = history[i][2]

        replace_siteinds!(ux, siteinds(history[1][1]))
        replace_siteinds!(uy, siteinds(history[1][2]))

        differences = MPSCFD.reverse_mapping(apply(dx, ux)) .+ MPSCFD.reverse_mapping(apply(dy, uy))

        mean(differences) ± std(differences)
    end)
end

# ╔═╡ a1a90f4b-88c1-470c-972a-70a629b2a67e
plot(Ekin)

# ╔═╡ 503a36f0-ea9e-46d5-ad58-ad8bea0331e9
# ╠═╡ disabled = true
#=╠═╡
map(i -> (eltype(newest[1][i]), eltype(newest[2][i])), 1:8)
  ╠═╡ =#

# ╔═╡ 4c2f6c47-1869-475b-a1a6-024af709d4cf
(; N, dim) = MPSCFD.get_params(history[end][1])

# ╔═╡ 226ef868-0d98-4698-940d-f79a847bd474
# hm(args...) = heatmap(args...; xlims=(0.0, 2^N+0.5), ylims=(0.0, 2^N+0.5), aspect_ratio=1, color=:balance)
hm = MPSCFD.heat_map

# ╔═╡ 5cfd7e97-1bb8-433a-82d6-a3ea4e76e2c2
function hm2(data)
    m = max(-minimum(data), abs(maximum(data)))

    xmax, ymax = size(data)

    heatmap(data;
        color=:balance,
        clims=(-m, m)
    )
end

# ╔═╡ 1e4dba95-dec7-45af-9ccf-9eda48c2ce33
# MPSCFD.heat_map(τ)
hm2(τ)

# ╔═╡ fc0256a0-da81-423c-b40a-adb169d6b07d
hm(MPSCFD.reverse_mapping(history[end][1]))

# ╔═╡ 6c129bd2-e151-4685-bc4c-82f18d3bce35
hm(MPSCFD.reverse_mapping(history[end][2]))

# ╔═╡ 7c602429-43a4-4f62-ae2b-49f6b77bb3d8
#=╠═╡
hm(MPSCFD.compute_vorticity(MPSCFD.reverse_mapping.(history[end])...; coefficients=coefficients_1_8))
  ╠═╡ =#

# ╔═╡ a1e90529-4f5a-4f00-b372-5f7519b03727
ω = let
    δx = 1.0 * 2.0^(-length(history[1][1]))
    order = 8

    dx = MPSCFD.central_differences(history[1][1]; dim=2, nth=1, order=order, dx=δx)
    dy = MPSCFD.central_differences(history[1][2]; dim=1, nth=1, order=order, dx=δx)

    map(1:length(history)) do i
        MPSCFD.compute_vorticity_mps(
            history[i][1], history[i][2],
            dx, dy
        )
    end
end

# ╔═╡ 5e8a3172-afd0-4a7d-85d3-94164509f10c
let
    m = maximum(Iterators.map(1:length(history)) do i
        f = MPSCFD.reverse_mapping(ω[i])
        max(abs(-minimum(f)), abs(maximum(f)))
    end)

    anim = @animate for i in 1:length(history)
        heatmap(MPSCFD.reverse_mapping(ω[i]),
            color=:balance,
            clims=(-m, m),
        )
    end

    gif(anim)
end

# ╔═╡ 89b86bfe-0428-4351-a549-963ae625aff0
plot(τ[:, end])

# ╔═╡ e89aeecc-8158-48f8-921e-ba7538c094e4
let
    τ_min, τ_max = minimum(τ), maximum(τ)
    anim = @animate for i in 1:size(τ)[2]
        plot(τ[:, i]; ylims=(τ_min, τ_max))
    end
    gif(anim)
end

# ╔═╡ 0c3ecd40-e1fc-4dbb-9d2e-560abbeb2e85
function animate_u(history, d::Int64)
    fields = map(i -> MPSCFD.reverse_mapping(history[i][d]), 1:length(history))

    m = maximum(Iterators.map(1:length(history)) do i
        max(-minimum(fields[i]), abs(maximum(fields[i])))
    end)

    @animate for i in 1:length(history)
        heatmap(fields[i],
            color=:balance,
            clims=(-m, m),
        )
    end
end

# ╔═╡ 2ea6dd3e-70b6-463d-bedc-42c43f0dd23e
gif(animate_u(history, 1))

# ╔═╡ 9ba08f84-d6cf-4024-be9a-7f9e81aa8fff
gif(animate_u(history, 2))

# ╔═╡ 404cc727-c66d-4b65-b948-4675d5fe4fdd
let
    anim = @animate for i in 1:length(history)
        x = MPSCFD.reverse_mapping(history[i][1])
        y = MPSCFD.reverse_mapping(history[i][2])


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

        Ek = (Ek_x .+ Ek_y) / 2.0

        # , ylims=(1e-6, 1e2)
        plot(1:length(Ek), Ek; xscale=:log10, yscale=:log10)
    end

    gif(anim)
end

# ╔═╡ Cell order:
# ╠═1118a27d-fb7d-45c8-a7e3-866663c1e6e6
# ╠═47ce2bf4-9590-4400-ba1c-57e15a889420
# ╠═5bf13147-9e30-4c9d-9d90-8ea5e4f86b13
# ╠═84e456e5-e857-476c-9eb0-25f1a1884b40
# ╠═1fc396d9-2991-45c0-9cc6-e748dc209355
# ╠═2cd9a79c-fbf3-45bc-a0f7-a907fdf3527f
# ╠═39e73690-4c53-4669-8f7f-1c7aaf4e893b
# ╠═57d108d4-90e0-4b7e-aaa0-a0589c2c87f2
# ╠═ecb7e578-d0d0-41bc-8d84-7265362608f9
# ╠═9b39354d-7a54-4a21-9549-358e0f3ca518
# ╠═83d6bb22-9e6a-4311-89b0-7c93006a7d1d
# ╠═c6a9021b-aa6a-4095-88ec-d1c4418d0240
# ╠═da99a110-40cc-49b8-84f7-7ee8fae9c806
# ╠═36ef217d-d9a3-4b7b-a4d8-9dc5b0111020
# ╠═dec7576a-2b26-49fb-8829-3b35c06276ec
# ╠═150a7653-acd7-426b-a881-508d38f7d53e
# ╠═7deb1a03-ea4a-4286-9510-1c1b9e425858
# ╠═7d44fb2d-6aad-49c0-b34f-94d1f1fd7892
# ╠═f3fb5773-3426-4309-9e87-cfc982ba67cc
# ╠═37ae0686-d0a8-41e6-a839-606d2eb4b9d6
# ╠═c0d9ff6e-aa26-476b-80bc-cd991d57b9a4
# ╠═1e4dba95-dec7-45af-9ccf-9eda48c2ce33
# ╠═fe6a25c9-85a6-4658-a86e-8aced48af9c3
# ╠═129d8670-4059-4103-bce3-e36419353c91
# ╠═28be7053-64c5-416a-a85e-36e501fa0b2a
# ╠═5e91465d-ce2c-4130-90b3-44dbe25825b8
# ╠═952db7c2-29ac-4fdd-a30d-bc93bc5b3eca
# ╠═0529a1b9-435b-4ceb-a5d5-c4305187a436
# ╠═a1a90f4b-88c1-470c-972a-70a629b2a67e
# ╠═503a36f0-ea9e-46d5-ad58-ad8bea0331e9
# ╠═4c2f6c47-1869-475b-a1a6-024af709d4cf
# ╠═226ef868-0d98-4698-940d-f79a847bd474
# ╠═5cfd7e97-1bb8-433a-82d6-a3ea4e76e2c2
# ╠═fc0256a0-da81-423c-b40a-adb169d6b07d
# ╠═6c129bd2-e151-4685-bc4c-82f18d3bce35
# ╠═7c602429-43a4-4f62-ae2b-49f6b77bb3d8
# ╠═a1e90529-4f5a-4f00-b372-5f7519b03727
# ╠═5e8a3172-afd0-4a7d-85d3-94164509f10c
# ╠═89b86bfe-0428-4351-a549-963ae625aff0
# ╠═e89aeecc-8158-48f8-921e-ba7538c094e4
# ╠═0c3ecd40-e1fc-4dbb-9d2e-560abbeb2e85
# ╠═2ea6dd3e-70b6-463d-bedc-42c43f0dd23e
# ╠═9ba08f84-d6cf-4024-be9a-7f9e81aa8fff
# ╠═1cea98af-1da7-4613-8c06-812b8a9186a5
# ╠═404cc727-c66d-4b65-b948-4675d5fe4fdd
