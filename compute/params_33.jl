M = 10

χ = 33
χ_rhs = χ
χ_mpo = 128 # constrained through cutoff_mpo

cutoff = 0
cutoff_rhs = 0
cutoff_mpo = 1e-13

load_setup = true
precompile = false

Re = 2e5
T = 2.0
L = 1.0
u_0 = 1.0
ϵ = 1e-8

Δt = 0.1 * 2.0^-(M - 1)
Δx = L * 2.0^-M
μ = Δx^2 * 10^4
h = 1 / 200

order = 8
jet_params = MPSCFD.JetParams(u_0, h, 0.4, 0.6, L)
ν = u_0 * L / Re

linsolve_params = (
    maxiter=10,
    krylovdim=30,
)

params = MPSCFD.Parameters(
    χ, χ_mpo, χ_rhs,
    cutoff, cutoff_mpo, cutoff_rhs,
    Δt, T, μ, ν, L, order, linsolve_params
)

filename = "res_$χ.h5"