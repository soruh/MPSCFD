function move_mps(mps::MPS, f::Function)::MPS
    MPS(map(i -> f(mps[i]), 1:length(mps)))
end

function move_mpo(mpo::MPO, f::Function)::MPO
    MPO(map(i -> f(mpo[i]), 1:length(mpo)))
end

function move_tensors!(solver_state::SolverState, f::Function)
    move_tensors!(solver_state.ops, f)
    move_tensors!(solver_state.contractions, f)

    solver_state.ux = move_mps(solver_state.ux, f)
    solver_state.uy = move_mps(solver_state.uy, f)

    if !isnothing(solver_state.rhs)
        move_tensors!(solver_state.rhs[1], f)
        move_tensors!(solver_state.rhs[2], f)
    end
end

function move_tensors!(ops::Operators, f::Function)
    ops.dx = move_mpo(ops.dx, f)
    ops.dy = move_mpo(ops.dy, f)
    ops.ddx = move_mpo(ops.ddx, f)
    ops.ddy = move_mpo(ops.ddy, f)
    ops.dxdx = move_mpo(ops.dxdx, f)
    ops.dydy = move_mpo(ops.dydy, f)
    ops.dxdy = move_mpo(ops.dxdy, f)
    ops.dydx = move_mpo(ops.dydx, f)
end

function move_tensors!(contractions::CachedSolverContractions, f::Function)
    move_tensors!(contractions.Pxx, f)
    move_tensors!(contractions.Pxy, f)
    move_tensors!(contractions.Pyx, f)
    move_tensors!(contractions.Pyy, f)
end

function move_tensors!(P::CachedPartialContraction, f::Function)
    P.H = move_mpo(P.H, f)
    for i in 1:length(P.LR)
        if isdefined(P.LR, i)
            P.LR[i] = f(P.LR[i])
        end
    end
end

function move_tensors!(rhs::RhsTerms, f::Function)
    for term in rhs.terms
        move_tensors!(term, f)
    end
end

function move_tensors!(rhs::RhsTerm, f::Function)
    rhs.ket = move_mps(rhs.ket, f)
    move_tensors!(rhs.P, f)
end