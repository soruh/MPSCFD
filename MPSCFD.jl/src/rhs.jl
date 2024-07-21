# a term of the right-hand-side to be partially contracted with the trial solution and scaled by a factor
# to yield a summand of the full right-hand-side tensor β
struct RhsTerm
    factor::Float64
    ket::MPS
    P::CachedPartialContraction
end

struct RhsTerms
    terms::Vector{RhsTerm}
end

function RhsTerm(factor::Float64, O::MPO, ket::MPS)::RhsTerm
    RhsTerm(factor, ket, CachedPartialContraction(O))
end

function RhsTerm(factor::Float64, ket::MPS)::RhsTerm
    s = siteinds(ket)
    idmpo = MPO(s, "I")
    RhsTerm(factor, idmpo, ket)
end

# bra must be orthogonalized at b
# partially contract the rhs term `term` with the trial solution term `bra`.
# This is equivalent to contracting the folowing MPS - MPO - MPS product:
#
# V: tensors belonging to trial solution `bra``
# O: tensors belonging to the operator part of term
# K: tensors belonging to the MPS part of term
# 
# 1 2 3   b
#         
# K-K-K-K-K-K-K-K
# | | | | | | | |
# O-O-O-O-O-O-O-O
# | | | | | | | |
# V-V-V-V- -V-V-V
#
# the resulting Tensor is then returned, scaled by term.factor
function compute_β(term::RhsTerm, bra::MPS, b::Int64)::ITensor
    orthogonalize!(term.ket, b)

    position!(term.P, bra, term.ket, b)

    term.factor * product(term.P, term.ket[b])
end

compute_β(rhs::RhsTerms, bra::MPS, b::Int64)::ITensor = noprime(sum(Iterators.map(term -> compute_β(term, bra, b), rhs.terms)))

function compute_β_mps(term::RhsTerm)::MPS
    term.factor * apply(term.P.H, term.ket)
end

compute_β_mps(rhs::RhsTerms)::MPS = noprime(sum(Iterators.map(term -> compute_β_mps(term), rhs.terms)))
