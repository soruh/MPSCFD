import Base: adjoint, convert, promote_rule, copy, -, +, *, /, ^, getindex

struct LadderOperator
    dimension::UInt64
    power::UInt64
    is_adjoint::Bool
end

LadderOperator(dimension::UInt64; is_adjoint=false) = LadderOperator(dimension, 1, is_adjoint)
LadderOperator(dimension::Int64; is_adjoint=false) = LadderOperator(UInt64(dimension), 1, is_adjoint)

RaisingOperator(dimension::Union{Int64,UInt64}) = LadderOperator(dimension; is_adjoint=false)
LoweringOperator(dimension::Union{Int64,UInt64}) = LadderOperator(dimension; is_adjoint=true)

struct LadderOperatorSum
    terms::Dict{LadderOperator,Float64}
end
LadderOperatorSum(op::LadderOperator) = convert(LadderOperatorSum, op)

getindex(opsm::LadderOperatorSum, op::LadderOperator) = opsm.terms[op]

convert(::Type{LadderOperatorSum}, op::LadderOperator) = LadderOperatorSum(Dict(op => 1))
LinearAlgebra.adjoint(op::LadderOperator)::LadderOperator = LadderOperator(op.dimension, op.power, !op.is_adjoint)

copy(x::LadderOperatorSum)::LadderOperatorSum = LadderOperatorSum(copy(x.terms))

function (^)(op::LadderOperator, pow::Int64)::LadderOperator
    pow == 0 && return LadderOperator(0, 0, false)
    return LadderOperator(op.dimension, op.power * UInt64(pow), op.is_adjoint)
end


(*)(factor::Real, op::LadderOperator)::LadderOperatorSum = factor * LadderOperatorSum(op)
(/)(op::LadderOperator, div::Real)::LadderOperatorSum = 1.0 / div * op

(*)(op::LadderOperator, factor::Real)::LadderOperatorSum = factor * op

(/)(op::LadderOperatorSum, div::Real)::LadderOperatorSum = 1.0 / div * op


(*)(factor::Real, op::LadderOperatorSum)::LadderOperatorSum = LadderOperatorSum(Dict(map(k -> k => op.terms[k] * factor, collect(keys(op.terms)))))
(*)(op::LadderOperatorSum, factor::Real)::LadderOperatorSum = factor * op

(-)(op::LadderOperator)::LadderOperator = -1.0 * op
(-)(op::LadderOperatorSum)::LadderOperatorSum = -1.0 * op

function (+)(lhs::LadderOperatorSum, rhs::LadderOperatorSum)::LadderOperatorSum
    res = copy(lhs)
    for (k, v) in rhs.terms
        if haskey(res.terms, k)
            res.terms[k] += v
        else
            res.terms[k] = v
        end
    end
    res
end

(+)(lhs::LadderOperatorSum, rhs::LadderOperator)::LadderOperatorSum = lhs + 1.0 * rhs
(+)(lhs::LadderOperator, rhs::LadderOperatorSum)::LadderOperatorSum = rhs + lhs
(+)(lhs::LadderOperator, rhs::LadderOperator)::LadderOperatorSum = 1.0 * rhs + lhs
(-)(lhs::LadderOperatorSum, rhs::LadderOperator)::LadderOperatorSum = lhs - 1.0 * rhs
(-)(lhs::LadderOperator, rhs::LadderOperatorSum)::LadderOperatorSum = -rhs + lhs
(-)(lhs::LadderOperator, rhs::LadderOperator)::LadderOperatorSum = -rhs + lhs

(+)(lhs::Real, rhs::LadderOperatorSum)::LadderOperatorSum = lhs * LadderOperator(0, 0, false) + rhs
(+)(lhs::Real, rhs::LadderOperator)::LadderOperatorSum = lhs + 1.0 * rhs
(-)(lhs::Real, rhs::LadderOperatorSum)::LadderOperatorSum = lhs * LadderOperator(0, 0, false) - rhs
(-)(lhs::Real, rhs::LadderOperator)::LadderOperatorSum = lhs - 1.0 * rhs

