struct FiniteDifferenceCoefficients
    center::Int64
    coeff::Vector{Float64}
end

function finite_difference_coefficients_to_opsum(dim::Int64, coeffs::FiniteDifferenceCoefficients)::LadderOperatorSum
    @assert dim > 0

    Sp = RaisingOperator(dim)
    Sm = LoweringOperator(dim)

    P(x::Int64) = (x < 0 ? Sm : Sp)^abs(x)

    res = 0

    for i in 1:length(coeffs.coeff)
        if !iszero(coeffs.coeff[i])
            res += coeffs.coeff[i] * P(i - coeffs.center)
        end
    end

    res
end

# stencil used to compute finite difference coefficients with 2p + 1 coefficients
function central_difference_stencil(p::Int64)::Matrix{BigInt}

    n_coeff = 2p + 1

    A = zeros(BigInt, n_coeff, n_coeff)

    # create stencil matrix
    for i in CartesianIndices(A)
        a = -p + i[2] - 1
        b = i[1] - 1

        A[i] = b == 0 ? 1 : a^b
    end

    A
end


#_cached_coefficients = Dict{Tuple{Symbol,Int64,Int64},FiniteDifferenceCoefficients}()

#function central_difference_coefficients(nth::Int64, order::Int64)::FiniteDifferenceCoefficients
#key = (:central, nth, order)
#if haskey(_cached_coefficients, key)
#_cached_coefficients[key]
#else
#_cached_coefficients[key] = compute_central_difference_coefficients(nth, order)
#end
#end


# compute finite difference coefficients from stencil
function central_difference_coefficients(nth::Int64, order::Int64)::FiniteDifferenceCoefficients
    @assert nth >= 0
    @assert iseven(order) "central differences always have even order"

    n_coeff = 2 * ((nth + 1) ÷ 2) - 1 + order
    p = Int64((n_coeff - 1) / 2)

    # create stencil matrix
    A = Rational{BigInt}.(central_difference_stencil(p))

    # solve stencil matrix for coefficients for nth order
    b = zeros(Rational{BigInt}, n_coeff)
    b[nth+1] = factorial(big(nth)) // 1

    coeff = A \ b

    @assert A * coeff == b

    FiniteDifferenceCoefficients(p + 1, coeff)
end
