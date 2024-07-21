using MPSCFD: central_difference_coefficients

using Test

@testset "finite_differences" begin
    @test (central_difference_coefficients(1, 2).coeff ≈ [-1 / 2, 0.0, 1 / 2])
    @test (central_difference_coefficients(1, 4).coeff ≈ [1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12])
    @test (central_difference_coefficients(1, 6).coeff ≈ [-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60])
    @test (central_difference_coefficients(1, 8).coeff ≈ [1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105, -1 / 280])

    @test (central_difference_coefficients(2, 2).coeff ≈ [1.0, -2, 1.0])
    @test (central_difference_coefficients(2, 4).coeff ≈ [-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12])
    @test (central_difference_coefficients(2, 6).coeff ≈ [1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90])
    @test (central_difference_coefficients(2, 8).coeff ≈ [-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5, 8 / 315, -1 / 560])

    @test (central_difference_coefficients(3, 2).coeff ≈ [-1 / 2, 1.0, 0.0, -1.0, 1 / 2])
    @test (central_difference_coefficients(3, 4).coeff ≈ [1 / 8, -1.0, 13 / 8, 0.0, -13 / 8, 1.0, -1 / 8])
    @test (central_difference_coefficients(3, 6).coeff ≈ [-7 / 240, 3 / 10, -169 / 120, 61 / 30, 0.0, -61 / 30, 169 / 120, -3 / 10, 7 / 240])

    @test (central_difference_coefficients(4, 2).coeff ≈ [1.0, -4.0, 6.0, -4.0, 1.0])
    @test (central_difference_coefficients(4, 4).coeff ≈ [-1 / 6, 2.0, -13 / 2, 28 / 3, -13 / 2, 2.0, -1 / 6])
    @test (central_difference_coefficients(4, 6).coeff ≈ [7 / 240, -2 / 5, 169 / 60, -122 / 15, 91 / 8, -122 / 15, 169 / 60, -2 / 5, 7 / 240])

    @test (central_difference_coefficients(5, 2).coeff ≈ [-1 / 2, 2.0, -5 / 2, 0.0, 5 / 2, -2.0, 1 / 2])
    @test (central_difference_coefficients(5, 4).coeff ≈ [1 / 6, -3 / 2, 13 / 3, -29 / 6, 0.0, 29 / 6, -13 / 3, 3 / 2, -1 / 6])
    @test (central_difference_coefficients(5, 6).coeff ≈ [-13 / 288, 19 / 36, -87 / 32, 13 / 2, -323 / 48, 0.0, 323 / 48, -13 / 2, 87 / 32, -19 / 36, 13 / 288])

    @test (central_difference_coefficients(6, 2).coeff ≈ [1.0, -6.0, 15.0, -20.0, 15.0, -6.0, 1.0])
    @test (central_difference_coefficients(6, 4).coeff ≈ [-1 / 4, 3.0, -13.0, 29.0, -75 / 2, 29.0, -13, 3.0, -1 / 4])
    @test (central_difference_coefficients(6, 6).coeff ≈ [13 / 240, -19 / 24, 87 / 16, -39 / 2, 323 / 8, -1023 / 20, 323 / 8, -39 / 2, 87 / 16, -19 / 24, 13 / 240])
end
