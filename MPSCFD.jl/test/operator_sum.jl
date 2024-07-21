using MPSCFD: RaisingOperator, LoweringOperator, adjoint, LadderOperatorSum

using Test

@testset "operator_sum" begin
    dim = 1
    
    Sp = RaisingOperator(dim)
    @test Sp.dimension === UInt64(dim)
    @test Sp.power === UInt64(1)
    @test Sp.is_adjoint === false

    Sm = LoweringOperator(dim)
    @test Sm.is_adjoint == !Sp.is_adjoint
    @test Sm.power == Sp.power
    @test Sm.dimension === Sp.dimension

    Sm′ = adjoint(Sp)

    @test Sm′.is_adjoint === !Sp.is_adjoint
    @test Sm′.power === Sp.power
    @test Sm′.dimension === Sp.dimension
    

    for p in 1:10
        Sppwr = Sp^p
        @test Sppwr.dimension === Sp.dimension
        @test Sppwr.power === Sp.power*p
        @test Sppwr.is_adjoint === false
    end
    id = Sp^0
    @test id.dimension === UInt64(0)
    @test id.power === UInt64(0)
    @test id.is_adjoint === false

    ops = 1*Sp
    @test ops isa LadderOperatorSum
    @test ops[Sp] == 1

    ops = ops + Sp
    
    @test length(keys(ops.terms)) === 1
    @test ops[Sp] == 2

    ops = ops + 2.3*Sm
    @test length(keys(ops.terms)) === 2
    @test ops[Sm] == 2.3

    ops = ops + ops
    @test length(keys(ops.terms)) === 2
    @test ops[Sm] == 2 * 2.3
    @test ops[Sp] == 4

    ops = ops * 3

    @test length(keys(ops.terms)) === 2
    @test ops[Sm] == 2 * 3 * 2.3
    @test ops[Sp] == 4 * 3

    ops2 = 5 * Sp^2 + 4 * Sm^2

    ops = ops + ops2

    @test length(keys(ops.terms)) === 4
    @test ops[Sm] == 2 * 3 * 2.3
    @test ops[Sp] == 4 * 3
    @test ops[Sm^2] == 4
    @test ops[Sp^2] == 5


    ops3 = 0
    ops3 += Sp
    @test ops3[Sp] == 1
    @test ops3[Sp^0] == 0
    @test length(ops3.terms) === 2
    
    ops4 = 0
    ops4 += 1.0*Sp
    @test ops3[Sp] == 1.0
    @test ops3[Sp^0] == 0
    @test length(ops3.terms) === 2
    



end
