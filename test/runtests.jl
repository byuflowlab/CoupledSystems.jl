using CoupledSystems
using Test

@testset "ExplicitComponent" begin

    # This uses the paraboloid example from OpenMDAO

    f! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    df! = function(dydx, x)
        dydx[1,1] = 2*(x[1]-3) + x[2]
        dydx[1,2] = x[1] + 2*(x[2]+4)
        return dydx
    end
    fdf! = function(y, dydx, x)
        f!(y, x)
        df!(dydx, x)
        return y, dydx
    end
    x = zeros(2)
    y = zeros(1)
    dydx = zeros(1,2)
    comp1 = ExplicitComponent(x, y; f=f!, df=df!, fdf=fdf!, dydx=dydx)
    comp2 = ExplicitComponent(x, y; f=f!, df=df!, fdf=fdf!)
    comp3 = ExplicitComponent(x, y; f=f!, df=df!)
    comp4 = ExplicitComponent(x, y; fdf=fdf!)

    for comp in [comp2, comp3, comp4]

        x = rand(2)
        @test outputs(comp1, x) == outputs(comp, x)
        @test outputs!(comp1, y, x) == outputs!(comp, y, x)
        @test outputs!(comp1, x) == outputs!(comp, x)
        @test outputs!!(comp1, x) == outputs!!(comp, x)
        @test outputs(comp1) == outputs(comp)

        x = rand(2)
        @test jacobian(comp1, x) == jacobian(comp, x)
        @test jacobian!(comp1, dydx, x) == jacobian!(comp, dydx, x)
        @test jacobian!(comp1, x) == jacobian!(comp, x)
        @test jacobian!!(comp1, x) == jacobian!!(comp, x)
        @test jacobian(comp1) == jacobian(comp)

        x = rand(2)
        @test outputs_and_jacobian(comp1, x) == outputs_and_jacobian(comp, x)
        @test outputs_and_jacobian!(comp1, y, dydx, x) == outputs_and_jacobian!(comp, y, dydx, x)
        @test outputs_and_jacobian!(comp1, x) == outputs_and_jacobian!(comp, x)
        @test outputs_and_jacobian!!(comp1, x) == outputs_and_jacobian!!(comp, x)
        @test outputs_and_jacobian(comp1) == outputs_and_jacobian(comp)
    end
end

@testset "ExplicitComponent - Derivatives" begin

    # This uses the paraboloid example from OpenMDAO

    f! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    df! = function(dydx, x)
        dydx[1,1] = 2*(x[1]-3) + x[2]
        dydx[1,2] = x[1] + 2*(x[2]+4)
        return dydx
    end
    fdf! = function(y, dydx, x)
        f!(y, x)
        df!(dydx, x)
        return y, dydx
    end
    x = zeros(2)
    y = zeros(1)
    dydx = zeros(1,2)
    comp = ExplicitComponent(x, y; f=f!, df=df!, fdf=fdf!)

    comp_FAD = ExplicitComponent(x, y; f=f!, deriv=ForwardAD())
    comp_RAD = ExplicitComponent(x, y; f=f!, deriv=ReverseAD())
    comp_FFD = ExplicitComponent(x, y; f=f!, deriv=ForwardFD())
    comp_CFD = ExplicitComponent(x, y; f=f!, deriv=CentralFD())
    comp_XFD = ExplicitComponent(x, y; f=f!, deriv=ComplexFD())

    for dcomp in [comp_FAD, comp_RAD, comp_FFD, comp_CFD, comp_XFD]

        x = rand(2)
        @test isapprox(outputs(comp, x), outputs(dcomp, x), atol=1e-6)
        @test isapprox(outputs!(comp, y, x), outputs!(dcomp, y, x), atol=1e-6)
        @test isapprox(outputs!(comp, x), outputs!(dcomp, x), atol=1e-6)
        @test isapprox(outputs!!(comp, x), outputs!!(dcomp, x), atol=1e-6)
        @test isapprox(outputs(comp), outputs(dcomp), atol=1e-6)

        x = rand(2)
        @test isapprox(jacobian(comp, x), jacobian(dcomp, x), atol=1e-6)
        @test isapprox(jacobian!(comp, dydx, x), jacobian!(dcomp, dydx, x), atol=1e-6)
        @test isapprox(jacobian!(comp, x), jacobian!(dcomp, x), atol=1e-6)
        @test isapprox(jacobian!!(comp, x), jacobian!!(dcomp, x), atol=1e-6)
        @test isapprox(jacobian(comp), jacobian(dcomp), atol=1e-6)

        x = rand(2)
        @test all(isapprox.(outputs_and_jacobian(comp, x), outputs_and_jacobian(dcomp, x), atol=1e-6))
        @test all(isapprox.(outputs_and_jacobian!(comp, y, dydx, x), outputs_and_jacobian!(dcomp, y, dydx, x), atol=1e-6))
        @test all(isapprox.(outputs_and_jacobian!(comp, x), outputs_and_jacobian!(dcomp, x), atol=1e-6))
        @test all(isapprox.(outputs_and_jacobian!!(comp, x), outputs_and_jacobian!!(dcomp, x), atol=1e-6))
        @test all(isapprox.(outputs_and_jacobian(comp), outputs_and_jacobian(dcomp), atol=1e-6))
    end
end

@testset "Explicit to Implicit" begin

    # This uses the paraboloid example from OpenMDAO
    f! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    df! = function(dydx, x)
        dydx[1,1] = 2*(x[1]-3) + x[2]
        dydx[1,2] = x[1] + 2*(x[2]+4)
        return dydx
    end
    x = zeros(2)
    y = zeros(1)
    xcomp = ExplicitComponent(x, y; f=f!, df=df!)
    icomp = ImplicitComponent(xcomp)

    r = zeros(1)
    drdx = zeros(1,2)
    drdy = zeros(1,1)

    x = rand(2)
    y = rand(1)
    r1 = residuals(icomp, x, y)
    r2 = residuals!(icomp, r, x, y)
    r3 = residuals!(icomp, x, y)
    r4 = residuals!!(icomp, x, y)
    r5 = residuals(icomp)
    @test r1 == r2 == r3 == r4 == r5

    x = rand(2)
    y = rand(1)
    drdx1 = residual_input_jacobian(icomp, x, y)
    drdx2 = residual_input_jacobian!(icomp, drdx, x, y)
    drdx3 = residual_input_jacobian!(icomp, x, y)
    drdx4 = residual_input_jacobian!!(icomp, x, y)
    drdx5 = residual_input_jacobian(icomp)
    @test drdx1 == drdx2 == drdx3 == drdx4 == drdx5

    x = rand(2)
    y = rand(1)
    drdy1 = residual_output_jacobian(icomp, x, y)
    drdy2 = residual_output_jacobian!(icomp, drdy, x, y)
    drdy3 = residual_output_jacobian!(icomp, x, y)
    drdy4 = residual_output_jacobian!!(icomp, x, y)
    drdy5 = residual_output_jacobian(icomp)
    @test drdy1 == drdy2 == drdy3 == drdy4 == drdy5

    x = rand(2)
    y = rand(1)
    r1, drdx1 = residuals_and_input_jacobian(icomp, x, y)
    r2, drdx2 = residuals_and_input_jacobian!(icomp, r, drdx, x, y)
    r3, drdx3 = residuals_and_input_jacobian!(icomp, x, y)
    r4, drdx4 = residuals_and_input_jacobian!!(icomp, x, y)
    r5, drdx5 = residuals_and_input_jacobian(icomp)
    @test r1 == r2 == r3 == r4 == r5
    @test drdx1 == drdx2 == drdx3 == drdx4 == drdx5

    x = rand(2)
    y = rand(1)
    r1, drdy1 = residuals_and_output_jacobian(icomp, x, y)
    r2, drdy2 = residuals_and_output_jacobian!(icomp, r, drdy, x, y)
    r3, drdy3 = residuals_and_output_jacobian!(icomp, x, y)
    r4, drdy4 = residuals_and_output_jacobian!!(icomp, x, y)
    r5, drdy5 = residuals_and_output_jacobian(icomp)
    @test r1 == r2 == r3 == r4 == r5
    @test drdy1 == drdy2 == drdy3 == drdy4 == drdy5

    x = rand(2)
    y = rand(1)
    r1, drdx1, drdy1 = residuals_and_jacobians(icomp, x, y)
    r2, drdx2, drdy2 = residuals_and_jacobians!(icomp, r, drdx, drdy, x, y)
    r3, drdx3, drdy3 = residuals_and_jacobians!(icomp, x, y)
    r4, drdx4, drdy4 = residuals_and_jacobians!!(icomp, x, y)
    r5, drdx5, drdy5 = residuals_and_jacobians(icomp)
    @test r1 == r2 == r3 == r4 == r5
    @test drdx1 == drdx2 == drdx3 == drdx4 == drdx5
    @test drdy1 == drdy2 == drdy3 == drdy4 == drdy5

end

@testset "ExplicitSystem" begin

    # System Inputs
    xsys = zeros(5)

    # First Component: Paraboloid
    f1! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    x1 = zeros(2)
    y1 = zeros(1)
    mapping1 = [
        (0,1), # system inputs, index 1
        (0,2), # system inputs, index 2
        ]
    comp1 = ExplicitComponent(x1, y1; f=f1!, deriv=ForwardAD())

    # Second Component: Quadratic
    f2! = function(y, x)
        y[1] = x[1]*x[4]^2 + x[2]*x[4] + x[3]*x[4] + 1
        return y
    end
    x2 = zeros(4)
    y2 = zeros(1)
    mapping2 = [
        (0,3), # system inputs, index 3
        (0,4), # system inputs, index 4
        (0,5), # system inputs, index 5
        (1,1), # component 1 outputs, index 1
    ]
    comp2 = ExplicitComponent(x2, y2; f=f2!, deriv=ForwardAD())

    # Third Component: Trigometric Functions
    f3! = function(y, x)
        y[1] = sin(x[1])
        y[2] = cos(x[2])
        return y
    end
    x3 = zeros(2)
    y3 = zeros(2)
    mapping3 = [
        (1,1), # component 1 outputs, index 1
        (2,1), # component 2 outputs, index 2
    ]
    comp3 = ExplicitComponent(x3, y3; f=f3!, deriv=ForwardAD())

    # System Outputs
    ysys = zeros(2)
    output_mapping = [
        (3,1), # component 3 outputs, index 1
        (3,2), # component 3 outputs, index 2
    ]

    # put it together
    x0 = xsys
    y0 = ysys
    components = (comp1, comp2, comp3)
    component_mapping = (mapping1, mapping2, mapping3)

    sys = ExplicitSystem(x0, y0, components, component_mapping, output_mapping)

    x = rand(5)
    y = rand(2)
    y1 = outputs!(sys, y, x)
    y2 = outputs!(sys, x)
    y3 = outputs!!(sys, x)
    y4 = outputs!!!(sys, x)
    y5 = outputs(sys, x)
    @test y1 == y2 == y3 == y4 == y5

    x = rand(5)
    y = rand(2)
    dydx1 = jacobian!(sys, dydx, x)
    dydx2 = jacobian!(sys, x)
    dydx3 = jacobian!!(sys, x)
    dydx4 = jacobian!!!(sys, x)
    dydx5 = jacobian(sys, x)
    @test dydx1 == dydx2 == dydx3 == dydx4 == dydx5

    y1, dydx1 = outputs_and_jacobian!(sys, y, dydx, x)
    y2, dydx2 = outputs_and_jacobian!(sys, x)
    y3, dydx3 = outputs_and_jacobian!!(sys, x)
    y4, dydx4 = outputs_and_jacobian!!!(sys, x)
    y5, dydx5 = outputs_and_jacobian(sys, x)
    @test y1 == y2 == y3 == y4 == y5
    @test dydx1 == dydx2 == dydx3 == dydx4 == dydx5

end

@testset "ExplicitSystem - Derivatives" begin

    # System Inputs
    xsys = zeros(5)

    # First Component: Paraboloid
    f1! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    x1 = zeros(2)
    y1 = zeros(1)
    mapping1 = [
        (0,1), # system inputs, index 1
        (0,2), # system inputs, index 2
        ]
    comp1 = ExplicitComponent(x1, y1; f=f1!, deriv=ForwardAD())

    # Second Component: Quadratic
    f2! = function(y, x)
        y[1] = x[1]*x[4]^2 + x[2]*x[4] + x[3]*x[4] + 1
        return y
    end
    x2 = zeros(4)
    y2 = zeros(1)
    mapping2 = [
        (0,3), # system inputs, index 3
        (0,4), # system inputs, index 4
        (0,5), # system inputs, index 5
        (1,1), # component 1 outputs, index 1
    ]
    comp2 = ExplicitComponent(x2, y2; f=f2!, deriv=ForwardAD())

    # Third Component: Trigometric Functions
    f3! = function(y, x)
        y[1] = sin(x[1])
        y[2] = cos(x[2])
        return y
    end
    x3 = zeros(2)
    y3 = zeros(2)
    mapping3 = [
        (1,1), # component 1 outputs, index 1
        (2,1), # component 2 outputs, index 2
    ]
    comp3 = ExplicitComponent(x3, y3; f=f3!, deriv=ForwardAD())

    # System Outputs
    ysys = zeros(2)
    output_mapping = [
        (3,1), # component 3 outputs, index 1
        (3,2), # component 3 outputs, index 2
    ]

    # put it together
    x0 = xsys
    y0 = ysys
    components = (comp1, comp2, comp3)
    component_mapping = (mapping1, mapping2, mapping3)

    sys = ExplicitSystem(x0, y0, components, component_mapping, output_mapping)

    # initialize storage for outputs and jacobian
    y = zeros(2)
    dydx = zeros(2,5)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_f = jacobian!(sys, dydx, x, Forward())
    @test isapprox(dydx_ad, dydx_f)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_r = jacobian!(sys, dydx, x, Reverse())
    @test isapprox(dydx_ad, dydx_r)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_f = jacobian!(sys, x, Forward())
    @test isapprox(dydx_ad, dydx_f)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_r = jacobian!(sys, x, Reverse())
    @test isapprox(dydx_ad, dydx_r)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_f = jacobian!!(sys, x, Forward())
    @test isapprox(dydx_ad, dydx_f)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_r = jacobian!!(sys, x, Reverse())
    @test isapprox(dydx_ad, dydx_r)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_f = jacobian!!!(sys, x, Forward())
    @test isapprox(dydx_ad, dydx_f)

    x = rand(5)
    dydx_ad = ForwardDiff.jacobian(x -> outputs(sys, x), x)
    dydx_r = jacobian!!!(sys, x, Reverse())
    @test isapprox(dydx_ad, dydx_r)

end

@testset "ImplicitComponent" begin

    # This uses the quadratic example from OpenMDAO

    f! = function(r, x, y)
        r[1] = x[1]*y[1]^2 + x[2]*y[1] + x[3]
        return r
    end
    dfdx! = function(drdx, x, y)
        drdx[1,1] = y[1]^2
        drdx[1,2] = y[1]
        drdx[1,3] = 1
        return drdx
    end
    dfdy! = function(drdy, x, y)
        drdy[1,1] = 2*x[1]*y[1] + x[2]
        return drdy
    end
    fdfdx! = function(r, drdx, x, y)
        f!(r, x, y)
        dfdx!(drdx, x, y)
        return r, drdx
    end
    fdfdy! = function(r, drdy, x, y)
        f!(r, x, y)
        dfdy!(drdy, x, y)
        return r, drdy
    end
    fdf! = function(r, drdx, drdy, x, y)
        f!(r, x, y)
        dfdx!(drdx, x, y)
        dfdy!(drdy, x, y)
        return r, drdx, drdy
    end
    x = [2,0,-1]
    y = zeros(1)
    r = zeros(1)
    drdx = zeros(1,3)
    drdy = zeros(1,1)
    comp1 = ImplicitComponent(x, y, r; f=f!, dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!, drdx=drdx, drdy=drdy)
    comp2 = ImplicitComponent(x, y, r; f=f!, dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!)
    comp3 = ImplicitComponent(x, y, r; f=f!, dfdx=dfdx!, dfdy=dfdy!)
    comp4 = ImplicitComponent(x, y, r; fdfdx=fdfdx!, fdfdy=fdfdy!)
    comp5 = ImplicitComponent(x, y, r; fdf=fdf!)

    for comp in [comp2, comp3, comp4, comp5]
        x = rand(3)
        y = rand(1)
        @test residuals(comp1, x, y) == residuals(comp, x, y)
        @test residuals!(comp1, r, x, y) == residuals!(comp, r, x, y)
        @test residuals!(comp1, x, y) == residuals!(comp, x, y)
        @test residuals!!(comp1, x, y) == residuals!!(comp, x, y)
        @test residuals(comp1) == residuals(comp)

        x = rand(3)
        y = rand(1)
        @test residual_input_jacobian(comp1, x, y) == residual_input_jacobian(comp, x, y)
        @test residual_input_jacobian!(comp1, drdx, x, y) == residual_input_jacobian!(comp, drdx, x, y)
        @test residual_input_jacobian!(comp1, x, y) == residual_input_jacobian!(comp, x, y)
        @test residual_input_jacobian!!(comp1, x, y) == residual_input_jacobian!!(comp, x, y)
        @test residual_input_jacobian(comp1) == residual_input_jacobian(comp)

        x = rand(3)
        y = rand(1)
        @test residual_output_jacobian(comp1, x, y) == residual_output_jacobian(comp, x, y)
        @test residual_output_jacobian!(comp1, drdy, x, y) == residual_output_jacobian!(comp, drdy, x, y)
        @test residual_output_jacobian!(comp1, x, y) == residual_output_jacobian!(comp, x, y)
        @test residual_output_jacobian!!(comp1, x, y) == residual_output_jacobian!!(comp, x, y)
        @test residual_output_jacobian(comp1) == residual_output_jacobian(comp)

        x = rand(3)
        y = rand(1)
        @test residuals_and_input_jacobian(comp1, x, y) == residuals_and_input_jacobian(comp, x, y)
        @test residuals_and_input_jacobian!(comp1, r, drdx, x, y) == residuals_and_input_jacobian!(comp, r, drdx, x, y)
        @test residuals_and_input_jacobian!(comp1, x, y) == residuals_and_input_jacobian!(comp, x, y)
        @test residuals_and_input_jacobian!!(comp1, x, y) == residuals_and_input_jacobian!!(comp, x, y)
        @test residuals_and_input_jacobian(comp1) == residuals_and_input_jacobian(comp)

        x = rand(3)
        y = rand(1)
        @test residuals_and_output_jacobian(comp1, x, y) == residuals_and_output_jacobian(comp, x, y)
        @test residuals_and_output_jacobian!(comp1, r, drdy, x, y) == residuals_and_output_jacobian!(comp, r, drdy, x, y)
        @test residuals_and_output_jacobian!(comp1, x, y) == residuals_and_output_jacobian!(comp, x, y)
        @test residuals_and_output_jacobian!!(comp1, x, y) == residuals_and_output_jacobian!!(comp, x, y)
        @test residuals_and_output_jacobian(comp1) == residuals_and_output_jacobian(comp)

        x = rand(3)
        y = rand(1)
        @test residuals_and_jacobians(comp1, x, y) == residuals_and_jacobians(comp, x, y)
        @test residuals_and_jacobians!(comp1, r, drdx, drdy, x, y) == residuals_and_jacobians!(comp, r, drdx, drdy, x, y)
        @test residuals_and_jacobians!(comp1, x, y) == residuals_and_jacobians!(comp, x, y)
        @test residuals_and_jacobians!!(comp1, x, y) == residuals_and_jacobians!!(comp, x, y)
        @test residuals_and_jacobians(comp1) == residuals_and_jacobians(comp)
    end

end

@testset "ImplicitComponent - Derivatives" begin

    # This uses the quadratic example from OpenMDAO

    f! = function(r, x, y)
        r[1] = x[1]*y[1]^2 + x[2]*y[1] + x[3]
        return r
    end
    dfdx! = function(drdx, x, y)
        drdx[1,1] = y[1]^2
        drdx[1,2] = y[1]
        drdx[1,3] = 1
        return drdx
    end
    dfdy! = function(drdy, x, y)
        drdy[1,1] = 2*x[1]*y[1] + x[2]
        return drdy
    end
    x = [2,0,-1]
    y = zeros(1)
    r = zeros(1)
    drdx = zeros(1,3)
    drdy = zeros(1,1)
    comp = ImplicitComponent(x, y, r; f=f!, dfdx=dfdx!, dfdy=dfdy!)

    comp_FAD = ImplicitComponent(x, y, r; f=f!, xderiv=ForwardAD(), yderiv=ForwardAD())
    comp_RAD = ImplicitComponent(x, y, r; f=f!, xderiv=ReverseAD(), yderiv=ReverseAD())
    comp_FFD = ImplicitComponent(x, y, r; f=f!, xderiv=ForwardFD(), yderiv=ForwardFD())
    comp_CFD = ImplicitComponent(x, y, r; f=f!, xderiv=CentralFD(), yderiv=CentralFD())
    comp_XFD = ImplicitComponent(x, y, r; f=f!, xderiv=ComplexFD(), yderiv=ComplexFD())

    i = 0
    for dcomp in [comp_FAD, comp_RAD, comp_FFD, comp_CFD, comp_XFD]

        i += 1

        x = rand(3)
        y = rand(1)
        @test isapprox(residuals(comp, x, y), residuals(dcomp, x, y))
        @test isapprox(residuals!(comp, r, x, y), residuals!(dcomp, r, x, y))
        @test isapprox(residuals!(comp, x, y), residuals!(dcomp, x, y))
        @test isapprox(residuals!!(comp, x, y), residuals!!(dcomp, x, y))
        @test isapprox(residuals(comp), residuals(dcomp))

        x = rand(3)
        y = rand(1)
        @test isapprox(residual_input_jacobian(comp, x, y), residual_input_jacobian(dcomp, x, y))
        @test isapprox(residual_input_jacobian!(comp, drdx, x, y), residual_input_jacobian!(dcomp, drdx, x, y))
        @test isapprox(residual_input_jacobian!(comp, x, y), residual_input_jacobian!(dcomp, x, y))
        @test isapprox(residual_input_jacobian!!(comp, x, y), residual_input_jacobian!!(dcomp, x, y))
        @test isapprox(residual_input_jacobian(comp), residual_input_jacobian(dcomp))

        x = rand(3)
        y = rand(1)
        @test isapprox(residual_output_jacobian(comp, x, y), residual_output_jacobian(dcomp, x, y))
        @test isapprox(residual_output_jacobian!(comp, drdy, x, y), residual_output_jacobian!(dcomp, drdy, x, y))
        @test isapprox(residual_output_jacobian!(comp, x, y), residual_output_jacobian!(dcomp, x, y))
        @test isapprox(residual_output_jacobian!!(comp, x, y), residual_output_jacobian!!(dcomp, x, y))
        @test isapprox(residual_output_jacobian(comp), residual_output_jacobian(dcomp))

        x = rand(3)
        y = rand(1)
        @test all(isapprox.(residuals_and_input_jacobian(comp, x, y), residuals_and_input_jacobian(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_input_jacobian!(comp, r, drdx, x, y), residuals_and_input_jacobian!(dcomp, r, drdx, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_input_jacobian!(comp, x, y), residuals_and_input_jacobian!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_input_jacobian!!(comp, x, y), residuals_and_input_jacobian!!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_input_jacobian(comp), residuals_and_input_jacobian(dcomp), atol=1e-6))

        x = rand(3)
        y = rand(1)
        @test all(isapprox.(residuals_and_output_jacobian(comp, x, y), residuals_and_output_jacobian(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_output_jacobian!(comp, r, drdy, x, y), residuals_and_output_jacobian!(dcomp, r, drdy, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_output_jacobian!(comp, x, y), residuals_and_output_jacobian!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_output_jacobian!!(comp, x, y), residuals_and_output_jacobian!!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_output_jacobian(comp), residuals_and_output_jacobian(dcomp), atol=1e-6))

        x = rand(3)
        y = rand(1)
        @test all(isapprox.(residuals_and_jacobians(comp, x, y), residuals_and_jacobians(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_jacobians!(comp, r, drdx, drdy, x, y), residuals_and_jacobians!(dcomp, r, drdx, drdy, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_jacobians!(comp, x, y), residuals_and_jacobians!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_jacobians!!(comp, x, y), residuals_and_jacobians!!(dcomp, x, y), atol=1e-6))
        @test all(isapprox.(residuals_and_jacobians(comp), residuals_and_jacobians(dcomp), atol=1e-6))

    end
end

@testset "Implicit to Explicit" begin

    # This uses the quadratic example from OpenMDAO

    f! = function(r, x, y)
        r[1] = x[1]*y[1]^2 + x[2]*y[1] + x[3]
        return r
    end

    dfdx! = function(drdx, x, y)
        drdx[1,1] = y[1]^2
        drdx[1,2] = y[1]
        drdx[1,3] = 1
        return drdx
    end

    dfdy! = function(drdy, x, y)
        drdy[1,1] = 2*x[1]*y[1] + x[2]
        return drdy
    end

    x = [2,0,-1]
    y = zeros(1)
    r = zeros(1)

    icomp = ImplicitComponent(x, y, r; f=f!, dfdx=dfdx!, dfdy=dfdy!)
    xcomp = ExplicitComponent(icomp)

    dydx = zeros(1,3)

    y1 = outputs!(xcomp, y, x)
    y2 = outputs!(xcomp, x)
    y3 = outputs!!(xcomp, x)
    y4 = outputs!!!(xcomp, x)
    y5 = outputs(xcomp, x)
    @test isapprox(y1, y2, atol=1e-6)
    @test isapprox(y1, y3, atol=1e-6)
    @test isapprox(y1, y4, atol=1e-6)
    @test isapprox(y1, y5, atol=1e-6)

    dydx1 = jacobian!(xcomp, dydx, x)
    dydx2 = jacobian!(xcomp, x)
    dydx3 = jacobian!!(xcomp, x)
    dydx4 = jacobian!!!(xcomp, x)
    dydx5 = jacobian(xcomp, x)
    @test isapprox(dydx1, dydx2, atol=1e-6)
    @test isapprox(dydx1, dydx3, atol=1e-6)
    @test isapprox(dydx1, dydx4, atol=1e-6)
    @test isapprox(dydx1, dydx5, atol=1e-6)

    x = -rand(3)
    y1, dydx1 = outputs_and_jacobian!(xcomp, y, dydx, x)
    y2, dydx2 = outputs_and_jacobian!(xcomp, x)
    y3, dydx3 = outputs_and_jacobian!!(xcomp, x)
    y4, dydx4 = outputs_and_jacobian!!!(xcomp, x)
    y5, dydx5 = outputs_and_jacobian(xcomp, x)
    @test isapprox(y1, y2, atol=1e-6)
    @test isapprox(y1, y3, atol=1e-6)
    @test isapprox(y1, y4, atol=1e-6)
    @test isapprox(y1, y5, atol=1e-6)
    @test isapprox(dydx1, dydx2, atol=1e-6)
    @test isapprox(dydx1, dydx3, atol=1e-6)
    @test isapprox(dydx1, dydx4, atol=1e-6)
    @test isapprox(dydx1, dydx5, atol=1e-6)

end

@testset "ImplicitSystem" begin

    # We use an explicit system here, modified to become implicit

    # System Inputs
    xsys = zeros(5)

    # First Component: Paraboloid
    f1! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end
    x1 = zeros(2)
    y1 = zeros(1)
    mapping1 = [
        (0,1), # system inputs, index 1
        (0,2), # system inputs, index 2
        ]
    xcomp1 = ExplicitComponent(x1, y1; f=f1!, deriv=ForwardAD())
    icomp1 = ImplicitComponent(xcomp1)

    # Second Component: Quadratic
    f2! = function(y, x)
        y[1] = x[1]*x[4]^2 + x[2]*x[4] + x[3]*x[4] + 1
        return y
    end
    x2 = zeros(4)
    y2 = zeros(1)
    mapping2 = [
        (0,3), # system inputs, index 3
        (0,4), # system inputs, index 4
        (0,5), # system inputs, index 5
        (1,1), # component 1 outputs, index 1
    ]
    xcomp2 = ExplicitComponent(x2, y2; f=f2!, deriv=ForwardAD())
    icomp2 = ImplicitComponent(xcomp2)

    # Third Component: Trigometric Functions
    f3! = function(y, x)
        y[1] = sin(x[1])
        y[2] = cos(x[2])
        return y
    end
    x3 = zeros(2)
    y3 = zeros(2)
    mapping3 = [
        (1,1), # component 1 outputs, index 1
        (2,1), # component 2 outputs, index 2
    ]
    xcomp3 = ExplicitComponent(x3, y3; f=f3!, deriv=ForwardAD())
    icomp3 = ImplicitComponent(xcomp3)

    # System Outputs
    ysys = zeros(2)
    output_mapping = [
        (3,1), # component 3 outputs, index 1
        (3,2), # component 3 outputs, index 2
    ]

    # put it together
    x0 = xsys
    y0 = ysys
    r0 = copy(ysys)
    components = (icomp1, icomp2, icomp3)
    component_mapping = (mapping1, mapping2, mapping3)

    sys = ImplicitSystem(x0, components, component_mapping, output_mapping)

    r = zeros(4)
    drdx = zeros(4, 5)
    drdy = zeros(4, 4)

    x = rand(5)
    y = rand(4)
    r1 = residuals!(sys, r, x, y)
    r2 = residuals!(sys, x, y)
    r3 = residuals!!(sys, x, y)
    r4 = residuals!!!(sys, x, y)
    r5 = residuals(sys, x, y)
    @test isapprox(r1, r2)
    @test isapprox(r1, r3)
    @test isapprox(r1, r4)
    @test isapprox(r1, r5)

    x = rand(5)
    y = rand(4)
    drdx1 = residual_input_jacobian!(sys, drdx, x, y)
    drdx2 = residual_input_jacobian!(sys, x, y)
    drdx3 = residual_input_jacobian!!(sys, x, y)
    drdx4 = residual_input_jacobian!!!(sys, x, y)
    drdx5 = residual_input_jacobian(sys, x, y)
    @test isapprox(drdx1, drdx2)
    @test isapprox(drdx1, drdx3)
    @test isapprox(drdx1, drdx4)
    @test isapprox(drdx1, drdx5)

    x = rand(5)
    y = rand(4)
    drdy1 = residual_output_jacobian!(sys, drdy, x, y)
    drdy2 = residual_output_jacobian!(sys, x, y)
    drdy3 = residual_output_jacobian!!(sys, x, y)
    drdy4 = residual_output_jacobian!!!(sys, x, y)
    drdy5 = residual_output_jacobian(sys, x, y)
    @test isapprox(drdy1, drdy2)
    @test isapprox(drdy1, drdy3)
    @test isapprox(drdy1, drdy4)
    @test isapprox(drdy1, drdy5)

    x = rand(5)
    y = rand(4)
    r1, drdx1 = residuals_and_input_jacobian!(sys, r, drdx, x, y)
    r2, drdx2 = residuals_and_input_jacobian!(sys, x, y)
    r3, drdx3 = residuals_and_input_jacobian!!(sys, x, y)
    r4, drdx4 = residuals_and_input_jacobian!!!(sys, x, y)
    r5, drdx5 = residuals_and_input_jacobian(sys, x, y)
    @test isapprox(r1, r2)
    @test isapprox(r1, r3)
    @test isapprox(r1, r4)
    @test isapprox(r1, r5)
    @test isapprox(drdx1, drdx2)
    @test isapprox(drdx1, drdx3)
    @test isapprox(drdx1, drdx4)
    @test isapprox(drdx1, drdx5)

    x = rand(5)
    y = rand(4)
    r1, drdy1 = residuals_and_output_jacobian!(sys, r, drdy, x, y)
    r2, drdy2 = residuals_and_output_jacobian!(sys, x, y)
    r3, drdy3 = residuals_and_output_jacobian!!(sys, x, y)
    r4, drdy4 = residuals_and_output_jacobian!!!(sys, x, y)
    r5, drdy5 = residuals_and_output_jacobian(sys, x, y)
    @test isapprox(r1, r2)
    @test isapprox(r1, r3)
    @test isapprox(r1, r4)
    @test isapprox(r1, r5)
    @test isapprox(drdy1, drdy2)
    @test isapprox(drdy1, drdy3)
    @test isapprox(drdy1, drdy4)
    @test isapprox(drdy1, drdy5)

    x = rand(5)
    y = rand(4)
    r1, drdx1, drdy1 = residuals_and_jacobians!(sys, r, drdx, drdy, x, y)
    r2, drdx2, drdy2 = residuals_and_jacobians!(sys, x, y)
    r3, drdx3, drdy3 = residuals_and_jacobians!!(sys, x, y)
    r4, drdx4, drdy4 = residuals_and_jacobians!!!(sys, x, y)
    r5, drdx5, drdy5 = residuals_and_jacobians(sys, x, y)
    @test isapprox(r1, r2)
    @test isapprox(r1, r3)
    @test isapprox(r1, r4)
    @test isapprox(r1, r5)
    @test isapprox(drdx1, drdx2)
    @test isapprox(drdx1, drdx3)
    @test isapprox(drdx1, drdx4)
    @test isapprox(drdx1, drdx5)
    @test isapprox(drdy1, drdy2)
    @test isapprox(drdy1, drdy3)
    @test isapprox(drdy1, drdy4)
    @test isapprox(drdy1, drdy5)

end
