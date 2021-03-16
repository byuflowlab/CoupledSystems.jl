using CoupledSystems
using ForwardDiff
using Test

@testset "Named Variables" begin

    # default values for each variable
    x1 = rand()
    y1 = rand(10)
    z1 = rand(10, 10)

    # define each named variable
    @var x = x1
    @var y = y1
    @var z = z1

    # create tuple of named variables
    vars = (x, y, z)

    # combine values into a single vector
    v = combine(vars)

    # separate variables
    x2, y2, z2 = separate(vars, v)

    # check that variable values didn't change
    @test x1 == x2
    @test y1 == y2
    @test z1 == z2
end

@testset "ExplicitComponent" begin

    # This uses the paraboloid example from OpenMDAO

    # define variables and set defaults
    @var x = 0.0
    @var y = 0.0
    @var fxy = 0.0

    # construct paraboloid function (and define inputs and outputs)
    fin = (x, y)
    fout = (fxy,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct optional component output function
    f! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end

    # construct optional component jacobian function
    df! = function(dydx, x)
        dydx[1,1] = 2*(x[1]-3) + x[2]
        dydx[1,2] = x[1] + 2*(x[2]+4)
        return dydx
    end

    # construct optional combined component output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        df!(dydx, x)
        return y, dydx
    end

    # define optional jacobian matrix storage
    dydx = zeros(1,2)

    # test a bunch of different constructors
    comp1 = ExplicitComponent(func, fin, fout, foutin; f=f!, df=df!, fdf=fdf!, dydx=dydx)
    comp2 = ExplicitComponent(func, fin, fout, foutin; df=df!, fdf=fdf!, dydx=dydx)
    comp3 = ExplicitComponent(func, fin, fout, foutin; df=df!, fdf=fdf!)
    comp4 = ExplicitComponent(func, fin, fout, foutin; df=df!)
    comp5 = ExplicitComponent(func, fin, fout, foutin; fdf=fdf!)
    comp6 = ExplicitComponent(fin, fout, foutin; f=f!, df=df!, fdf=fdf!, dydx=dydx)
    comp7 = ExplicitComponent(fin, fout, foutin; f=f!, df=df!, fdf=fdf!)
    comp8 = ExplicitComponent(fin, fout, foutin; f=f!, df=df!)
    comp9 = ExplicitComponent(fin, fout, foutin; fdf=fdf!)

    # make sure all the constructors yield the same results
    for comp in [comp2, comp3, comp4, comp5, comp6, comp7, comp8, comp9]

        # storage for outputs and jacobian
        y = zeros(1)
        dydx = zeros(1,2)

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

    # define variables and set defaults
    @var x = 0.0
    @var y = 0.0
    @var fxy = 0.0

    # construct template function
    fin = (x, y)
    fout = (fxy,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct optional component output function
    f! = function(y, x)
        y[1] = (x[1]-3)^2 + x[1]*x[2] + (x[2]+4)^2 - 3
        return y
    end

    # construct optional component jacobian function
    df! = function(dydx, x)
        dydx[1,1] = 2*(x[1]-3) + x[2]
        dydx[1,2] = x[1] + 2*(x[2]+4)
        return dydx
    end

    # construct optional combined component output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        df!(dydx, x)
        return y, dydx
    end

    x = zeros(2)
    y = zeros(1)
    dydx = zeros(1,2)
    comp = ExplicitComponent(func, fin, fout, foutin; f=f!, df=df!, fdf=fdf!)

    comp_FAD = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())
    comp_RAD = ExplicitComponent(func, fin, fout, foutin; deriv=ReverseAD())
    comp_FFD = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardFD())
    comp_CFD = ExplicitComponent(func, fin, fout, foutin; deriv=CentralFD())
    comp_XFD = ExplicitComponent(func, fin, fout, foutin; deriv=ComplexFD())

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

    # define variables and set defaults
    @var x = 0.0
    @var y = 0.0
    @var fxy = 0.0

    # construct template function
    fin = (x, y)
    fout = (fxy,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct explicit component
    xcomp = ExplicitComponent(func, fin, fout, foutin)

    # construct implicit component from explicit component
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

    # define system variables and defaults
    @var x = 0.0
    @var y = 0.0
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var fp = 0.0
    @var fq = 0.0
    @var ft = zeros(2)

    # construct template function for paraboloid component
    fin = (x, y)
    fout = (fp,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct paraboloid component
    paraboloid = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Second Component: Quadratic

    # construct template function for quadratic component
    fin = (a, b, c, fp)
    fout = (fq,)
    foutin = ()
    func = function(a, b, c, fp)
        fq = a*fp^2 + b*fp + c*fp + 1
        return fq
    end

    # construct quadratic component
    quadratic = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Third Component: Trigometric Functions

    # construct template function for trigometric component
    fin = (fp, fq)
    fout = ()
    foutin = (ft,)
    func = function(ft, fp, fq)
        ft[1] = sin(fp)
        ft[2] = cos(fq)
        return ft
    end

    # construct trigometric component
    trigometric = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Combined System

    argin = (x, y, a, b, c)
    argout = (ft,)
    components = (paraboloid, quadratic, trigometric)

    sys = ExplicitSystem(components, argin, argout)

    y = zeros(2)
    dydx = zeros(2,5)

    x = rand(5)
    y1 = outputs!(sys, y, x)
    y2 = outputs!(sys, x)
    y3 = outputs!!(sys, x)
    y4 = outputs!!!(sys, x)
    y5 = outputs(sys, x)
    @test y1 == y2 == y3 == y4 == y5

    x = rand(5)
    dydx1 = jacobian!(sys, dydx, x)
    dydx2 = jacobian!(sys, x)
    dydx3 = jacobian!!(sys, x)
    dydx4 = jacobian!!!(sys, x)
    dydx5 = jacobian(sys, x)
    @test dydx1 == dydx2 == dydx3 == dydx4 == dydx5

    x = rand(5)
    y1, dydx1 = outputs_and_jacobian!(sys, y, dydx, x)
    y2, dydx2 = outputs_and_jacobian!(sys, x)
    y3, dydx3 = outputs_and_jacobian!!(sys, x)
    y4, dydx4 = outputs_and_jacobian!!!(sys, x)
    y5, dydx5 = outputs_and_jacobian(sys, x)
    @test y1 == y2 == y3 == y4 == y5
    @test dydx1 == dydx2 == dydx3 == dydx4 == dydx5
end

@testset "ExplicitSystem - Derivatives" begin

    # define system variables and defaults
    @var x = 0.0
    @var y = 0.0
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var fp = 0.0
    @var fq = 0.0
    @var ft = zeros(2)

    # construct template function for paraboloid component
    fin = (x, y)
    fout = (fp,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct paraboloid component
    paraboloid = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Second Component: Quadratic

    # construct template function for quadratic component
    fin = (a, b, c, fp)
    fout = (fq,)
    foutin = ()
    func = function(a, b, c, fp)
        fq = a*fp^2 + b*fp + c*fp + 1
        return fq
    end

    # construct quadratic component
    quadratic = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Third Component: Trigometric Functions

    # construct template function for trigometric component
    fin = (fp, fq)
    fout = ()
    foutin = (ft,)
    func = function(ft, fp, fq)
        ft[1] = sin(fp)
        ft[2] = cos(fq)
        return ft
    end

    # construct trigometric component
    trigometric = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Combined System
    argin = (x, y, a, b, c)
    argout = (ft,)
    components = (paraboloid, quadratic, trigometric)

    sys = ExplicitSystem(components, argin, argout)

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

    # variables
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var x = 0.0

    # template residual function
    argin = (a, b, c)
    argout = (x,)
    func = function(r, a, b, c, x)
        r[1] = a*x^2 + b*x + c
        return r
    end

    # vector form of residual function
    f! = function(r, x, y)
        r[1] = x[1]*y[1]^2 + x[2]*y[1] + x[3]
        return r
    end

    # jacobian wrt inputs
    dfdx! = function(drdx, x, y)
        drdx[1,1] = y[1]^2
        drdx[1,2] = y[1]
        drdx[1,3] = 1
        return drdx
    end

    # jacobian wrt outputs
    dfdy! = function(drdy, x, y)
        drdy[1,1] = 2*x[1]*y[1] + x[2]
        return drdy
    end

    # residual and jacobian wrt inputs
    fdfdx! = function(r, drdx, x, y)
        f!(r, x, y)
        dfdx!(drdx, x, y)
        return r, drdx
    end

    # residual and jacobian wrt outputs
    fdfdy! = function(r, drdy, x, y)
        f!(r, x, y)
        dfdy!(drdy, x, y)
        return r, drdy
    end

    # residual and jacobians
    fdf! = function(r, drdx, drdy, x, y)
        f!(r, x, y)
        dfdx!(drdx, x, y)
        dfdy!(drdy, x, y)
        return r, drdx, drdy
    end

    # storage for results
    r = zeros(1)
    drdx = zeros(1,3)
    drdy = zeros(1,1)

    comp1 = ImplicitComponent(func, argin, argout, r; f=f!, dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!, drdx=drdx, drdy=drdy)
    comp2 = ImplicitComponent(func, argin, argout, r; dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!, drdx=drdx, drdy=drdy)
    comp3 = ImplicitComponent(func, argin, argout, r; dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!)
    comp4 = ImplicitComponent(func, argin, argout, r; dfdx=dfdx!, dfdy=dfdy!)
    comp5 = ImplicitComponent(func, argin, argout; fdfdx=fdfdx!, fdfdy=fdfdy!)
    comp6 = ImplicitComponent(func, argin, argout; fdf=fdf!)

    comp7 = ImplicitComponent(argin, argout; f=f!, dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!, drdx=drdx, drdy=drdy)
    comp8 = ImplicitComponent(argin, argout; f=f!, dfdx=dfdx!, dfdy=dfdy!, fdfdx=fdfdx!,
        fdfdy=fdfdy!, fdf=fdf!)
    comp9 = ImplicitComponent(argin, argout; f=f!, dfdx=dfdx!, dfdy=dfdy!)
    comp10 = ImplicitComponent(argin, argout; fdfdx=fdfdx!, fdfdy=fdfdy!)
    comp11 = ImplicitComponent(argin, argout; fdf=fdf!)

    for comp in [comp2, comp3, comp4, comp5, comp6, comp7, comp8, comp9, comp10, comp11]
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

    # variables
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var x = 0.0

    # template residual function
    fin = (a, b, c)
    fout = (x,)
    func = function(r, a, b, c, x)
        r[1] = a*x^2 + b*x + c
        return r
    end

    # jacobian wrt inputs
    dfdx! = function(drdx, x, y)
        drdx[1,1] = y[1]^2
        drdx[1,2] = y[1]
        drdx[1,3] = 1
        return drdx
    end

    # jacobian wrt outputs
    dfdy! = function(drdy, x, y)
        drdy[1,1] = 2*x[1]*y[1] + x[2]
        return drdy
    end

    comp = ImplicitComponent(func, fin, fout; dfdx=dfdx!, dfdy=dfdy!)

    comp_FAD = ImplicitComponent(func, fin, fout; xderiv=ForwardAD(), yderiv=ForwardAD())
    comp_RAD = ImplicitComponent(func, fin, fout; xderiv=ReverseAD(), yderiv=ReverseAD())
    comp_FFD = ImplicitComponent(func, fin, fout; xderiv=ForwardFD(), yderiv=ForwardFD())
    comp_CFD = ImplicitComponent(func, fin, fout; xderiv=CentralFD(), yderiv=CentralFD())
    comp_XFD = ImplicitComponent(func, fin, fout; xderiv=ComplexFD(), yderiv=ComplexFD())

    for dcomp in [comp_FAD, comp_RAD, comp_FFD, comp_CFD, comp_XFD]

        x = rand(3)
        y = rand(1)
        @test isapprox(residuals(comp, x, y), residuals(dcomp, x, y), atol=1e-6)
        @test isapprox(residuals!(comp, r, x, y), residuals!(dcomp, r, x, y), atol=1e-6)
        @test isapprox(residuals!(comp, x, y), residuals!(dcomp, x, y), atol=1e-6)
        @test isapprox(residuals!!(comp, x, y), residuals!!(dcomp, x, y), atol=1e-6)
        @test isapprox(residuals(comp), residuals(dcomp), atol=1e-6)

        x = rand(3)
        y = rand(1)
        @test isapprox(residual_input_jacobian(comp, x, y), residual_input_jacobian(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_input_jacobian!(comp, drdx, x, y), residual_input_jacobian!(dcomp, drdx, x, y), atol=1e-6)
        @test isapprox(residual_input_jacobian!(comp, x, y), residual_input_jacobian!(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_input_jacobian!!(comp, x, y), residual_input_jacobian!!(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_input_jacobian(comp), residual_input_jacobian(dcomp), atol=1e-6)

        x = rand(3)
        y = rand(1)
        @test isapprox(residual_output_jacobian(comp, x, y), residual_output_jacobian(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_output_jacobian!(comp, drdy, x, y), residual_output_jacobian!(dcomp, drdy, x, y), atol=1e-6)
        @test isapprox(residual_output_jacobian!(comp, x, y), residual_output_jacobian!(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_output_jacobian!!(comp, x, y), residual_output_jacobian!!(dcomp, x, y), atol=1e-6)
        @test isapprox(residual_output_jacobian(comp), residual_output_jacobian(dcomp), atol=1e-6)

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

    # variables
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var x = 0.0

    # template residual function
    fin = (a, b, c)
    fout = (x,)
    func = function(r, a, b, c, x)
        r[1] = a*x^2 + b*x + c
        return r
    end

    # create implicit component
    icomp = ImplicitComponent(func, fin, fout)

    # create explicit component
    xcomp = ExplicitComponent(icomp)

    # inputs
    x = [2, 0, -1]

    # storage for outputs
    y = zeros(1)
    dydx = zeros(1, 3)

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
    # define system variables and defaults
    @var x = 0.0
    @var y = 0.0
    @var a = 0.0
    @var b = 0.0
    @var c = 0.0
    @var fp = 0.0
    @var fq = 0.0
    @var ft = zeros(2)

    # construct template function for paraboloid component
    fin = (x, y)
    fout = (fp,)
    foutin = ()
    func = (x,y) -> (x - 3)^2 + x*y + (y + 4)^2 - 3

    # construct paraboloid component
    paraboloid = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Second Component: Quadratic

    # construct template function for quadratic component
    fin = (a, b, c, fp)
    fout = (fq,)
    foutin = ()
    func = function(a, b, c, fp)
        fq = a*fp^2 + b*fp + c*fp + 1
        return fq
    end

    # construct quadratic component
    quadratic = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # Third Component: Trigometric Functions

    # construct template function for trigometric component
    fin = (fp, fq)
    fout = ()
    foutin = (ft,)
    func = function(ft, fp, fq)
        ft[1] = sin(fp)
        ft[2] = cos(fq)
        return ft
    end

    # construct trigometric component
    trigometric = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

    # components in implicit system
    components = (paraboloid, quadratic, trigometric)

    # arguments to implicit system
    argin = (x,y,a,b,c)

    # construct implicit system
    sys = ImplicitSystem(components, argin)

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
