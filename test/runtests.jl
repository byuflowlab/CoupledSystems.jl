using CoupledSystems
using Test

@testset "Constructors" begin

    # System 1

    # residual of first system
    function f1!(F, x, u)
        F[1] = 4*x[1]^2 - x[2] + u[1]
        F[2] = 2*x[1] + (1 - x[2])^2 + u[2]
        return F
    end

    # output of first system
    function g1!(G, x, u)
        G[1] = x[1]
        G[2] = x[1] + x[2]
        return G
    end

    # (partial) derivative of residual wrt x
    function dfdx1!(dF, x, u)
        dF[1,1] = 8*x[1]
        dF[1,2] = -1
        dF[2,1] = 2
        dF[2,2] = -2*(1 - x[2])
        return dF
    end

    # (partial) derivative of residual wrt u
    function dfdu1!(dF, x, u)
        dF[1,1] = 1
        dF[1,2] = 0
        dF[2,1] = 0
        dF[2,2] = 1
        return dF
    end

    # (partial) derivative of output wrt x
    function dgdx1!(dG, x, u)
        dG[1,1] = 1
        dG[1,2] = 0
        dG[2,1] = 1
        dG[2,2] = 1
        return dG
    end

    # (partial) derivative of output wrt u
    function dgdu1!(dG, x, u)
        dG[1,1] = 0
        dG[1,2] = 0
        dG[2,1] = 0
        dG[2,2] = 0
        return dG
    end

    # System 2

    # residual and output of second system
    function f2!(F, x, u)
        F[1] = x[1] + x[2] + u[1]
        F[2] = (x[1]-3)^2 - x[2]^2 + u[2]
        return F
    end

    # output of first system
    function g2!(G, x, u)
        G[1] = x[1] + 3*x[2]
        G[2] = cos(x[1] + x[2]^2)
        return G
    end

    # (partial) derivative of residual wrt x
    function dfdx2!(dF, x, u)
        dF[1,1] = 1
        dF[1,2] = 1
        dF[2,1] = 2*(x[1]-3)
        dF[2,2] = -2*x[2]
        return dF
    end

    # (partial) derivative of residual wrt u
    function dfdu2!(dF, x, u)
        dF[1,1] = 1
        dF[1,2] = 0
        dF[2,1] = 0
        dF[2,2] = 1
        return dF
    end

    # (partial) derivative of output wrt x
    function dgdx2!(dG, x, u)
        dG[1,1] = 1
        dG[1,2] = 3
        dG[2,1] = -sin(x[1] + x[2]^2)
        dG[2,2] = 2*x[2]*sin(x[1] + x[2]^2)
        return dG
    end

    # (partial) derivative of output wrt u
    function dgdu2!(dG, x, u)
        dG[1,1] = 0
        dG[1,2] = 0
        dG[2,1] = 0
        dG[2,2] = 0
        return dG
    end

    nsys = 2

    x0 = [zeros(2) for i = 1:2]
    u0 = [zeros(2) for i = 1:2]
    f0 = [zeros(2) for i = 1:2]
    g0 = [zeros(2) for i = 1:2]
    dfdx0 = [zeros(2,2) for i = 1:2]
    dfdu0 = [zeros(2,2) for i = 1:2]
    dgdx0 = [zeros(2,2) for i = 1:2]
    dgdu0 = [zeros(2,2) for i = 1:2]
    mapping = [3,4,1,2];

    f! = (f1!, f2!)
    g! = (g1!, g2!)
    dfdx! = (dfdx1!, dfdx2!)
    dfdu! = (dfdu1!, dfdu2!)
    dgdx! = (dgdx1!, dgdx2!)
    dgdu! = (dgdu1!, dgdu2!)

    mapping_system = [[2, 2], [1, 1]]
    mapping_index = [[1, 2], [1, 2]]

    # construction from individual equations
    system = CoupledSystem(f0, g0, dfdx0, dfdu0, dgdx0, dgdu0, mapping_system,
        mapping_index; f!, g!, dfdx!, dfdu!, dgdx!, dgdu!)

    # construct fg! from f! and g!
    fg! = Vector{Function}(undef, nsys)
    for i = 1:nsys
        fi!, gi! = f![i], g![i]
        fg![i] = function(F, G, x, u)
            fi!(F, x, u)
            gi!(G, x, u)
            return nothing
        end
    end

    # construct dfg! from dfdx!, dfdu!, dgdx!, and dgdu!
    dfg! = Vector{Function}(undef, nsys)
    for i = 1:nsys
        dfdxi!, dfdui! = dfdx![i], dfdu![i]
        dgdxi!, dgdui! = dgdx![i], dgdu![i]
        dfg![i] = function(DFX, DFU, DGX, DGU, x, u)
            dfdxi!(DFX, x, u)
            dfdui!(DFU, x, u)
            dgdxi!(DGX, x, u)
            dgdui!(DGU, x, u)
            return nothing
        end
    end

    # construction from combined residuals/outputs and combined partials
    system = CoupledSystem(f0, g0, dfdx0, dfdu0, dgdx0, dgdu0, mapping_system,
        mapping_index; fg!, dfg!)

    # construct fgdfg! from fg! and dfg!
    fgdfg! = Vector{Function}(undef, nsys)
    for i = 1:nsys
        fgi! = fg![i]
        dfgi! = dfg![i]
        fgdfg![i] = function(F, G, DFX, DFU, DGX, DGU, x, u)
            fgi!(F, G, x, u)
            dfgi!(DFX, DFU, DGX, DGU, x, u)
            return nothing
        end
    end

    # construction from combined residuals, outputs, and partials
    system = CoupledSystem(f0, g0, dfdx0, dfdu0, dgdx0, dgdu0, mapping_system,
        mapping_index; fgdfg!)

end
