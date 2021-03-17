function DifferentialEquations.ODEFunction(comp::ExplicitComponent, uvar::Tuple, duvar::Tuple,
    pvar::Tuple, tvar::NamedVar)

    # component arguments
    cargin = comp.argin
    cargout = comp.argout

    # get default vectors
    u0 = combine(uvar)
    p0 = combine(pvar)
    du0 = combine(duvar)

    # check that there is a derivative for each state vector
    @assert length(u0) == length(du0)

    # number of state variables and parameters
    nu = length(u0)
    np = length(p0)

    # construct mapping matrices...
    dudx = mapping_matrix(cargin, uvar)
    dpdx = mapping_matrix(cargin, pvar)
    dtdx = mapping_matrix(cargin, (tvar,))
    ddudy = mapping_matrix(cargout, duvar)

    # ...and their inverses
    dxdu = dudx'
    dxdp = dpdx'
    dxdt = dtdx'
    dyddu = ddudy'

    # create storage for intermediate matrices
    dydx = jacobian(comp)
    dydt = dydx*dxdt
    dydu = dydx*dxdu
    dydp = dydx*dxdp

    # prototype jacobian
    jac_prototype = ddudy*dydu

    # function for updating outputs
    x = copy(inputs(comp))
    f = function(du, u, p, t)
        TF = promote_type(eltype(du), eltype(du0))
        if TF <: eltype(du0)
            # map inputs to component inputs
            mul!(x, dxdu, u)
            mul!(x, dxdp, p, 1, 1)
            mul!(x, dxdt, t, 1, 1)
            # get outputs
            y = outputs!(comp, x)
        else
            # map inputs to component inputs
            x = dxdu*u + dxdp*p + dxdt*t
            # get outputs
            y = outputs(comp, x)
        end
        # map component outputs to derivatives
        mul!(du, ddudy, y)
        # return derivatives of state variables
        return du
    end

    # function for gradient of time
    tgrad = function(dT, u, p, t)
        TF = promote_type(eltype(dT), eltype(du0))
        if TF <: eltype(du0)
            # map inputs to component inputs
            mul!(x, dxdu, u)
            mul!(x, dxdp, p, 1, 1)
            mul!(x, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, x)
        else
            # map inputs to component inputs
            x = dxdu*u + dxdp*p + dxdt*t
            # get outputs
            y = jacobian(comp, x)
        end
        # calculate time gradient from component jacobian
        mul!(dydt, dydx, dxdt)
        mul!(dT, ddudy, dydt)
        return dT
    end

    # function for jacobian wrt state variables
    jac = function(J, u, p, t)
        TF = promote_type(eltype(J), eltype(du0))
        if TF <: eltype(du0)
            # map inputs to component inputs
            mul!(x, dxdu, u)
            mul!(x, dxdp, p, 1, 1)
            mul!(x, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, x)
        else
            # map inputs to component inputs
            x = dxdu*u + dxdp*p + dxdt*t
            # get outputs
            y = jacobian(comp, x)
        end
        # calculate time gradient from component jacobian
        mul!(dydu, dydx, dxdu)
        mul!(J, ddudy, dydu)
        return J
    end

    # function for jacobian wrt parameters
    paramjac = function(pJ, u, p, t)
        TF = promote_type(eltype(pJ), eltype(du0))
        if TF <: eltype(du0)
            # map inputs to component inputs
            mul!(x, dxdu, u)
            mul!(x, dxdp, p, 1, 1)
            mul!(x, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, x)
        else
            # map inputs to component inputs
            x = dxdu*u + dxdp*p + dxdt*t
            # get outputs
            y = jacobian(comp, x)
        end
        # calculate time gradient from component jacobian
        mul!(dydp, dydx, dxdp)
        mul!(pJ, ddudy, dydp)
        return pJ
    end

    return DifferentialEquations.ODEFunction{true, true}(f;
         tgrad = tgrad, # (dT,u,p,t) or (u,p,t)
         jac = jac, # (J,u,p,t) or (u,p,t)
         jac_prototype = jac_prototype, # Type for the Jacobian
         paramjac = paramjac, # (pJ,u,p,t) or (u,p,t)
         colorvec = matrix_colors(jac_prototype),
         syms = name.(uvar))
end
