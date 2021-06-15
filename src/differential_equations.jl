"""
    ODEFunction(comp::AbstractExplicitComponent, duvar::Tuple, uvar::Tuple, pvar::Tuple,
        tvar::NamedVar)

Construct an ODEFunction for use with DifferentialEquations from an explicit
component.
"""
function DifferentialEquations.ODEFunction(comp::AbstractExplicitComponent, duvar::Tuple, uvar::Tuple,
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
    ddudy = mapping_matrix(cargout, duvar)
    dudx = mapping_matrix(cargin, uvar)
    dpdx = mapping_matrix(cargin, pvar)
    dtdx = mapping_matrix(cargin, (tvar,))

    # ...and their inverses
    dyddu = ddudy'
    dxdu = dudx'
    dxdp = dpdx'
    dxdt = dtdx'

    # create storage for intermediate matrices
    dydx = jacobian(comp)
    dydt = dydx*dxdt
    dydu = dydx*dxdu
    dydp = dydx*dxdp

    # prototype jacobian
    jac_prototype = ddudy*dydu

    # cache for inputs
    xcache = copy(inputs(comp))

    # function for updating outputs
    f = function(du, u, p, t)
        TF = promote_type(eltype(xcache), eltype(u), eltype(p), eltype(t))
        if TF <: eltype(xcache)
            # map inputs to component inputs
            mul!(xcache, dxdu, u)
            mul!(xcache, dxdp, p, 1, 1)
            mul!(xcache, dxdt, t, 1, 1)
            # get outputs
            y = outputs!(comp, xcache)
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
        TF = promote_type(eltype(xcache), eltype(u), eltype(p), eltype(t))
        if TF <: eltype(xcache)
            # map inputs to component inputs
            mul!(xcache, dxdu, u)
            mul!(xcache, dxdp, p, 1, 1)
            mul!(xcache, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, xcache)
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
        TF = promote_type(eltype(xcache), eltype(u), eltype(p), eltype(t))
        if TF <: eltype(xcache)
            # map inputs to component inputs
            mul!(xcache, dxdu, u)
            mul!(xcache, dxdp, p, 1, 1)
            mul!(xcache, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, xcache)
        else
            # map inputs to component inputs
            x = dxdu*u + dxdp*p + dxdt*t
            # get outputs
            dydx = jacobian(comp, x)
        end
        # calculate time gradient from component jacobian
        mul!(dydu, dydx, dxdu)
        mul!(J, ddudy, dydu)
        return J
    end

    # function for jacobian wrt parameters
    paramjac = function(pJ, u, p, t)
        TF = promote_type(eltype(xcache), eltype(u), eltype(p), eltype(t))
        if TF <: eltype(xcache)
            # map inputs to component inputs
            mul!(xcache, dxdu, u)
            mul!(xcache, dxdp, p, 1, 1)
            mul!(xcache, dxdt, t, 1, 1)
            # get component jacobian
            dydx = jacobian!(comp, xcache)
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
         syms = get_names(uvar))
end

"""
    DAEFunction(comp::AbstractImplicitComponent, duvar::Tuple, uvar::Tuple, pvar::Tuple,
        tvar::NamedVar)

Construct a DAEFunction for use with DifferentialEquations from an implicit
component.
"""
function DifferentialEquations.DAEFunction(comp::AbstractImplicitComponent, duvar::Tuple, uvar::Tuple,
    pvar::Tuple, tvar::NamedVar)

end

"""
    stability_analysis(comp, duvar, uvar, pvar)

Construct an explicit component with inputs corresponding to an ODE or DAE's
state variables and parameters at a stationary point and outputs corresponding
to the system's eigenvalues.

For derivative calculations, the eigenvalues are assumed to be unique.
"""
stability_analysis

function stability_analysis(comp::AbstractExplicitComponent, duvar, uvar, pvar;
    nev=6, which=LR())

    # get jacobian of explicit component
    y_x = jacobian(comp)

    # split jacobian into state and parameter jacobians
    mul!(du_u, u_y, y_x)
    mul!(du_p, p_y, y_x)

    # compute eigenvalues and eigenvectors
    λ, V = partialeigen(partialschur(du_u; nev, which)[1])

    # sort eigenvalues by magnitude
    perm = sortperm(λ, by=(λ)->(abs(λ),imag(λ)), rev=true)
    λ .= λ[perm]
    V .= V[:,perm]

    # compute left eigenvectors
    U = left_eigenvectors(K, λ, V)

    # calculate eigenvalue derivatives wrt the parameters
    dλ = U*du_p*V

end

function stability_analysis(comp::AbstractImplicitComponent, duvar, uvar, pvar)

end

function left_eigenvectors(K, M, λ, V)

    # problem type and dimensions
    TC = eltype(V)
    nx = size(V,1)
    nev = size(V,2)

    # allocate storage
    U = rand(TC, nev, nx)
    u = Vector{TC}(undef, nx)
    tmp = Vector{TC}(undef, nx)

    # get entries in M
    iM, jM, valM = findnz(M)

    # compute eigenvectors for each eigenvalue
    for iλ = 1:nev

        # factorize (K + λ*M)'
        KmλMfact = factorize(K' - λ[iλ]'*M')

        # initialize left eigenvector
        for i = 1:nx
            u[i] = U[iλ,i]
        end

        # perform a few iterations to converge the left eigenvector
        for ipass = 1:3
            # get updated u
            mul!(tmp, M, u)
            ldiv!(u, KmλMfact, tmp)
            # normalize u
            unorm = zero(TC)
            for k = 1:length(valM)
                unorm += conj(u[iM[k]])*valM[k]*V[jM[k],iλ]
            end
            rdiv!(u, conj(unorm))
        end

        # store conjugate of final eigenvector
        for i = 1:nx
            U[iλ,i] = conj(u[i])
        end
    end

    return U
end

"""
    add_implicit_ode_solver(comp, duvar, uvar, cvar, pvar, tvar, disc; kwargs...)

Constructs an implicit system component from an ODE problem using a pseudospectral
method.

# Arguments:
 - `comp`: Explicit subcomponent which defines the ODE
 - `duvar`: Tuple of named variables corresponding to subcomponent state rates
 - `uvar`: Tuple of named variables corresponding to subcomponent states
 - `cvar`: Tuple of named variables corresponding to subcomponent (dynamic) controls
 - `pvar`: Tuple of named variables corresponding to subcomponent (static) parameters
 - `tvar`: Named variable corresponding to subcomponent time
 - `disc`: Discretization of time into segments and nodes
 - `duration`: Named variable corresponding to the simulation duration.
 - `boundary_states`: Named variable (or values) corresponding to the boundary
    state variables.  Whether these parameters represent the initial or final
    state variables depends on the value of the `fix_final` flag.

# Keyword Arguments:
 - `states`: Tuple of named variables corresponding to each state in `uvar`.
    Defaults to using the names and default values of the corresponding parameters
    in `uvar`, but with an additional dimension for time with length corresponding
    to the number of discretization nodes in `disc`.
 - `controls`: Tuple of named variables corresponding to each control variable in
    `cvar`.  Defaults to using the names and default values of the corresponding
    parameters in `cvar`, but with additional dimension for time with length
    corresponding to the number of nodes in `disc`.
 - `parameters`: Tuple of named variables corresponding to each parameter in
    `pvar`. Defaults to using the names and default values of the corresponding
    parameters in `pvar`.
 - `initial_time`: Named variable (or value) corresponding to initial time. Defaults to `0`.
 - `fix_initial`: Flag indicating whether boundary conditions should be enforced
    at the initial rather than final time.  Defaults to `true`.
 - `component_inputs`: Names (and order) of the inputs to this component.  Defaults to
    all possible variables (all named variables in `controls`, `parameters,
    `duration, `initial_time`, `initial_states`, and/or `final_states`).
 - `component_outputs`: Names (and order) of the outputs from this component.  Defaults to
    all possible variables (all named variables in `states`)
"""
function add_implicit_ode_solver(comp, duvar, uvar, cvar, pvar, tvar, disc,
    duration, boundary_states; states = nothing, controls = nothing, parameters = nothing,
    initial_time = 0.0, fix_initial = true, component_inputs = nothing, component_outputs = nothing)

    # extract discretization properties
    nsegs = number_of_segments(disc)
    nnodes = number_of_nodes(disc)
    ndnodes = number_of_discretization_nodes(disc)
    ncnodes = number_of_collocation_nodes(disc)
    nsegnodes = number_of_segment_nodes(disc)
    nsegdnodes = number_of_segment_discretization_nodes(disc)
    nsegcnodes = number_of_segment_collocation_nodes(disc)
    isegdnodes = get_discretization_indices(disc)
    isegcnodes = get_collocation_indices(disc)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_nodes(disc)
    compact = iscompact(disc)

    # populate system states if not provided
    if isnothing(states)
        varnames = get_name.(uvar)
        varvalues = get_value.(uvar)
        vardims = ndims.(varvalues)
        states = Tuple(NamedVar(varnames[i], repeat(make_array(varvalues[i]), outer=(ones(Int, vardims[i])..., ndnodes))) for i = 1:length(varnames))
    end

    # check state variable inputs
    @assert length(states) == length(uvar) "Number of component and subcomponent state variables don't match"
    for i = 1:length(uvar)
        @assert size(states[i])[1:end-1] == size(uvar[i]) "Dimensions of component and subcomponent state variables for state variable $i don't match"
        @assert size(states[i])[end] == ndnodes "Final dimension of the component input for state variable $i doesn't match the number of discretization nodes $ndnodes"
    end

    # populate system controls if not provided
    if isnothing(controls)
        varnames = get_name.(cvar)
        varvalues = get_value.(cvar)
        vardims = ndims.(varvalues)
        controls = Tuple(NamedVar(varnames[i], repeat(make_array(varvalues[i]), outer=(ones(Int, vardims[i])..., nnodes))) for i = 1:length(varnames))
    end

    # check control parameter inputs
    @assert length(controls) == length(cvar) "Number of component and subcomponent control parameters don't match"
    for i = 1:length(cvar)
        @assert size(controls[i])[1:end-1] == size(cvar[i]) "Dimensions of component and subcomponent control parameters for control parameter $i don't match"
        @assert size(controls[i])[end] == nnodes "Final dimension of the component input for control parameter $i doesn't match the total number of discretization and collocation nodes $nnodes"
    end

    # populate system parameters if not provided
    if isnothing(parameters)
        parameters = pvar
    end

    # check static parameter inputs
    @assert length(parameters) == length(pvar) "Number of component and subcomponent static parameters don't match"
    for i = 1:length(pvar)
        @assert size(parameters[i]) == size(pvar[i]) "Dimensions of component and subcomponent control parameters for control parameter $i don't match"
    end

    # check time inputs
    @assert length(tvar) == 1 "Subcomponent time input must be a single variable"
    @assert length(initial_time) == 1 "Initial time input must be a single variable"
    @assert length(duration) == 1 "Duration input must be a single variable"

    # combine inputs and outputs
    combined_inputs = (controls..., parameters..., duration, initial_time, boundary_states...)
    combined_outputs = states

    # set names of component inputs if not provided
    if isnothing(component_inputs)
        component_inputs = get_names(combined_inputs)
    end

    # set names of component outputs if not provided
    if isnothing(component_outputs)
        component_outputs = get_names(combined_outputs)
    end

    # check that specified inputs/outputs correspond to actual inputs/outputs
    @assert all(in(get_names(combined_inputs)), component_inputs) "Not all specified inputs are present in `controls`, `parameters`, `duration`, and `initial_time`"
    @assert all(in(get_names(combined_outputs)), component_outputs) "Not all specified outputs are present in `states`"

    # get indices of component inputs and outputs
    component_input_indices = findall(in(component_inputs), get_name.(combined_inputs))
    component_output_indices = findall(in(component_outputs), get_name.(combined_outputs))

    # named variables corresponding to component inputs/outputs
    argin = combined_inputs[component_input_indices]
    argout = combined_outputs[component_output_indices]

    # assemble default system inputs
    u0 = combine(states)
    idx = 0
    for ivar = 1:length(states)
        if get_name(states[ivar]) in component_outputs
            u0[idx + 1 : idx + length(states[ivar])] .= 0.0
        end
        idx += length(states[ivar])
    end

    c0 = combine(controls)
    idx = 0
    for ivar = 1:length(controls)
        if get_name(controls[ivar]) in component_inputs
            c0[idx + 1 : idx + length(controls[ivar])] .= 0.0
        end
        idx += length(controls[ivar])
    end

    p0 = combine(parameters)
    idx = 0
    for ivar = 1:length(parameters)
        if get_name(parameters[ivar]) in component_inputs
            p0[idx + 1 : idx + length(parameters[ivar])] .= 0.0
        end
        idx += length(parameters[ivar])
    end

    ti0 = combine(initial_time)
    if get_name(initial_time) in component_inputs
        ti0 .= 0.0
    end

    td0 = combine(duration)
    if get_name(duration) in component_inputs
        td0 .= 0.0
    end

    ub0 = combine(boundary_states)
    idx = 0
    for ivar = 1:length(boundary_states)
        if get_name(boundary_states[ivar]) in component_inputs
            ub0[idx + 1 : idx + length(boundary_states[ivar])] .= 0.0
        end
        idx += length(boundary_states[ivar])
    end

    # create system mapping matrices
    u_y = mapping_matrix(argout, states)
    c_x = mapping_matrix(argin, controls)
    p_x = mapping_matrix(argin, parameters)
    ti_x = mapping_matrix(argin, (initial_time,))
    td_x = mapping_matrix(argin, (duration,))
    ub_x = mapping_matrix(argin, boundary_states)

    # create subcomponent mapping matrices
    xsub_usub = mapping_matrix(uvar, comp.argin)
    xsub_csub = mapping_matrix(cvar, comp.argin)
    xsub_p = mapping_matrix(pvar, comp.argin)
    xsub_tsub = mapping_matrix((tvar,), comp.argin)
    dusub_ysub = mapping_matrix(comp.argout, duvar)

    # get problem dimensions
    nu = length(combine(uvar)) # number of subcomponent states
    nc = length(combine(cvar)) # number of subcomponent controls
    np = length(combine(pvar)) # number of subcomponent parameters
    nt = 1 # number of subcomponent time variables
    nx = nu + nc + np + nt # number of subcomponent inputs
    ny = nu # number of subcomponent outputs

    # component internal storage
    x_f = combine(argin)
    y_f = combine(argout)
    x_dfdx = copy(x_f)
    y_dfdx = copy(y_f)
    x_dfdy = copy(x_f)
    y_dfdy = copy(y_f)
    r = copy(y_f) .* NaN
    drdx = spzeros(promote_type(eltype(x_f), eltype(y_f)), length(r), length(x_f))
    drdy = spzeros(promote_type(eltype(x_f), eltype(y_f)), length(r), length(y_f))

    # reshape defaults and mapping to match internal representation
    u0 = permutedims(reshape(u0, nu, ndnodes))[:]
    u_y = sparse(reshape(permutedims(reshape(u_y, ndnodes, nu, length(y_f)), [2,1,3]), nu*ndnodes, length(y_f)))
    c0 = permutedims(reshape(c0, nc, nnodes))[:]
    c_x = sparse(reshape(permutedims(reshape(c_x, nnodes, nc, length(x_f)), [2,1,3]), nc*nnodes, length(x_f)))

    # cache variables
    ucache = similar(u0, nu*nnodes) # system state inputs
    ccache = similar(c0, nc*nnodes) # system control inputs
    pcache = similar(p0, np) # parameter inputs
    ticache = similar(ti0, nt) # initial time input
    tdcache = similar(td0, nt) # duration input
    ubcache = similar(ub0, nu) # boundary states input
    ducache = similar(combine(duvar), nu*nnodes) # system state rate outputs
    xsub_cache = similar(inputs(comp)) # subcomponent input vector
    ysub_usub_cache = similar(outputs(comp) .* combine(uvar)') # subcomponent output derivatives wrt states
    ysub_csub_cache = similar(outputs(comp) .* combine(cvar)') # subcomponent output derivatives wrt controls
    ysub_tsub_cache = similar(outputs(comp) .* combine(tvar)') # subcomponent output derivatives wrt local time
    ysub_p_cache = similar(outputs(comp) .* combine(pvar)') # subcomponent output derivatives wrt parameters
    ysub_ti_cache = similar(outputs(comp) .* combine(initial_time)') # subcomponent output derivatives wrt initial time
    ysub_td_cache = similar(outputs(comp) .* combine(duration)') # subcomponent output derivatives wrt problem duration
    ysub_usegd_cache = similar(ysub_usub_cache, ny, nu*nsegdnodes) # subcomponent output derivatives wrt states at segment discretization nodes
    ysub_csegd_cache = similar(ysub_csub_cache, ny, nc*nsegdnodes) # subcomponent output derivatives wrt states at control discretization nodes
    du_u_cache = similar(ysub_usub_cache, nu, nnodes, nu, nsegdnodes)
    du_c_cache = similar(ysub_csub_cache, nu, nnodes, nc, nsegnodes)
    du_p_cache = similar(ysub_p_cache, nu, nnodes, np)
    du_ti_cache = similar(ysub_ti_cache, nu, nnodes)
    du_td_cache = similar(ysub_td_cache, nu, nnodes)
    r_u_cache = similar(ysub_usub_cache, nu, ncnodes, nu, nsegdnodes)
    r_c_cache = similar(ysub_csub_cache, nu, ncnodes, nc, nsegnodes)
    r_p_cache = similar(ysub_p_cache, nu, ncnodes, np)
    r_ti_cache = similar(ysub_ti_cache, nu, ncnodes)
    r_td_cache = similar(ysub_td_cache, nu, ncnodes)
    r_ub_cache = similar(ysub_usub_cache, nu, ncnodes, nu)
    r_usys_cache = spzeros(eltype(ysub_usub_cache), nu*ncnodes, nu*ndnodes)
    r_csys_cache = spzeros(eltype(ysub_csub_cache), nu*ncnodes, nc*nnodes)

    f = function(r, x, y)
        # unpack system cache variables
        u = ucache # system state inputs
        c = ccache # system control inputs
        p = pcache # parameter inputs
        ti = ticache # initial time input
        td = tdcache # duration input
        ub = ubcache # boundary states input
        du = ducache # system state rate outputs
        xsub = xsub_cache # subcomponent inputs
        # set default system values
        copyto!(u, u0)
        copyto!(c, c0)
        copyto!(p, p0)
        copyto!(ti, ti0)
        copyto!(td, td0)
        copyto!(ub, ub0)
        # set variable system values
        mul!(u, u_y, y, 1, 1)
        mul!(c, c_x, x, 1, 1)
        mul!(p, p_x, x, 1, 1)
        mul!(ti, ti_x, x, 1, 1)
        mul!(td, td_x, x, 1, 1)
        mul!(ub, ub_x, x, 1, 1)
        # reshape system values for more convenient use
        r = reshape(r, nu, ncnodes+1)
        u = reshape(u, nu, nnodes)
        du = reshape(du, nu, nnodes)
        c = reshape(c, nc, nnodes)
        ti = ti[1]
        td = td[1]
        # loop through each segment
        for iseg = 1:nsegs
            # calculate segment initial time and duration
            tiseg = ti + segment_ends[iseg]*td
            tdseg = (segment_ends[iseg+1] - segment_ends[iseg])*td
            # loop through all nodes in this segment
            for inode = 1:nsegnodes
                # calculate state rates at the state discretization nodes
                if inode in isegdnodes
                    # current index in time
                    it = (iseg - 1)*(nsegnodes - compact) + inode
                    # get inputs to the ODE at this point in time
                    usub = view(u, 1:nu, it)
                    csub = view(c, 1:nc, it)
                    tsub = tiseg + segment_nodes[inode] * tdseg
                    # map inputs to subcomponent inputs
                    mul!(xsub, xsub_usub, usub)
                    mul!(xsub, xsub_csub, csub, 1, 1)
                    mul!(xsub, xsub_p, p, 1, 1)
                    mul!(xsub, xsub_tsub, tsub, 1, 1)
                    # get outputs
                    ysub = outputs!(comp, xsub)
                    # map outputs to state rates
                    dusub = view(du, 1:nu, it)
                    mul!(dusub, dusub_ysub, ysub)
                end
            end
            # interpolate each state to the collocation nodes (if necessary)
            for iu = 1:nu
                # range of nodes corresponding to this segment
                irange = (iseg - 1)*(nsegnodes - compact) + 1 : iseg*(nsegnodes - compact) + compact
                # current state variables and rates across segment
                useg = view(u, iu, irange)
                duseg = view(du, iu, irange)
                # interpolate states
                set_collocation_node_values!(useg, duseg, tdseg, disc)
            end
            # loop through all nodes in this segment
            for inode = 1:nsegnodes
                # calculate state rates at the collocation nodes (if necessary)
                if (inode in isegcnodes) && !(inode in isegdnodes)
                    # set global time index
                    it = (iseg - 1)*(nsegnodes - compact) + inode
                    # subcomponent state, control, and time inputs
                    usub = view(u, :, it)
                    csub = view(c, :, it)
                    tsub = tiseg + segment_nodes[inode] * tdseg
                    # map inputs to subcomponent inputs
                    mul!(xsub, xsub_usub, usub)
                    mul!(xsub, xsub_csub, csub, 1, 1)
                    mul!(xsub, xsub_p, p, 1, 1)
                    mul!(xsub, xsub_tsub, tsub, 1, 1)
                    # get outputs
                    ysub = outputs!(comp, xsub_cache)
                    # map outputs to state rates
                    dusub = view(du, 1:nu, it)
                    mul!(dusub, dusub_ysub, ysub)
                end
            end
            # calculate the polynomial fit defects
            for iu = 1:nu
                # range of nodes corresponding to this segment
                irange = (iseg - 1)*(nsegnodes - compact) + 1 : iseg*(nsegnodes - compact) + compact
                # range of collocation nodes corresponding to this segment
                irange_c = view(irange, isegcnodes)
                # portion of residual vector corresponding to this segment and state variable
                rseg = view(r, iu, irange_c)
                # current state variables and rates across segment
                useg = view(u, iu, irange)
                duseg = view(du, iu, irange)
                # calculate defects
                @time calculate_defects!(rseg, useg, duseg, tdseg, disc)
            end
        end
        # add residual equations corresponding to the boundary condition
        if fix_initial
            # boundary condition is on initial states
            r[1:nu, end] .= view(u, :, 1) .- ub
        else
            # boundary condition is on final states
            r[1:nu, end] .= view(u, :, ncnodes+1) .- ub
        end
        return r
    end

    fdf = function(r, drdx, drdy, x, y)
        # unpack system cache variables
        u = ucache # system state inputs
        c = ccache # system control inputs
        p = pcache # parameter inputs
        ti = ticache # initial time input
        td = tdcache # duration input
        ub = ubcache
        du = ducache # system state rate outputs
        du_u = du_u_cache # derivative of state rates wrt states at segment discretization nodes
        du_c = du_c_cache # derivative of state rates wrt states at segment nodes
        du_p = du_p_cache # derivative of state rates wrt parameters
        du_ti = du_ti_cache # derivative of state rates wrt initial time
        du_td = du_td_cache # derivative of state rates wrt duration
        r_u = r_u_cache
        r_c = r_c_cache
        r_p = r_p_cache
        r_ti = r_ti_cache
        r_td = r_td_cache
        r_ub = r_ub_cache
        r_usys = r_usys_cache
        r_csys = r_csys_cache
        # unpack subcomponent cache variables
        xsub = xsub_cache # inputs
        ysub_usub = ysub_usub_cache # derivatives of outputs wrt states
        ysub_csub = ysub_csub_cache # derivatives of outputs wrt controls
        ysub_tsub = ysub_tsub_cache # derivatives of outputs wrt time
        ysub_p = ysub_p_cache # derivatives of outputs wrt parameters
        ysub_ti = ysub_ti_cache # derivatives of outputs wrt initial time
        ysub_td = ysub_td_cache # derivatives of outputs wrt duration
        ysub_usegd = ysub_usegd_cache # derivatives of outputs wrt segment discretization node states
        ysub_csegd = ysub_csegd_cache # derivatives of outputs wrt segment discretization node controls
        # clear all system variables
        u .= 0.0
        c .= 0.0
        p .= 0.0
        ti .= 0.0
        td .= 0.0
        ub .= 0.0
        # set default system values
        u .+= u0
        c .+= c0
        p .+= p0
        ti .+= ti0
        td .+= td0
        ub .+= ub0
        # set variable system values
        mul!(u, u_y, y, 1, 1)
        mul!(c, c_x, x, 1, 1)
        mul!(p, p_x, x, 1, 1)
        mul!(ti, ti_x, x, 1, 1)
        mul!(td, td_x, x, 1, 1)
        mul!(ub, ub_x, x, 1, 1)
        # reshape system values for more convenient use
        r = reshape(r, nu, ncnodes)
        u = reshape(u, nu, nnodes)
        du = reshape(du, nu, nnodes)
        c = reshape(c, nc, nnodes)
        ti = ti[1]
        td = td[1]
        # begin calculations for each segment
        for iseg = 1:nsegs
            # calculate segment initial time and duration (and partials)
            tiseg = ti + segment_ends[iseg]*td
            tiseg_ti = 1
            tiseg_td = segment_ends[iseg]

            tdseg = (segment_ends[iseg+1] - segment_ends[iseg])*td
            tdseg_ti = 0
            tdseg_td = (segment_ends[iseg+1] - segment_ends[iseg])

            # loop through all nodes in this segment
            for inode = 1:nsegnodes
                # local discretization node index
                inode_d = 0
                # calculate state rates at the state discretization nodes
                if inode in isegdnodes
                    # set global time index
                    it = (iseg - 1)*(nsegnodes - compact) + inode
                    # increment local discretization node index
                    inode_d += 1
                    # subcomponent state, control, and time inputs
                    usub = view(u, 1:nu, it)
                    csub = view(c, 1:nc, it)
                    tsub = tiseg + segment_nodes[inode] * tdseg
                    # propagate derivatives
                    tsub_ti = tiseg_ti + segment_nodes[inode] * tdseg_ti
                    tsub_td = tiseg_td + segment_nodes[inode] * tdseg_td
                    # map inputs to subcomponent inputs
                    mul!(xsub, xsub_usub, usub)
                    mul!(xsub, xsub_csub, csub, 1, 1)
                    mul!(xsub, xsub_p, p, 1, 1)
                    mul!(xsub, xsub_tsub, tsub, 1, 1)
                    # get outputs and jacobian
                    ysub, ysub_xsub = outputs_and_jacobian!(comp, xsub)
                    # propagate derivatives
                    mul!(ysub_usub, ysub_xsub, xsub_usub)
                    mul!(ysub_csub, ysub_xsub, xsub_csub)
                    mul!(ysub_p, ysub_xsub, xsub_p)
                    mul!(ysub_tsub, ysub_xsub, xsub_tsub)
                    mul!(ysub_ti, ysub_tsub, tsub_ti)
                    mul!(ysub_td, ysub_tsub, tsub_td)
                    # map outputs to state rates
                    dusub = view(du, 1:nu, it)
                    mul!(dusub, dusub_ysub, ysub)
                    # propagate derivatives
                    dusub_usub = view(du_u, 1:nu, it, 1:nu, inode_d)
                    dusub_csub = view(du_c, 1:nu, it, 1:nc, inode)
                    dusub_p = view(du_p, 1:nu, it, 1:np)
                    dusub_ti = view(du_ti, 1:nu, it)
                    dusub_td = view(du_td, 1:nu, it)
                    mul!(dusub_usub, dusub_ysub, ysub_usub)
                    mul!(dusub_csub, dusub_ysub, ysub_csub)
                    mul!(dusub_p, dusub_ysub, ysub_p)
                    mul!(dusub_ti, dusub_ysub, ysub_ti)
                    mul!(dusub_td, dusub_ysub, ysub_td)
                end
            end
            # interpolate each state to the collocation nodes (if necessary)
            for iu = 1:nu
                # range of nodes corresponding to this segment
                irange = (iseg - 1)*(nsegnodes - compact) + 1 : iseg*(nsegnodes - compact) + compact
                # current state variables and rates across segment
                useg = view(u, iu, irange)
                duseg = view(du, iu, irange)
                # partials of current state variables and rates across segment
                useg_usegd = view(u_u, iu, irange, 1:nu, 1:nsegdnodes)
                useg_csegd = view(u_c, iu, irange, 1:nc, isegdnodes)
                useg_p = view(u_p, iu, irange, 1:np)
                useg_ti = view(u_ti, iu, irange)
                useg_td = view(u_td, iu, irange)
                duseg_usegd = view(du_u, iu, irange, 1:nu, 1:nsegdnodes)
                duseg_csegd = view(du_c, iu, irange, 1:nc, isegdnodes)
                duseg_psegd = view(du_p, iu, irange, 1:np)
                duseg_ti = view(du_ti, iu, irange)
                duseg_td = view(du_td, iu, irange)
                # interpolate states
                set_collocation_node_values!(useg, duseg, tdseg, disc)
                # propagate derivatives
                set_collocation_node_values_u!(useg_usegd, duseg_usegd, tdseg, disc)
                set_collocation_node_values_c!(useg_csegd, duseg_csegd, tdseg, disc)
                set_collocation_node_values_p!(useg_psegd, useg_psegd, tdseg, disc)
                set_collocation_node_values_ti!(useg_ti, duseg_ti, tdseg, disc)
                set_collocation_node_values_td!(useg_td, duseg_td, tdseg, tdseg_td, duseg, disc)
            end
            # loop through all nodes in this segment
            for inode = 1:nsegnodes
                # calculate state rates at the collocation nodes (if necessary)
                if (inode in isegcnodes) && !(inode in isegdnodes)
                    # set global time index
                    it = (iseg - 1)*(nsegnodes - compact) + inode
                    # subcomponent state, control, and time inputs
                    usub = view(u, :, it)
                    csub = view(c, :, it)
                    tsub = tiseg + segment_nodes[inode] * tdseg
                    # propagate derivatives
                    tsub_ti = tiseg_ti + segment_nodes[inode] * tdseg_ti
                    tsub_td = tiseg_td + segment_nodes[inode] * tdseg_td
                    # extract derivatives of state wrt discretization node parameters
                    usub_usegd = view(u_u, 1:nu, it, 1:nu, 1:nsegdnodes)
                    usub_csegd = view(u_c, 1:nu, it, 1:nc, isegdnodes)
                    usub_p = view(u_p, 1:nu, it, 1:np)
                    usub_ti = view(u_ti, 1:nu, it)
                    usub_td = view(u_td, 1:nu, it)
                    # map inputs to subcomponent inputs
                    mul!(xsub, xsub_usub, usub)
                    mul!(xsub, xsub_csub, csub, 1, 1)
                    mul!(xsub, xsub_p, p, 1, 1)
                    mul!(xsub, xsub_tsub, tsub, 1, 1)
                    # get outputs and jacobian
                    ysub, ysub_xsub = outputs_and_jacobian!(comp, xsub)
                    # propagate derivatives
                    mul!(ysub_usub, ysub_xsub, xsub_usub)
                    mul!(ysub_csub, ysub_xsub, xsub_csub) # collocation node controls
                    mul!(ysub_tsub, ysub_xsub, xsub_tsub)
                    mul!(mul!(ysub_p, ysub_usub, usub_p), ysub_xsub, xsub_p, 1, 1) # parameters
                    mul!(mul!(ysub_ti, ysub_usub, usub_ti), ysub_tsub, tsub_ti, 1, 1) # initial time
                    mul!(mul!(ysub_td, ysub_usub, usub_td), ysub_tsub, tsub_td, 1, 1) # duration
                    mul!(ysub_usegd, ysub_usub, usub_usegd) # discretization node states
                    mul!(ysub_csegd, ysub_usub, usub_csegd) # discretization node controls
                    # map outputs to state rates
                    dusub = view(du, 1:nu, it)
                    mul!(dusub, dusub_ysub, ysub)
                    # propagate derivatives
                    dusub_csub = view(du_c, 1:nu, inode, 1:nc, inode)
                    dusub_p = view(du_p, 1:nu, inode, 1:np, isegdnodes)
                    dusub_ti = view(du_ti, 1:nu, inode)
                    dusub_td = view(du_td, 1:nu, inode)
                    dusub_usegd = view(du_u, 1:nu, inode, 1:nu, 1:nsegdnodes)
                    dusub_csegd = view(du_c, 1:nu, inode, 1:nc, isegdnodes)
                    mul!(dusub_csub, dusub_ysub, ysub_csub) # collocation node controls
                    mul!(dusub_p, dusub_ysub, ysub_p) # parameters
                    mul!(dusub_ti, dusub_ysub, ysub_ti) # initial time
                    mul!(dusub_td, dusub_ysub, ysub_td) # final time
                    mul!(dusub_usegd, dusub_ysub, ysub_usegd) # discretization node states
                    mul!(dusub_csegd, dusub_ysub, ysub_csegd) # discretization node controls
                end
            end
            # calculate the polynomial fit defects
            for iu = 1:nu
                # range of nodes corresponding to this segment
                irange = (iseg - 1)*(nsegnodes - compact) + 1 : iseg*(nsegnodes - compact) + compact
                # range of collocation nodes corresponding to this segment
                irange_c = irange[isegcnodes]
                # portion of residual vector corresponding to this segment and state variable
                rseg = view(r, iu, irange_c)
                rseg_usegd = view(r_u, iu, irange_c, 1:nu, 1:nsegdnodes)
                rseg_cseg = view(r_c, iu, irange_c, 1:nc, 1:nsegnodes)
                rseg_p = view(r_p, iu, irange_c, 1:np)
                rseg_ti = view(r_ti, iu, irange_c)
                rseg_td = view(r_td, iu, irange_c)
                # current state variables and rates across segment
                useg = view(u, iu, irange)
                duseg = view(du, iu, irange)
                # derivatives of current state variables and rates across segment
                useg_usegd = view(u_u, iu, irange, 1:nu, 1:nsegdnodes)
                useg_cseg = view(u_c, iu, irange, 1:nc, 1:nsegnodes)
                useg_p = view(u_p, iu, irange, 1:np)
                useg_ti = view(u_ti, iu, irange)
                useg_td = view(u_td, iu, irange)
                duseg_usegd = view(du_u, iu, irange, 1:nu, 1:nsegdnodes)
                duseg_cseg = view(du_c, iu, irange, 1:nc, 1:nsegnodes)
                duseg_p = view(du_p, iu, irange, 1:np)
                duseg_ti = view(du_ti, iu, irange)
                duseg_td = view(du_td, iu, irange)
                # calculate defects
                calculate_defects!(rseg, useg, duseg, tdseg, disc)
                # propagate derivatives
                calculate_defects_u!(rseg_usegd, useg_usegd, duseg_usegd, tdseg, disc)
                calculate_defects_c!(rseg_cseg, useg_cseg, duseg_cseg, tdseg, disc)
                calculate_defects_p!(rseg_p, useg_p, duseg_p, tdseg, disc)
                calculate_defects_ti!(rseg_ti, useg_ti, duseg_ti, tdseg, disc)
                calculate_defects_td!(rseg_td, useg_td, duseg_td, tdseg, disc)
            end
        end
        # add residual equations corresponding to the boundary condition
        if fix_initial
            # boundary condition is on initial states
            r[1:nu, end] .= view(u, 1:nu, 1) .- ub
            # set derivatives of boundary condition expression
            r_u[1:nu, end, 1:nu, 1] .= 1
            r_ub[1:nu, end, 1:nu] .= -1
        else
            # boundary condition is on final states
            r[1,nu, end] .= view(u, 1:nu, size(u, 2)) .- ub
            # set derivatives of boundary condition expression
            r_u[1:nu, end, 1:nu, size(u, 2)] .= 1
            r_ub[1:nu, end, 1:nu] .= -1
        end
        # reshape calculated derivatives for more convenient use
        r_u = reshape(r_u, ncnodes*nu, :)
        r_c = reshape(r_c, ncnodes*nu, :)
        r_p = reshape(r_p, :, 1:np)
        r_ti = reshape(r_ti, :)
        r_td = reshape(r_td, :)
        r_ub = reshape(r_ub, :, 1:nu)
        # decompress state and control derivatives
        for iseg = 1:nsegs
            durange = (iseg-1)*nsegcnodes*nu : iseg*nsegcnodes*nu
            urange = (iseg-1)*nsegdnodes*nu : iseg*nsegdnodes*nu
            crange = (iseg-1)*nsegnodes*nc : iseg*nsegnodes*nc
            r_usys[durange, urange] = view(r_u, durange, :)
            r_csys[durange, crange] = view(r_c, durange, :)
        end
        # map calculated derivatives to system jacobian matrices
        mul!(drdy, r_usys, u_y)
        mul!(drdx, r_csys, c_x)
        mul!(drdx, r_p, p_x, 1, 1)
        mul!(drdx, r_ti, ti_x, 1, 1)
        mul!(drdx, r_td, td_x, 1, 1)
        mul!(drdx, r_ub, ub_x, 1, 1)
        return r, drdx, drdy
    end

    dfdx = let r = copy(r), drdy = copy(drdy)
        function(drdx, x, y)
            fdf(r, drdx, drdy, x, y)
            return drdx
        end
    end

    dfdy = let r = copy(r), drdx = copy(drdx)
        function(drdy, x, y)
            fdf(r, drdx, drdy, x, y)
            return drdy
        end
    end

    fdfdx = let drdy = copy(drdy)
        function(r, drdx, x, y)
            fdf(r, drdx, drdy, x, y)
            return r, drdx
        end
    end

    fdfdy = let drdx = copy(drdx)
        function(r, drdy, x, y)
            fdf(r, drdx, drdy, x, y)
            return r, drdy
        end
    end

    return ImplicitComponent(f, dfdx, dfdy, fdfdx, fdfdy, fdf, x_f, y_f, x_dfdx,
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, argin, argout)
end
