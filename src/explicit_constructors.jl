############################################
# --- Explicit Component from Function --- #
############################################

"""
    ExplicitComponent([func,] fin, fout, foutin; kwargs...)

Construct a system component defined by the explicit function `func` with inputs
corresponding to `fin`, outputs corresponding to `fout`, and in-place outputs
corresponding to `foutin`.

# Arguments
 - `func`: Template output function, of the form `(fout..., foutin...) = func(foutin..., fin...)`.
 - `fin`: Tuple of named or unnamed variables (see [`NamedVar`](@ref)) corresponding to inputs to `func`
 - `fout`: Tuple of named or unnamed variables (see [`NamedVar`](@ref)) corresponding to outputs from `func`
 - `foutin`: Tuple of named or unnamed variables (see [`NamedVar`](@ref)) corresponding to in-place function outputs from `func`

# Keyword Arguments
 - `f`: In-place output function `f(y, x)`. `x` is an input vector containing values
    corresponding to all of the named variables in `fin`. `y` is an output
    vector containing values corresponding to all of the named variables in
    `fout` and `foutin`.
 - `df`: In-place jacobian function `df(dydx, x)`.
 - `fdf`: In-place combined output and jacobian function `fdf(y, dydx, x)`.
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    a dense jacobian matrix is used.
 - `deriv`: Method used to calculate the jacobian if both `df` and `fdf` are not
    provided.
 - `sparsity`: Sparsity structure used to to calculate the jacobian if both `df`
    and `fdf` are not provided.

Note that it is not necessary to specify the template output function `func` if
the vectorized form of the output function has been provided through the keyword
arguments `f` or `fdf`.
"""
function ExplicitComponent(func, fin, fout, foutin; f=nothing, df=nothing,
    fdf=nothing, dydx = nothing, deriv=ForwardFD(), sparsity = DensePattern())

    # ensure the output function is defined
    @assert any((!isnothing(func), !isnothing(f), !isnothing(fdf))) "Output function not defined"

    # template function input/output arguments
    tin = (fin...,)
    tout = (fout..., foutin...)

    # template function default input/output values
    tin_val = get_values(tin)
    tout_val = get_values(tout)

    # component input/output argument indices (as a tuple)
    cin_idx = (findall(in(get_names(tin)), get_name.(tin))...,)
    cout_idx = (findall(in(get_names(tout)), get_name.(tout))...,)

    # component input/output arguments
    cin = getindices(tin, cin_idx)
    cout = getindices(tout, cout_idx)

    # component default input/output values
    cin_val = get_values(cin)
    cout_val = get_values(cout)

    # default component input/output vectors
    x0 = combine(cin_val)
    y0 = combine(cout_val)

    # ensure sizes of `x`, `y`, and `dydx` are compatibile
    @assert isnothing(dydx) || (length(y0) == size(dydx, 1) && length(x0) == size(dydx, 2))

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* x0
    x_df = NaN .* x0
    y = NaN .* y0
    dydx = isnothing(dydx) ? allocate_jacobian(x0, y0) : NaN .* dydx

    # construct output function (if necessary)
    if isnothing(f)
        if !isnothing(func)
            # construct residual function from `func`
            f = function(y, x)
                # expand component inputs and outputs to provided sizes/types
                fin_val = separate!(cin_val, x)
                fout_val = separate!(cout_val, y)
                # insert component inputs/outputs into template inputs/outputs
                fin_val = setindices(tin_val, fin_val, cin_idx)
                fout_val = setindices(tout_val, fout_val, cout_idx)
                # extract inplace outputs from general outputs
                foutin_val = fout_val[length(fout) + 1 : end]
                # call template function
                fout_val = Tuple(func(foutin_val..., fin_val...))
                # extract component outputs from the general outputs
                fout_val = getindices(fout_val, cout_idx)
                # combine the outputs into a single vector
                combine!(y, fout_val)
                # return the result
                return y
            end
        else
            # use `fdf` to construct `f`
            f = let dydx = copy(dydx)
                function(y, x)
                    fdf(y, dydx, x)
                    return y
                end
            end
        end
    end

    # construct jacobian functions (if necessary)
    if isnothing(df) && isnothing(fdf)
        # construct df and fdf functions from f
        df, fdf = create_output_jacobian_functions(f, x0, y0, deriv, sparsity)
    elseif isnothing(df)
        # construct df function from fdf
        df = let y = copy(y)
            function(dydx, x)
                fdf(y, dydx, x)
                return dydx
            end
        end
    elseif isnothing(fdf)
        # construct fdf function from f and df
        fdf = function(y, dydx, x)
            f(y, x)
            df(dydx, x)
            return y, dydx
        end
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, cin, cout)
end

ExplicitComponent(fin, fout, foutin; kwargs...) = ExplicitComponent(nothing, fin, fout, foutin; kwargs...)

###############################################################
# --- Explicit Component from Implicit Component --- #
###############################################################

"""
    make_explicit(component::AbstractImplicitComponent, argout)

Construct an explicit component from an implicit component with outputs
corresponding to the implicit component's residuals and inputs corresponding to
the implicit component's inputs and outputs.

 # Arguments:
 - `component`: Implicit component (or system) to be converted
 - `argout`: Named variables corresponding to the values of the residuals
"""
function make_explicit(component::AbstractImplicitComponent, argout;
    dydx = nothing)

    # component input arguments
    argin = (component.argin..., component.argout...)

    # component default input/output values
    argin_val = get_values(argin)
    argout_val = get_values(argout)

    # default component input/output vectors
    x0 = combine(argin_val)
    y0 = combine(argout_val)

    # allocate storage and initialize with NaNs
    x_f = NaN .* x0
    x_df = NaN .* x0
    y = NaN .* y0
    dydx = NaN .* hcat(component.drdx, component.drdy)

    # number of inputs and outputs from the implicit component
    nin = length(component.x_f)
    nout = length(component.y_f)

    # storage types
    TY = eltype(y)
    TDY = eltype(dydx)

    # output function
    f = function(y, x)
        TF = promote_type(eltype(y), TY)
        # determine whether to use the internal cache
        if TF <: TY
            # use existing cache because we can store the outputs
            copyto!(y, residuals!(component, view(x, 1:nin), view(x, nin+1:nin+nout)))
        else
            # use new cache because we can't store the outputs
            residuals!(component, y, view(x, 1:nin), view(x, nin+1:nin+nout))
        end
        # return results
        return y
    end

    # jacobian function
    df = function(dydx, x)
        TF = promote_type(eltype(dydx), TDY)
        # split inputs into subcomponent inputs and outputs
        xs, ys = view(x, 1:nin), view(x, nin+1:nin+nout)
        # split output jacobian into subcomponent jacobians
        drdx, drdy = view(dydx, :, 1:nin), view(dydx, :, nin+1:nin+nout)
        # determine whether to use the internal cache
        if TF <: TDY
            # use existing cache
            copyto!(drdx, residual_input_jacobian!(component, xs, ys))
            copyto!(drdy, residual_output_jacobian!(component, xs, ys))
        else
            # use new cache
            residual_input_jacobian!(component, drdx, xs, ys)
            residual_output_jacobian!(component, drdy, xs, ys)
        end
        # return results
        return dydx
    end

    # function and partials
    fdf = function(y, dydx, x)
        TF1 = promote_type(eltype(y), TY)
        TF2 = promote_type(eltype(dydx), TDY)
        # split inputs into subcomponent inputs and outputs
        xs, ys = view(x, 1:nin), view(x, nin+1:nin+nout)
        # split output jacobian into subcomponent jacobians
        drdx, drdy = view(dydx, :, 1:nin), view(dydx, :, nin+1:nin+nout)
        # determine whether to use the internal cache
        if (TF1 <: TY) && (TF2 <: TDY)
            # use existing cache
            outputs = residuals_and_jacobians!(component, xs, ys)
            copyto!(y, outputs[1])
            copyto!(drdx, outputs[2])
            copyto!(drdy, outputs[3])
        else
            # use new cache
            residuals_and_jacobians!(component, y, drdx, drdy, xs, ys)
        end
        # return results
        return y, dydx
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)
end

###############################################################
# --- Explicit Component from Implicit Component + Solver --- #
###############################################################

"""
    add_solver(component::AbstractImplicitComponent [, fout]; kwargs...)

Couple an implicit component (or system) with a solver to construct an explicit
component.

 # Arguments:
 - `component`: Implicit component (or system) to be converted
 - `fout`: (optional) Names of the state variables in `component` to be
    used as outputs from the newly created explicit component.  Defaults to all
    state variables in `component`

 # Keyword Arguments
 - `solver = Newton()`: Linear or nonlinear solver. May be either a function of the
        form `y = f(component, x, y0)` or an object of type [`AbstractSolver`](@ref)
 - `dydx`: Provides size and type information for the jacobian output.  If not
        specified, the size and type will be inferred.
 - `mode`: Mode in which to compute the analytic sensitivity equations.  May be
    either [`Direct()`](@ref) or [`Adjoint()`](@ref).  Defaults to `Direct()` if
    the number of inputs is less than the number of outputs and `Adjoint()`
    otherwise.
"""
function add_solver(component::AbstractImplicitComponent,
    fout::NTuple{N, Symbol} = get_names(component.argout);
    solver=Newton(), dydx = nothing, mode = nothing) where N

    # component inputs, outputs, and state variable arguments
    argin = component.argin
    argout = component.argout[findall(in(fout), get_name.(component.argout))]
    argstate = component.argout

    # default component input/output vectors
    x0 = combine(argin)
    y0 = combine(argout)
    u0 = combine(argstate)

    # mapping matrices from inputs and state variables to outputs
    dfdx, dfdu = mapping_matrices(argin, argstate, argout)

    # choose mode in which to apply analytic sensitivity equations (if not specified)
    if isnothing(mode)
        mode = ifelse(length(x0) < length(y0), Direct(), Adjoint())
    end

    # allocate storage and initialize with NaNs
    x_f = NaN .* x0
    x_df = NaN .* x0
    y = NaN .* y0
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y) : NaN .* dydx

    # initialize cache variables
    xcache = similar(x_f)
    cache = create_mutating_solver_cache(component, xcache, solver)

    # NOTE: xcache stores a reference to the current inputs to the solver and
    # is updated as necessary

    # output function
    f = function(y, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # call solver to get state variables
            u = solver(new_cache, x, eltype(y).(u0))
        end
        # map inputs and state variables to outputs: y = dfdx*x + dfdu*u
        y = mul!(mul!(y, dfdx, x), dfdu, u, 1, 1)
        return y
    end

    # partials only
    df = function(dydx, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian!(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian!(component, x, u)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # call solver to get state variables
            u = solver(new_cache, x, TF.(u0))
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian(component, x, u)
        end
        # get analytic sensitivities
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode))
        # return outputs
        return dydx
    end

    # function and partials
    fdf = function(y, dydx, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian!(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian!(component, x, u)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # call solver to get state variables
            u = solver(new_cache, x, eltype(y).(u0))
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian(component, x, u)
        end
        # map inputs and state variables to outputs: y = dfdx*x + dfdu*u
        y = mul!(mul!(y, dfdx, x), dfdu, u, 1, 1)
        # get analytic sensitivities
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode))
        # return outputs
        return y, dydx
    end

    xcomp = ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)

    # run now if the component is not dependent on the inputs
    if isempty(x_f)
        outputs_and_jacobian!!!(xcomp, x_f)
    end

    return xcomp
end

#################################################################################
# --- Explicit Component from Implicit Component + Solver + Output Equation --- #
#################################################################################

"""
    add_solver(component::AbstractImplicitComponent, output_component; kwargs...)

Couple an implicit component (or system) with a solver and an output equation to
construct an explicit component.

# Arguments:
 - `component`: Implicit component (or system) to be converted
 - `output_component`: Explicit component which takes as inputs a subset of the
    inputs and state variables in `component` and returns a new set of outputs.

 # Keyword Arguments
 - `solver = Newton()`: Linear or nonlinear solver. May be either a function of the
        form `y = f(component, x, y0)` or an object of type [`AbstractSolver`](@ref)
 - `dydx`: Provides size and type information for the jacobian output.  If not
        specified, the size and type will be inferred.
 - `mode`: Mode in which to compute the analytic sensitivity equations.  May be
        either [`Direct()`](@ref) or [`Adjoint()`](@ref).  Defaults to `Direct()` if
        the number of inputs is less than the number of outputs and `Adjoint()`
        otherwise.
"""
function add_solver(component::AbstractImplicitComponent,
    output_component::AbstractExplicitComponent; solver=Newton(),
    dydx = nothing, mode = nothing)

    # input variables
    argin = component.argin

    # state variables
    argstate = component.argout

    # output variables
    argout = output_component.argout

    # initial guess comes from default values for state variables
    u0 = combine(argstate)

    # pre-allocated storage vector for inputs to output function
    v = combine(output_component.argin)

    # pre-allocated storage matrices for jacobians of output function
    dfdx = outputs(output_component) * inputs(component)' .* NaN
    dfdu = outputs(output_component) * outputs(component)' .* NaN

    # construct dvdx and dvdu where `v` is a vector of inputs to `output_component`
    dvdx, dvdu = mapping_matrices(argin, argstate, output_component.argin)

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* inputs(component)
    x_df = NaN .* inputs(component)
    y = NaN .* outputs(output_component)
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y) : NaN .* dydx

    # choose mode in which to apply analytic sensitivity equations (if not specified)
    if isnothing(mode)
        mode = ifelse(length(x_f) < length(y), Direct(), Adjoint())
    end

    # input and state variable dimensions
    nx = length(component.x_f)
    nu = length(component.y_f)

    # NOTE: xcache stores the current inputs to the solver
    # As xcache is updated, the function upon which the solver operates is
    # updated as well.
    xcache = similar(x_f)

    # we distinguish between mutating and non-mutating cache so that we can
    # use different types when using the non-mutating cache
    cache = create_mutating_solver_cache(component, xcache, solver)

    # output function
    f = function(y, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = mul!(mul!(v, dvdx, x), dvdu, u, 1, 1)
            # get outputs from output function
            copyto!(y, outputs!(output_component, v))
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # call solver to get state variables
            u = solver(new_cache, x, eltype(y).(u0))
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = dvdx*x + dvdu*u
            # get outputs from output function
            copyto!(y, outputs(output_component, v))
        end
        # return result
        return y
    end

    # partials
    df = function(dydx, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = mul!(mul!(v, dvdx, x), dvdu, u, 1, 1)
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian!(component, x, u)
            # get jacobian of the residual function with respect to the state variables
            drdu = residual_output_jacobian!(component, x, u)
            # get jacobian of the output component with respect to its inputs
            dfdv = jacobian!(output_component, v)
            # extract output jacobian corresponding to inputs
            dfdx = mul!(dfdx, dfdv, dvdx)
            # extract output jacobian corresponding to state variables
            dfdu = mul!(dfdu, dfdv, dvdu)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # u0 might not have the right type so we convert it
            u = solver(new_cache, x, eltype(x).(u0))
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian(component, x, u)
            # get jacobian of the residual function with respect to the state variables
            drdu = residual_output_jacobian(component, x, u)
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = dvdx*x + dvdu*u
            # get jacobian of the output component with respect to its inputs
            dfdv = jacobian(output_component, v)
            # extract output jacobian corresponding to inputs
            dfdx = dfdv*dvdx
            # extract output jacobian corresponding to state variables
            dfdu = dfdv*dvdu
        end
        # apply analytic sensitivity equation in direct or adjoint mode
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode))
        return dydx
    end

    fdf = function(y, dydx, x)
        TF = promote_type(eltype(x), eltype(xcache))
        # determine whether to use internal cache
        if TF <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = mul!(mul!(v, dvdx, x), dvdu, u, 1, 1)
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian!(component, x, u)
            # get jacobian of the residual function with respect to the state variables
            drdu = residual_output_jacobian!(component, x, u)
            # get jacobian of the output component with respect to its inputs
            y[:], dfdv = outputs_and_jacobian!(output_component, v)
            # extract output jacobian corresponding to inputs
            dfdx = mul!(dfdx, dfdv, dvdx)
            # extract output jacobian corresponding to state variables
            dfdu = mul!(dfdu, dfdv, dvdu)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # u0 might not have the right type so we convert it
            u = solver(new_cache, x, eltype(x).(u0))
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian(component, x, u)
            # get jacobian of the residual function with respect to the state variables
            drdu = residual_output_jacobian(component, x, u)
            # map inputs and state variables to output function inputs: v = dvdx*x + dvdu*u
            v = dvdx*x + dvdu*u
            # get jacobian of the output component with respect to its inputs
            y[:], dfdv = outputs_and_jacobian(output_component, v)
            # extract output jacobian corresponding to inputs
            dfdx = dfdv*dvdx
            # extract output jacobian corresponding to state variables
            dfdu = dfdv*dvdu
        end
        # apply analytic sensitivity equation in direct or adjoint mode
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode))
        # return results
        return y, dydx
    end

    xcomp = ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)

    # run now if the component is not dependent on the inputs
    if isempty(x_f)
        outputs_and_jacobian!!!(xcomp, x_f)
    end

    return xcomp
end

########################################################################
# --- Explicit System from Sequentially Called Explicit Components --- #
########################################################################

"""
    ExplicitSystem(components, argin, argout; kwargs...)

Construct an explicit system component from a chain of explicit system
components, called sequentially.

# Arguments
- `components`: Collection of explicit components, in calling order
- `argin`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to
    component inputs
- `argout`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to
    component outputs

# Keyword Arguments
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    a dense jacobian matrix will be used
 - `mode`: Mode used to calculate system derivatives using the chain rule.  May
    be either [`Forward()`](@ref) or [`Reverse()`](@ref), defaults to
    [`Forward`](@ref) if the number of outputs exceeds the number of inputs and
    [`Reverse()`](@ref) otherwise.
"""
function ExplicitSystem(components, argin, argout; dydx = nothing, mode=nothing)

    # number of components
    nc = length(components)

    # construct input and output vectors
    x0 = combine(argin)
    y0 = combine(argout)

    # choose mode in which to apply analytic sensitivity equations (if not specified)
    if isnothing(mode)
        mode = ifelse(length(x0) < length(y0), Forward(), Reverse())
    end

    # construct reverse-mode mapping
    component_input_mapping = system_component_mapping(argin, components)
    output_mapping = system_output_mapping(components, argin, argout)

    # construct forward-mode mapping
    input_mapping, component_output_mapping = forward_mode_mapping(x0,
        components, component_input_mapping, output_mapping)

    # ensure all components are explicit
    @assert all(isa.(components, AbstractExplicitComponent)) "All components of an explicit system must be explicit"

    # ensure sizes of `x`, `y`, and `dydx` are compatibile
    @assert isnothing(dydx) || (length(y0) == size(dydx, 1) && length(x0) == size(dydx, 2))

    # check that there is an input mapping for each component
    @assert length(components) == length(component_input_mapping) "There must be an input mapping for each component"
    for ic = 1:nc
        # check that there is an input mapping for each input to the component
        @assert length(components[ic].x_f) == length(component_input_mapping[ic]) "Input mapping for component $ic must have the same length as the number of inputs to component $ic"
        for ix = 1:length(component_input_mapping[ic])
            (jc, jy) = component_input_mapping[ic][ix]
            if iszero(jc)
                # check that component input uses available system input
                @assert jy <= length(x0) "Input $ix of "*
                    "component $ic corresponds to non-existant system input $jy"
            else
                # check that each component input only uses available component outputs
                @assert jc <= nc "Input $ix of component $ic comes from "*
                    "non-existant component $jc"
                @assert jc < ic "Implicit dependency in explicit system: Input $ix "*
                    "of component $ic depends on the output of component $jc"
                @assert jy <= length(outputs(components[jc])) "Input $ix of component $ic "*
                    "corresponds to non-existant output $jx of component $jc"
            end
        end
        for iy = 1:length(output_mapping)
            (jc, jy) = output_mapping[iy]
            if iszero(jc)
                # check that system output uses available system inputs
                @assert jy <= length(x0) "System output $iy "*
                    "corresponds to non-existant system input $jy"
            else
                # check that each component input only uses available component outputs
                @assert jc <= nc "System output $iy comes from non-existant "
                    "component $jc"
                @assert jy <= length(outputs(components[jc])) "System output $iy "*
                    "corresponds to non-existant output $jy of component $jc"
            end
        end
    end

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* x0
    x_df = NaN .* x0
    y = NaN .* y0
    dydx = isnothing(dydx) ? allocate_jacobian(x0, y0) : NaN .* dydx

    return ExplicitSystem(components, input_mapping, component_output_mapping,
        component_input_mapping, output_mapping, x_f, x_df, y, dydx, argin, argout, mode)
end

####################################################################
# --- Auxiliary Functions for Constructing Explicit Components --- #
####################################################################

"""
    forward_mode_mapping(inputs, components, component_input_mapping, output_mapping)

Returns (`input_mapping, component_output_mapping`) where `input_mapping` contains
a mapping from each system input to component inputs and/or system outputs and
`reversed_component_mapping` contains a mapping from each component output to
corresponding component inputs and/or system outputs.
"""
function forward_mode_mapping(inputs, components, component_input_mapping, output_mapping)

    # get number of inputs and components
    nx = length(inputs)
    nc = length(components)

    # initialize input mapping
    input_mapping = Vector{NTuple{2, Vector{Int}}}(undef, nx)
    # populate with empty arrays
    for ix = 1:nx
        input_mapping[ix] = (Int[], Int[])
    end

    # initialize reversed component mapping (outputs to inputs)
    component_output_mapping = Vector{Vector{NTuple{2, Vector{Int}}}}(undef, nc)
    for ic = 1:nc
        # number of outputs from this component
        ncy = length(outputs(components[ic]))
        # initialize output to input mapping for this component
        component_output_mapping[ic] = Vector{NTuple{2, Vector{Int}}}(undef, ncy)
        # populate with empty arrays
        for icy = 1:ncy
            component_output_mapping[ic][icy] = (Int[], Int[])
        end
    end

    # add mappings from component_mapping
    for jc = 1:nc
        # number of inputs from this component
        ncx = length(component_input_mapping[jc])
        for jcx = 1:ncx
            ic, icy = component_input_mapping[jc][jcx]
            if iszero(ic)
                ix = icy
                push!(input_mapping[ix][1], jc)
                push!(input_mapping[ix][2], jcx)
            else
                push!(component_output_mapping[ic][icy][1], jc)
                push!(component_output_mapping[ic][icy][2], jcx)
            end
        end
    end

    # add mappings from output_mapping
    ny = length(output_mapping)
    for iy = 1:ny
        ic, icy = output_mapping[iy]
        if iszero(ic)
            ix = icy
            push!(input_mapping[ix][1], nc+1)
            push!(input_mapping[ix][2], iy)
        else
            push!(component_output_mapping[ic][icy][1], nc+1)
            push!(component_output_mapping[ic][icy][2], iy)
        end
    end

    return input_mapping, component_output_mapping
end
