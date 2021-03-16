"""
    allocate_jacobian(x, y, sparsity=DensePattern())

Allocates a jacobian matrix given the inputs, outputs, and (optionally) the
sparsity structure.
"""
allocate_jacobian

allocate_jacobian(x, y) = allocate_jacobian(x, y, DensePattern())
allocate_jacobian(x, y, ::DensePattern) = NaN .* y .* x'
function allocate_jacobian(x, y, sp::SparsePattern)
    TF = promote_type(Float64, eltype(x), eltype(y))
    return sparse(sp.rows, sp.cols, fill(TF(NaN), length(sp.rows)))
end

"""
    ExplicitComponent([func,] fin, fout, foutin; kwargs...)

Construct a system component defined by the explicit function `func` with inputs
corresponding to `fin`, outputs corresponding to `fout`, and in-place
outputs corresponding to `foutin`.

# Arguments
 - `func`: Output function, of the form `(fout..., foutin...) = func(foutin..., fin...)`
 - `fin`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to inputs to `func`
 - `fout`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to outputs from `func`
 - `foutin`: Tuple of named variables (see [`Namedvar`](@ref)) corresponding to in-place function outputs from `func`

# Keyword Arguments
 - `f`: In-place output function `f(y, x)`. `x` is an input vector containing values
    corresponding to all of the variables in `component_inputs`. `y` is an output
    vector containing values corresponding to all of the variables in `component_outputs`.
 - `df`: In-place jacobian function `df(dydx, x)`.
 - `fdf`: In-place combined output and jacobian function `fdf(y, dydx, x)`.
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    the size and type of the jacobian is inferred.
 - `deriv`: Method used to calculate the jacobian if both `df` and `fdf` are not
    provided.
 - `sparsity = DensePattern()`: Defines the sparsity structure of the jacobian
    if `dydx` is not provided
 - `component_inputs`: Names of the inputs from `fin` which are also used as
    inputs to `f`. Defaults to all variables in `inputs`.
 - `component_outputs`: Names of the outputs from `fout` and `foutin` which are
    also used as outputs from `f`. Defaults to all variables in `fout` and `foutin`.
"""
function ExplicitComponent(func, fin, fout, foutin;
    f=nothing, df=nothing, fdf=nothing, dydx = nothing, deriv=ForwardFD(),
    sparsity = DensePattern(),
    component_inputs = name.(fin),
    component_outputs = (name.(fout)..., name.(foutin)...))

    # ensure the output function is defined
    @assert any((!isnothing(func), !isnothing(f), !isnothing(fdf))) "Output function not defined"

    # combine inputs and outputs
    combined_inputs = (fin...,)
    combined_outputs = (fout..., foutin...)

    # get default inputs and outputs
    default_inputs = value.(combined_inputs)
    default_outputs = value.(combined_outputs)

    # check that specified inputs/outputs correspond to actual inputs/outputs
    @assert all(in(name.(combined_inputs)), component_inputs)
    @assert all(in(name.(combined_outputs)), component_outputs)

    # get indices of component inputs (as a tuple)
    component_input_indices = (
        findall(in(component_inputs), name.(fin))...,
        )

    # get indices of component outputs (as a tuple)
    component_output_indices = (
        findall(in(component_outputs), name.(fout))...,
        findall(in(component_outputs), name.(foutin))...,
        )

    # named variables corresponding to component inputs/outputs
    argin = getindices(combined_inputs, component_input_indices)
    argout = getindices(combined_outputs, component_output_indices)

    # construct input and output vectors
    x0 = combine(argin)
    y0 = combine(argout)

    # ensure sizes of `x`, `y`, and `dydx` are compatibile
    @assert isnothing(dydx) || (length(y0) == size(dydx, 1) && length(x0) == size(dydx, 2))

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* x0
    x_df = NaN .* x0
    y = NaN .* y0
    dydx = isnothing(dydx) ? allocate_jacobian(x0, y0, sparsity) : NaN .* dydx

    # construct output function (if necessary)
    if isnothing(f)
        if !isnothing(func)
            # NOTE: Any changes to this function need to be tested for avoiding allocations
            # use `func` to construct `f`
            f = function(y, x)
                # get new component inputs/outputs
                new_component_inputs = separate(argin, x)
                new_component_outputs = separate(argout, y)
                # replace default inputs with new component inputs/outputs
                new_inputs = setindices(default_inputs, new_component_inputs, component_input_indices)
                new_combined_outputs = setindices(default_outputs, new_component_outputs, component_output_indices)
                # separate not-in-place and in-place outputs
                new_outputs = new_combined_outputs[1:length(fout)]
                new_inplace = new_combined_outputs[length(fout) + 1 : end]
                # call function to get new outputs
                new_combined_outputs = Tuple(func(new_inplace..., new_inputs...))
                # update output vector
                combine!(y, getindices(new_combined_outputs, component_output_indices))
                # return result
                return y
            end
        else
            # use `fdf` to construct `f`
            let dydx = copy(dydx)
                f = function(y, x)
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
        let y = copy(y)
            df = function(dydx, x)
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

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)
end

ExplicitComponent(fin, fout, foutin; kwargs...) = ExplicitComponent(nothing, fin, fout, foutin; kwargs...)

"""
    ExplicitComponent(component::AbstractImplicitComponent [, fout]; kwargs...)

Couple an implicit component with a solver to construct an explicit component.

 # Arguments:
 - `component`: Implicit component or system to convert to an explicit component
 - `fout`: (optional) Tuple of named variables (see [`NamedVar`](@ref))
    corresponding to outputs from the resulting explicit component

 # Keyword Arguments
 - `solver = Newton()`: Solver, either a function of the form `y = f(component, x, y0)`
        or an object of type [`AbstractSolver`](@ref)
 - `dydx`: Provides size and type of jacobian output, otherwise it will be
        allocated based on the inputs, outputs, and the sparsity structure in `sparsity`
 - `sparsity = DensePattern()`: Defines the sparsity structure of the jacobian if `dydx` is not provided
 - `mode`: Mode in which to compute the analytic sensitivity equations.  May be
    either [`Direct()`](@ref) or [`Adjoint()`](@ref).  Defaults to `Direct()` if
    the number of inputs is less than the number of outputs and `Adjoint()`
    otherwise.
"""
function ExplicitComponent(component::AbstractImplicitComponent, argout=component.argout; solver=Newton(),
    dydx = nothing, sparsity=DensePattern(), mode=nothing)

    # input variables
    argin = component.argin

    # state variables
    argstate = component.argout

    # initial guess comes from default values for state variables
    u0 = combine(argstate)

    # construct dfdx and dfdu
    dfdx, dfdu = output_mapping_matrices(argin, argstate, argout)

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* inputs(component) # inputs are the same as component
    x_df = NaN .* inputs(component) # inputs are the same as component
    y = NaN .* combine(argout) # outputs correspond to specified outputs
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y, sparsity) : NaN .* dydx

    # choose mode in which to apply analytic sensitivity equations (if not specified)
    if isnothing(mode)
        mode = ifelse(length(x_f) < length(y), Direct(), Adjoint())
    end

    # NOTE: xcache stores the current inputs to the solver
    # As xcache is updated, the function upon which the solver operates is
    # updated as well.
    xcache = similar(x_f)

    # we distinguish between mutating and non-mutating cache so that we can use
    # automatic differentiation on the non-mutating cache.
    cache = create_mutating_solver_cache(component, xcache, solver)

    # output function
    f = function(y, x)
        # determine whether to use internal cache
        if eltype(x) <: eltype(xcache)
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

    # function and partials
    fdf = function(y, dydx, x)
        # determine whether to use internal cache
        if eltype(x) <: eltype(xcache)
            # use existing cache if we can store the inputs
            copyto!(xcache, x)
            # call solver to get state variables
            u = solver(cache, x, u0)
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian!(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian!(component, x, u)
            # map inputs and state variables to outputs: y = dfdx*x + dfdu*u
            y = mul!(mul!(y, dfdx, x), dfdu, u, 1, 1)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # call solver to get state variables
            u = solver(new_cache, x, eltype(y).(u0))
            # get jacobian of the residual function with respect to the inputs
            drdx = residual_input_jacobian(component, x, u)
            # get jacobian of the residual function with respect to the outputs
            drdu = residual_output_jacobian(component, x, u)
            # map inputs and state variables to outputs: y = dfdx*x + dfdu*u
            y = mul!(mul!(y, dfdx, x), dfdu, u, 1, 1)
        end
        # get analytic sensitivities
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode))
        # return outputs
        return y, dydx
    end

    # partials only
    df = function(dydx, x)
        y, dydx = fdf(y, dydx, x)
        return dydx
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)
end

"""
    ExplicitComponent(component::AbstractImplicitComponent, output_component; kwargs...)

Couple an implicit component with a solver to construct an explicit component.

Reduce the dimensionality of the outputs by incorporating the output function
contained in `output_component`.  The vector of inputs to `output_component` are
assumed to contain the input variables to `component` followed by the
output/state variables of `component` .

 # Keyword Arguments
 - `solver = Newton()`: Solver, either a function of the form `y = f(component, x, y0)`
        or an object of type [`AbstractSolver`](@ref)
 - `dydx`: Provides size and type of jacobian output, otherwise it will be
        allocated based on the inputs, outputs, and the sparsity structure in `sparsity`
 - `sparsity = DensePattern()`: Defines the sparsity structure of the jacobian if `dydx` is not provided
 - `mode`: Mode in which to compute the analytic sensitivity equations.  May be
    either [`Direct()`](@ref) or [`Adjoint()`](@ref).  Defaults to `Direct()` if
    the number of inputs is less than the number of outputs and `Adjoint()`
    otherwise.
"""
function ExplicitComponent(component::AbstractImplicitComponent, output_component::AbstractExplicitComponent;
    solver=Newton(), u0=rand(length(component.y_f)), dydx = nothing,
    sparsity=DensePattern(), mode = nothing)

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
    dvdx, dvdu = output_mapping_matrices(argin, argstate, output_component.argin)

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* inputs(component)
    x_df = NaN .* inputs(component)
    y = NaN .* outputs(output_component)
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y, sparsity) : NaN .* dydx

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
        if eltype(x) <: eltype(xcache)
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
        if eltype(x) <: eltype(xcache)
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
        if eltype(x) <: eltype(xcache)
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
        return y, dydx
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx, argin, argout)
end

"""
    analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode)

Compute the sensitivities of an implicit system using the analytic sensitivity
equation in `Direct()` or `Adjoint()` mode.
"""
analytic_sensitivity_equation
analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, ::Direct) = dfdx - dfdu*(drdu\drdx)
analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, ::Adjoint) = dfdx - transpose(transpose(drdu)\transpose(dfdu))*drdx

"""
    ExplicitSystem(components, argin, argout; kwargs...)

Construct an explicit system component from a collection of explicit system
components.

# Arguments
- `components`: Collection of components
- `argin`: Tuple of system variables (see [`NamedVar`](@ref)) corresponding to system inputs
- `argout`: Tuple of system variables (see [`NamedVar`](@ref)) corresponding to system outputs

# Keyword Arguments
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    the jacobian matrix size and type will be infered from the inputs, outputs, and
    the sparsity structure.
 - `sparsity`: Sparsity structure of the system jacobian matrix
 - `mode`: Mode used to calculate system derivatives using the chain rule.  May
    be either [`Forward()`](@ref) or [`Reverse()`](@ref), defaults to
    [`Forward`](@ref) if the number of outputs exceeds the number of inputs and
    [`Reverse()`](@ref) otherwise.
"""
function ExplicitSystem(components, argin, argout;
    dydx = nothing, sparsity=DensePattern(), mode=nothing)

    # number of components
    nc = length(components)

    # construct input and output vectors
    x0 = combine(argin)
    y0 = combine(argout)

    # choose mode in which to apply analytic sensitivity equations (if not specified)
    if isnothing(mode)
        mode = ifelse(length(x0) < length(y0), Forward(), Reverse())
    end

    # construct system mapping
    component_input_mapping = system_component_mapping(components, argin)

    # construct output mapping
    output_mapping = system_output_mapping(components, argin, argout)

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
    dydx = isnothing(dydx) ? allocate_jacobian(x0, y0, sparsity) : NaN .* dydx

    # construct forward mode mapping
    input_mapping, component_output_mapping = forward_mode_mapping(x0,
        components, component_input_mapping, output_mapping)

    return ExplicitSystem(components, input_mapping, component_output_mapping,
        component_input_mapping, output_mapping, x_f, x_df, y, dydx, argin, argout, mode)
end

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

"""
    ImplicitComponent([func,] fin, fout [, r0]; kwargs...)

Construct a system component defined by the in-place implicit residual function
`func` with inputs corresponding to `fin` and outputs corresponding to `fout`.
The output variables are identical to the state variables of the component.

# Arguments
 - `func`: In-place residual function, of the form `residuals = func(r, fin..., fout...)`
 - `fin`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to function inputs
 - `fout`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to function outputs
 - `r0`: Vector which defines the size and type of the component's residuals.
    Defaults to the size and type of the output vector. It's length must
    correspond to the total number of component outputs/states.

# Keyword Arguments
 - `f`: In-place residual function `f(r, x, y)`. `x` is an input vector containing values
    corresponding to all of the variables in `component_inputs`. `y` is an output
    vector containing values corresponding to all of the variables in
    `component_outputs`.
 - `dfdx`: In-place residual jacobian function with respect to the inputs `dfdx(drdx, x, y)`
 - `dfdy`: In-place residual jacobian function with respect to the outputs `dfdy(drdy, x, y)`
 - `fdfdx`: In-place combined residual and jacobian with respect to the inputs function `fdfdx(r, drdx, x, y)`.
 - `fdfdy`: In-place combined residual and jacobian with respect to the outputs function `fdfdy(r, drdy, x, y)`.
 - `fdf`: In-place combined residual and jacobians function `fdf(r, drdx, drdy, x, y)`.
 - `drdx`: Matrix used to define size and type of the jacobian of the residual
    with respect to the inputs. If omitted, a dense jacobian is used.
 - `drdy`: Matrix used to define size and type of the jacobian of the residual
     with respect to the outputs. If omitted, a dense jacobian is used.
 - `xderiv`: Method used to calculate the residual jacobian with respect to the
    inputs if it is not provided. Defaults to [`ForwardFD`](@ref).
 - `yderiv`: Method used to calculate the residual jacobian with respect to the
    outputs if it is not provided. Defaults to [`ForwardFD`](@ref).
 - `xsparsity = DensePattern()`: Defines the sparsity structure of the jacobian
    with respect to the inputs if `drdx` is not provided
 - `ysparsity = DensePattern()`: Defines the sparsity structure of the jacobian
    with respect to the outputs if `drdy` is not provided
 - `component_inputs`: Names of the inputs from `fin` which are also used as
    inputs to `f`. Defaults to all variables in `fin`.
 - `component_outputs`: Names of the outputs from `fout` which are also used
    as outputs from `f`. Defaults to all variables in `fout`.
"""
function ImplicitComponent(func, fin, fout, r0=nothing;
    f=nothing, dfdx=nothing, dfdy=nothing, fdfdx=nothing, fdfdy=nothing, fdf=nothing,
    drdx=nothing, drdy=nothing, xderiv=ForwardFD(), yderiv=ForwardFD(),
    xsparsity = DensePattern(), ysparsity = DensePattern(),
    component_inputs = name.(fin), component_outputs = name.(fout))

    # ensure the residual function is defined
    @assert any((!isnothing(func), !isnothing(f), !isnothing(fdfdx), !isnothing(fdfdy), !isnothing(fdf))) "Residual function not defined"

    # get default inputs and outputs
    default_inputs = value.(fin)
    default_outputs = value.(fout)

    # check that specified argin/argout correspond to actual argin/argout
    @assert all(in(name.(fin)), component_inputs)
    @assert all(in(name.(fout)), component_outputs)

    # get indices of component inputs (as a tuple)
    component_input_indices = (
        findall(in(component_inputs), name.(fin))...,
        )

    # get indices of component outputs (as a tuple)
    component_output_indices = (
        findall(in(component_outputs), name.(fout))...,
        )

    # named variables corresponding to component inputs/outputs
    argin = getindices(fin, component_input_indices)
    argout = getindices(fout, component_output_indices)

    # construct input and output vectors
    x0 = combine(argin)
    y0 = combine(argout)
    r0 = isnothing(r0) ? similar(y0, promote_type(eltype(x0), eltype(y0))) : r0

    # ensure sizes of `r` and `y` are compatible
    @assert length(r0) == length(y0) "The length of the residual vector must match the number of outputs/states"

    # ensure sizes of `x`, `r`, and `drdx` are compatibile
    @assert isnothing(drdx) || (length(r0) == size(drdx, 1) && length(x0) == size(drdx, 2))

    # ensure sizes of `y`, `r`, and `drdy` are compatibile
    @assert isnothing(drdy) || (length(r0) == size(drdy, 1) && length(y0) == size(drdy, 2))

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* x0
    y_f = NaN .* y0
    x_dfdx = NaN .* x0
    y_dfdx = NaN .* y0
    x_dfdy = NaN .* x0
    y_dfdy = NaN .* y0
    r = NaN .* r0
    drdx = isnothing(drdx) ? allocate_jacobian(x0, r0) : drdx .* NaN
    drdy = isnothing(drdy) ? allocate_jacobian(y0, r0) : drdy .* NaN

    # construct residual function if necessary
    if isnothing(f)
        if !isnothing(func)
            # construct residual function from `func`
            f = function(r, x, y)
                # get new component inputs/outputs
                new_component_inputs = separate(argin, x)
                new_component_outputs = separate(argout, y)
                # replace default inputs with new component inputs/outputs
                new_inputs = setindices(default_inputs, new_component_inputs, component_input_indices)
                new_outputs = setindices(default_outputs, new_component_outputs, component_output_indices)
                # call function to update residual values
                func(r, new_inputs..., new_outputs...)
                # return result
                return r
            end
        elseif !isnothing(fdfdx)
            # construct residual function from fdfdx
            let drdx = copy(drdx)
                f = function(r, x, y)
                    fdfdx(r, drdx, x, y)
                    return y
                end
            end
        elseif !isnothing(fdfdy)
            # construct residual function from fdfdy
            let drdy = copy(drdy)
                f = function(r, x, y)
                    fdf(r, drdy, x, y)
                    return y
                end
            end
        else
            # construct residual function from fdf
            let drdx = copy(drdx), drdy = copy(drdy)
                f = function(r, x, y)
                    fdf(r, drdx, drdy, x, y)
                    return y
                end
            end
        end
    end

    # construct jacobian functions with respect to the inputs
    if isnothing(dfdx) && isnothing(fdfdx) && isnothing(fdf)
        # construct dfdx and fdfdx from f
        dfdx, fdfdx = create_residual_input_jacobian_functions(f, x_f, y_f, r, xderiv, xsparsity)
    elseif isnothing(dfdx) && isnothing(fdfdx)
        # construct dfdx from fdf
        let r = copy(r), drdy = copy(drdy)
            dfdx = function(drdx, x, y)
                fdf(r, drdx, drdy, x, y)
                return drdx
            end
        end

        # construct fdfdx from fdf
        let drdy = copy(drdy)
            fdfdx = function(r, drdx, x, y)
                fdf(r, drdx, drdy, x, y)
                return r, drdx
            end
        end
    elseif isnothing(dfdx)
        # construct dfdx from fdfdx
        let r = copy(r)
            dfdx = function(drdx, x, y)
                fdfdx(r, drdx, x, y)
                return drdx
            end
        end
    elseif isnothing(fdfdx)
        # construct fdfdx from f and dfdx
        fdfdx = function(r, drdx, x, y)
            f(r, x, y)
            dfdx(drdx, x, y)
            return r, drdx
        end
    end

    # construct jacobian functions with respect to the outputs
    if isnothing(dfdy) && isnothing(fdfdy) && isnothing(fdf)
        # construct dfdy and fdfdy from f
        dfdy, fdfdy = create_residual_output_jacobian_functions(f, x_f, y_f, r, yderiv, ysparsity)
    elseif isnothing(dfdy) && isnothing(fdfdy)
        # construct dfdy from fdf
        let r = copy(r), drdx = copy(drdx)
            dfdy = function(drdy, x, y)
                fdf(r, drdx, drdy, x, y)
                return drdy
            end
        end

        # construct fdfdy from fdf
        let drdx = copy(drdx)
            fdfdy = function(r, drdy, x, y)
                fdf(r, drdx, drdy, x, y)
                return r, drdy
            end
        end
    elseif isnothing(dfdy)
        # construct dfdy from fdfdy
        let r = copy(r)
            dfdy = function(drdy, x, y)
                fdfdy(r, drdy, x, y)
                return drdy
            end
        end
    elseif isnothing(fdfdy)
        # construct fdfdy from f and dfdy
        fdfdy = function(r, drdy, x, y)
            f(r, x, y)
            dfdy(drdy, x, y)
            return r, drdy
        end
    end

    # construct combined output and jacobians function if necessary
    if isnothing(fdf)
        # construct fdf from dfdx and fdfdy
        fdf = function(r, drdx, drdy, x, y)
            dfdx(drdx, x, y)
            fdfdy(r, drdy, x, y)
            return r, drdx, drdy
        end
    end

    return ImplicitComponent(f, dfdx, dfdy, fdfdx, fdfdy, fdf, x_f, y_f, x_dfdx,
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, argin, argout)
end

ImplicitComponent(fin, fout, args...; kwargs...) = ImplicitComponent(nothing, fin, fout, args...; kwargs...)

make_implicit(component::ExplicitComponent) = ImplicitComponent(component)
make_implicit(component::ExplicitSystem) = ImplicitComponent(component)
make_implicit(component::ImplicitComponent) = component
make_implicit(component::ImplicitSystem) = component

"""
    ImplicitComponent(component::ExplicitComponent)

Constructs an implicit system component from an explicit system component.
"""
function ImplicitComponent(component::ExplicitComponent)

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* component.x_f
    y_f = NaN .* component.y
    x_dfdx = NaN .* component.x_f
    y_dfdx = NaN .* component.y
    x_dfdy = NaN .* component.x_f
    y_dfdy = NaN .* component.y
    r = NaN .* component.y
    drdx = NaN .* component.dydx
    drdy = Matrix(-I, length(r), length(y_f))

    f = function(r, x, y)
        # update outputs, store in residual
        outputs!(component, r, x)
        # subtract provided outputs
        r .-= y
        # return residual
        return r
    end

    dfdx = (drdx, x, y) -> jacobian!(component, drdx, x)

    fdfdx = function(r, drdx, x, y)
        # update outputs, store in residual
        outputs_and_jacobian!(component, r, drdx, x)
        # subtract provided outputs
        r .-= y
        # return residual
        return r
    end

    dfdy = function(drdy, x, y)
        # negative identity matrix
        drdy .= 0
        for i = 1:length(y)
            drdy[i,i] = -1
        end
    end

    fdfdy = function(r, drdy, x, y)
        f(r, x, y)
        dfdy(drdy, x, y)
        return r, drdy
    end

    fdf = function(r, drdx, drdy, x, y)
        fdfdx(r, drdx, x, y)
        dfdy(drdy, x, y)
        return r, drdx, drdy
    end

    # input/output arguments are preserved
    argin = component.argin
    argout = component.argout

    return ImplicitComponent(f, dfdx, dfdy, fdfdx, fdfdy, fdf, x_f, y_f, x_dfdx,
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, argin, argout)
end

"""
    ImplicitSystem(components, argin; kwargs...)

Constructs an implicit system component from a collection of implicit components.

# Arguments
 - `components`: Collection of components
 - `argin`: Tuple of system variables (see [`NamedVar`](@ref)) corresponding to system inputs

# Keyword Arguments
- `drdx`: Matrix used to define size and type of the jacobian of the residual
   with respect to the inputs.  If omitted, the jacobian matrix size and type
   will be inferred from the inputs, outputs, and the sparsity structure.
- `drdy`: Matrix used to define size and type of the jacobian of the residual
    with respect to the outputs.  If omitted, the jacobian matrix size and type
    will be infered from the inputs, outputs, and the sparsity structure.
- `xsparsity`: Sparsity structure of the jacobian of the residual with respect
    to the inputs. Defaults to [`DensePattern()`](@ref).
- `ysparsity`: Sparsity struction of the jacobian of the residual with respect
    to the outputs.  Defaults to [`DensePattern()`](@ref).
"""
function ImplicitSystem(components, argin;
    drdx=nothing, drdy=nothing, xsparsity=DensePattern(), ysparsity=DensePattern())

    # number of components
    nc = length(components)

    # convert all components to implicit components
    components = make_implicit.(components)

    # construct input vector
    x0 = combine(argin)

    # construct system mapping
    component_input_mapping = system_component_mapping(components, argin)

    # check that there is an input mapping for each component
    @assert length(components) == length(component_input_mapping) "There must be an input mapping for each component"
    for ic = 1:nc
        # check that there is an input mapping for each input to the component
        @assert length(components[ic].x_f) == length(component_input_mapping[ic]) "Input mapping for component $ic must have the same length as the number of inputs to component $ic"
        for ix = 1:length(component_input_mapping[ic])
            (jc, jy) = component_input_mapping[ic][ix]
            if iszero(jc)
                @assert jy <= length(x0) "Input $ix of "*
                    "component $ic corresponds to non-existant system input $jy"
            else
                # check that each component input only uses available component outputs
                @assert jc <= nc "Input $ix of component $ic comes from "*
                    "non-existant component $jc"
                @assert jy <= length(outputs(components[jc])) "Input $ix of component $ic "*
                    "corresponds to non-existant output $jx of component $jc"
            end
        end
    end

    # allocate storage and initialize with NaNs (since values are not yet defined)
    x_f = x0 .* NaN
    y_f = vcat(outputs.(components)...) .* NaN
    x_dfdx = copy(x_f)
    y_dfdx = copy(y_f)
    x_dfdy = copy(x_f)
    y_dfdy = copy(y_f)
    r = copy(y_f)
    drdx = isnothing(drdx) ? allocate_jacobian(x_f, r, xsparsity) : drdx .* NaN
    drdy = isnothing(drdy) ? allocate_jacobian(y_f, r, ysparsity) : drdy .* NaN

    # ensure jacobian sizes are compatibile
    @assert (length(r) == size(drdx, 1) && length(x_f) == size(drdx, 2))
    @assert (length(r) == size(drdy, 1) && length(y_f) == size(drdy, 2))

    # system outputs are all of the component outputs concatenated
    argout = (Iterators.flatten([comp.argout for comp in components])...,)

    # construct index for accessing outputs/residuals for each component
    idx = vcat(0, cumsum(length.(residuals.(components)))...)

    return ImplicitSystem(Tuple(components), collect(component_input_mapping),
        x_f, y_f, x_dfdx, y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, argin, argout, idx)
end
