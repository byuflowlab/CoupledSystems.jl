"""
    allocate_jacobian(x, y, sparsity=DensePattern())

Allocates a jacobian matrix given the inputs, outputs, and (optionally) the
sparsity structure.
"""
allocate_jacobian

allocate_jacobian(x, y) = alloc_J(x, y, DensePattern())
allocate_jacobian(x, y, ::DensePattern) = NaN .* y .* x'
function allocate_jacobian(x, y, sp::SparsePattern)
    TF = promote_type(Float64, eltype(x), eltype(y))
    return sparse(sp.rows, sp.cols, fill(TF(NaN), length(sp.rows)))
end

"""
    ExplicitComponent(x0, y0; kwargs...)

Construct a system component defined by the explicit vector-valued output
function: `y = f(x)`.

# Arguments
 - `x0`: Initial values for inputs (used for size and type information)
 - `y0`: Initial values for outputs (used for size and type information)

# Keyword Arguments
 - `f`: In-place output function `f(y, x)`.
 - `df`: In-place jacobian function `df(dydx, x)`.
 - `fdf`: In-place combined output and jacobian function `fdf(y, dydx, x)`.
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    the jacobian matrix size and type will be infered from the inputs, outputs, and
    the sparsity structure.
 - `deriv`: Method used to calculate the jacobian if it is not provided.
    Defaults to forward finite differencing.
 - `sparsity`: Sparsity structure of the jacobian. Defaults to [`DensePattern()`](@ref)

 **Either the `f` or `fdf` keyword arguments must be provided**
"""
function ExplicitComponent(x0, y0; f=nothing, df=nothing,
    fdf=nothing, dydx = nothing, deriv=ForwardFD(), sparsity=DensePattern())

    # ensure `f` is defined
    @assert any((!isnothing(f), !isnothing(fdf))) "Output function not defined"

    # ensure sizes of `x`, `y`, and `dydx` are compatibile
    @assert isnothing(dydx) || (length(y0) == size(dydx, 1) && length(x0) == size(dydx, 2))

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    # NOTE: We alias the inputs together when the only provided method of computation
    # is simultaneous computation
    x_f = NaN .* x0
    x_df = isnothing(f) && isnothing(df) ? x_f : NaN .* x0
    y = NaN .* y0
    dydx = isnothing(dydx) ? allocate_jacobian(x0, y0, sparsity) : NaN .* dydx

    # construct output function if necessary
    if isnothing(f)
        # construct output function from fdf
        # NOTE: This updates dydx as well, see above note about aliasing
        f = function(y, x)
            fdf(y, dydx, x)
            return y
        end
    end

    # construct jacobian functions if necessary
    if isnothing(df) && isnothing(fdf)
        # construct df and fdf functions from f
        df, fdf = create_output_jacobian_functions(f, x0, y0, deriv, sparsity)
    elseif isnothing(df)
        # construct df function from fdf
        # NOTE: this function updates `y` as well, see above note about aliasing
        df = function(dydx, x)
            fdf(y, dydx, x)
            return y, dydx
        end
    elseif isnothing(fdf)
        # construct fdf function from f and df
        fdf = function(y, dydx, x)
            f(y, x)
            df(dydx, x)
            return y, dydx
        end
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx)
end

"""
    ExplicitComponent(component::AbstractImplicitComponent; kwargs...)

Couple an implicit component with a solver to construct an explicit component.

For computational efficiency when computing derivatives it is recommended that
the version of this function which incorporates an output component is used
rather than this one.

 # Keyword Arguments
 - `solver = Newton()`: Solver, either a function of the form `y = f(component, x, y0)`
        or an object of type [`AbstractSolver`](@ref)
 - `y0 = rand(length(component.y_f))`: Initial guess for outputs
 - `dydx`: Provides size and type of jacobian output, otherwise it will be
        allocated based on the inputs, outputs, and the sparsity structure in `sparsity`
 - `sparsity = DensePattern()`: Defines the sparsity structure of the jacobian if `dydx` is not provided
"""
function ExplicitComponent(component::AbstractImplicitComponent; solver=Newton(),
    y0 = rand(length(component.y_f)), dydx = nothing, sparsity=DensePattern())

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* component.x_f
    x_df = NaN .* component.x_f
    y = NaN .* component.y_f
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y, sparsity) : NaN .* dydx

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
            copyto!(y, solver(cache, x, y0))
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # y0 might not have the right type so we use y instead
            y .= y0
            # call solver and copy results to output vector
            copyto!(y, solver(new_cache, x, y))
        end
        return y
    end

    # function and partials
    fdf = function(y, dydx, x)
        # determine whether to use internal cache
        if eltype(x) <: eltype(xcache)
            # use existing cache if we can store the inputs
            copyto!(xcache, x)
            copyto!(y, solver(cache, x, y0))
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # y0 might not have the right type so we use y instead
            y .= y0
            # call solver and copy results to output vector
            copyto!(y, solver(new_cache, x, y0))
        end
        # get jacobian of the residual function with respect to the inputs
        drdx = residual_input_jacobian!(component, x, y)
        # get jacobian of the residual function with respect to the outputs
        drdy = residual_output_jacobian!(component, x, y)
        # get analytic sensitivities
        copyto!(dydx, -drdy\drdx)
        # return outputs
        return y, dydx
    end

    # partials only
    df = function(dydx, x)
        y, dydx = fdf(y, dydx, x)
        return dydx
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx)
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
 - `u0 = rand(length(component.y_f))`: Initial guess for state variables
 - `dydx`: Provides size and type of jacobian output, otherwise it will be
        allocated based on the inputs, outputs, and the sparsity structure in `sparsity`
 - `sparsity = DensePattern()`: Defines the sparsity structure of the jacobian if `dydx` is not provided
 - `mode = Adjoint()`: Mode in which to compute the derivatives.  Choose between
    [`Direct()`](@ref) and [`Adjoint()`](@ref).
"""
function ExplicitComponent(component::AbstractImplicitComponent, output_component;
    solver=Newton(), u0=rand(length(component.y_f)), dydx = nothing,
    sparsity=DensePattern(), mode = Adjoint())

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* component.x_f
    x_df = NaN .* component.x_f
    y = NaN .* output_component.y_f
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y, sparsity) : NaN .* dydx

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

    # vector for storing the inputs to the output function
    x_output = vcat(component.x_f, outputs(component))

    # view into design variables of implicit system
    xx_output = view(x_output, 1:nx)

    # view into state variables of implicit system
    xu_output = view(x_output, nx+1:nx+nu)

    # output function
    f = function(y, x)
        if eltype(x) <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            u = solver(cache, x, u0)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # u0 might not have the right type so we convert it
            u = solver(new_cache, x, eltype(x).(u0))
        end
        # fill in inputs to output function
        copyto!(xx_output, x)
        copyto!(xu_output, u)
        # get outputs from output function
        outputs!(output_component, x_output)
        # copy outputs to return vector
        copyto!(y, outputs(output_component))
        return y
    end

    # partials
    df = function(dydx, x)
        if eltype(x) <: eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            u = solver(cache, x, u0)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # u0 might not have the right type so we convert it
            u = solver(new_cache, x, eltype(x).(u0))
        end
        # get jacobian of the residual function with respect to the inputs
        drdx = residual_input_jacobian!(component, x, u)
        # get jacobian of the residual function with respect to the state variables
        drdu = residual_output_jacobian!(component, x, u)
        # copy inputs and state variables to output function inputs
        copyto!(xx_output, x)
        copyto!(xu_output, u)
        # get jacobian of the outputs with respect to the inputs and state variables
        df = jacobian!(output_component, x_output)
        # extract output jacobian corresponding to inputs
        dfdx = view(df, :, 1 : nx)
        # extract output jacobian corresponding to state variables
        dfdu = view(df, :, nx + 1 : nx + nu)
        # apply analytic sensitivity equation in direct or adjoint mode
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, dydu, mode))
        return dydx
    end

    fdf = function(y, dydx, x)
        if eltype(x) == eltype(xcache)
            # use existing cache since we can store the inputs
            copyto!(xcache, x)
            u = solver(cache, x, u0)
        else
            # we can't use the existing cache, create a new cache
            new_cache = create_nonmutating_solver_cache(component, x, solver)
            # u0 might not have the right type so we convert it
            u = solver(new_cache, x, eltype(x).(u0))
        end
        # get residuals and jacobians of the residual function
        r, drdx, drdu = residuals_and_jacobian!(component, x, y)
        # copy inputs and outputs to output function inputs
        copyto!(xx_output, x)
        copyto!(xu_output, u)
        # get outputs and jacobian of the outputs with respect to the inputs and state variables
        y[:], df = outputs_and_jacobian!(output_component, x_output)
        # extract output jacobian corresponding to inputs
        dfdx = view(df, :, 1 : nx)
        # extract output jacobian corresponding to state variables
        dfdu = view(df, :, nx + 1 : nx + nu)
        # apply analytic sensitivity equation in direct or adjoint mode
        copyto!(dydx, analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, dydu, mode))
        return y, dydx
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx)
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
    ExplicitSystem(x0, y0, components, component_mapping, output_mapping)

Construct an explicit system component from a collection of explicit system
components.

# Arguments
 - `x0`: Initial values for system inputs (used for size and type information)
 - `y0`: Initial values for system outputs (used for size and type information)
 - `components::TC`: Collection of explicit components (see [`ExplicitComponents`](@ref))
    in calling order
 - `component_mapping::Vector{Vector{NTuple{2, Int}}`: Output to input mapping
    for each component.  The first index corresponds to the component from which
    a component input is taken, with index 0 corresponding to a value from the
    system inputs. The second index is an index into the specified array.
 - `output_mapping::Vector{NTuple{2, Int}}`: Output to input mapping
    for the combined system outputs. The first index corresponds to the component
    from which an input is taken, with index 0 corresponding to a value from the inputs.
    The second index is an index into the specified array.

# Keyword Arguments
 - `dydx`: Matrix used to define size and type of the jacobian matrix. If omitted,
    the jacobian matrix size and type will be infered from the inputs, outputs, and
    the sparsity structure.
 - `sparsity`: Sparsity structure of the system jacobian matrix
 - `mode`: Mode used to calculate system derivatives using the chain rule.  May
    be either [`Forward()`](@ref) or [`Reverse()`](@ref), defaults to [`Reverse()`](@ref)
"""
function ExplicitSystem(x0, y0, components, component_input_mapping, output_mapping;
    dydx = nothing, mode=Reverse(), sparsity=DensePattern())

    # number of components
    nc = length(components)

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
                    "corresponds to non-existant output $jx of component $jc"
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

    return ExplicitSystem(Tuple(components), collect(input_mapping),
        collect(component_output_mapping), collect(component_input_mapping),
        collect(output_mapping), x_f, x_df, y, dydx, mode)
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
    ImplicitComponent(x0, y0, r0; kwargs...)

Construct a system component defined by the implicit vector-valued residual
function: `0 = f(x, y)`.

# Arguments
 - `x0`: Initial values for inputs (used for size and type information)
 - `y0`: Initial values for outputs (used for size and type information)

# Keyword Arguments
 - `f`: In-place residual function `f(r, x, y)`.
 - `dfdx`: In-place residual jacobian function with respect to the inputs `dfdx(drdx, x, y)`
 - `dfdy`: In-place residual jacobian function with respect to the outputs `dfdy(drdy, x, y)`
 - `fdfdx`: In-place combined residual and jacobian with respect to the inputs function `fdfdx(r, drdx, x, y)`.
 - `fdfdy`: In-place combined residual and jacobian with respect to the outputs function `fdfdy(r, drdy, x, y)`.
 - `fdf`: In-place combined residual and jacobians function `fdf(r, drdx, drdy, x, y)`.
 - `drdx`: Matrix used to define size and type of the jacobian of the residual
    with respect to the inputs. If omitted, its size and type will be infered
    from the inputs, residuals, and sparsity.
 - `drdy`: Matrix used to define size and type of the jacobian of the residual
     with respect to the outputs. If omitted, its size and type will be infered
    from the outputs, residuals, and sparsity.
 - `xderiv`: Method used to calculate the residual jacobian with respect to the
    inputs if it is not provided. Defaults to [`ForwardFD`](@ref).
 - `yderiv`: Method used to calculate the residual jacobian with respect to the
    outputs if it is not provided. Defaults to [`ForwardFD`](@ref).
 - `xsparsity`: Sparsity structure of the residual jacobian with respect to the
    inputs. Defaults to [`DensePattern()`](@ref)
 - `ysparsity`: Sparsity structure of the residual jacobian with respect to the
    outputs. Defaults to [`DensePattern()`](@ref)

 **Either the `f`, `fdfdx`, `fdfdy`, or `fdf` keyword arguments must be provided**
"""
function ImplicitComponent(x0, y0, r0; f=nothing, dfdx=nothing, dfdy=nothing,
    fdfdx=nothing, fdfdy=nothing, fdf=nothing, drdx=nothing, drdy=nothing,
    xderiv=ForwardFD(), yderiv=ForwardFD(), xsparsity=DensePattern(),
    ysparsity=DensePattern())

    # ensure `f` is defined
    @assert any((!isnothing(f), !isnothing(fdfdx), !isnothing(fdfdy), !isnothing(fdf))) "Residual function not defined"

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
    drdx = isnothing(drdx) ? allocate_jacobian(x0, r0, xsparsity) : drdx .* NaN
    drdy = isnothing(drdy) ? allocate_jacobian(y0, r0, ysparsity) : drdy .* NaN

    # construct residual function if necessary
    if isnothing(f)
        if !isnothing(fdfdx)
            # construct residual function from fdfdx
            # NOTE: This updates drdx as well
            f = function(r, x, y)
                copyto!(x_dfdx, x)
                copyto!(y_dfdx, y)
                fdfdx(r, drdx, x, y)
                return y
            end
        elseif !isnothing(fdfdy)
            # construct residual function from fdfdy
            # NOTE: This updates drdy as well
            f = function(r, x, y)
                copyto!(x_dfdy, x)
                copyto!(y_dfdy, y)
                fdf(r, drdy, x, y)
                return y
            end
        else
            # construct residual function from fdf
            # NOTE: This updates drdx and drdy as well
            f = function(r, x, y)
                copyto!(x_dfdx, x)
                copyto!(y_dfdx, y)
                copyto!(x_dfdy, x)
                copyto!(y_dfdy, y)
                fdf(r, drdx, drdy, x, y)
                return y
            end
        end
    end

    # construct jacobian functions with respect to the inputs
    if isnothing(dfdx) && isnothing(fdfdx) && isnothing(fdf)
        # construct dfdx and fdfdx from f
        dfdx, fdfdx = create_residual_input_jacobian_functions(f, x_f, y_f, r, xderiv, xsparsity)
    elseif isnothing(dfdx) && isnothing(fdfdx)
        # construct dfdx from fdf
        # NOTE: This updates r and drdy as well
        dfdx = function(drdx, x, y)
            copyto!(x_f, x)
            copyto!(y_f, y)
            copyto!(x_dfdy, x)
            copyto!(y_dfdy, y)
            fdf(r, drdx, drdy, x, y)
            return drdx
        end

        # construct fdfdx from fdf
        # NOTE: This updates drdy as well
        fdfdx = function(r, drdx, x, y)
            copyto!(x_dfdy, x)
            copyto!(y_dfdy, y)
            fdf(r, drdx, drdy, x, y)
            return r, drdx
        end
    elseif isnothing(dfdx)
        # construct dfdx from fdfdx
        # NOTE: This updates r as well
        dfdx = function(drdx, x, y)
            copyto!(x_f, x)
            copyto!(y_f, y)
            fdfdx(r, drdx, x, y)
            return drdx
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
        # NOTE: This updates r and drdx as well
        dfdy = function(drdy, x, y)
            copyto!(x_f, x)
            copyto!(y_f, y)
            copyto!(x_dfdx, x)
            copyto!(y_dfdx, y)
            fdf(r, drdx, drdy, x, y)
            return drdy
        end

        # construct fdfdy from fdf
        # NOTE: This updates drdx as well
        fdfdy = function(r, drdy, x, y)
            copyto!(x_dfdx, x)
            copyto!(y_dfdx, y)
            fdf(r, drdx, drdy, x, y)
            return r, drdy
        end
    elseif isnothing(dfdy)
        # construct dfdy from fdfdy
        # NOTE: This updates r as well
        dfdy = function(drdy, x, y)
            copyto!(x_f, x)
            copyto!(y_f, y)
            fdfdy(r, drdy, x, y)
            return drdy
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
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy)
end

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

    return ImplicitComponent(f, dfdx, dfdy, fdfdx, fdfdy, fdf, x_f, y_f, x_dfdx,
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy)
end

"""
    ImplicitSystem(x0, components, component_input_mapping;
        drdx=nothing, drdy=nothing, xsparsity=DensePattern(), ysparsity=DensePattern())

Constructs an implicit system component from an explicit system component.
"""
function ImplicitSystem(x0, components, component_input_mapping;
    drdx=nothing, drdy=nothing, xsparsity=DensePattern(), ysparsity=DensePattern())

    # number of components
    nc = length(components)

    # ensure all components are implicit
    @assert all(isa.(components, AbstractImplicitComponent)) "All components of an implicit system must be implicit"

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

    # construct index for accessing outputs/residuals for each component
    idx = vcat(0, cumsum(length.(residuals.(components)))...)

    return ImplicitSystem(Tuple(components), collect(component_input_mapping),
        x_f, y_f, x_dfdx, y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, idx)
end
