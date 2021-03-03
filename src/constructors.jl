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
        end
    end

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx)
end

"""
    ExplicitComponent(component::ImplicitComponent; solver, y0, dydx, sparsity)

Constructs an explicit system component from an implicit system component
coupled with a solver and an initial guess.
"""
function ExplicitComponent(component::ImplicitComponent; solver=nothing,
    y0=rand(length(component.y_f)), dydx = nothing, sparsity=DensePattern())

    # allocate storage and initialize with NaNs (since values are as of yet undefined)
    x_f = NaN .* component.x_f
    x_df = x_f # alias with x_f since both are computed simulataneously
    y = NaN .* component.y_f
    dydx = isnothing(dydx) ? allocate_jacobian(x_f, y, sparsity) : NaN .* dydx

    f = function(y, x)
        # assemble inputs to nonlinear solver
        f! = (r, y) -> residuals!(component, r, x, y)
        j! = (drdy, y) -> residual_output_jacobian!(component, drdy, x, y)
        fj! = function(r, drdy, y)
            residuals!(component, r, x, y)
            residual_output_jacobian!(component, drdy, x, y)
            return r, drdy
        end
        r = residuals(component)
        drdy = residual_output_jacobian(component)
        y_f = component.y_f
        y_df = component.y_dfdy
        f_calls = [0,]
        df_calls = [0,]

        df = OnceDifferentiable(f!, j!, fj!, r, drdy, y_f, y_df, f_calls, df_calls)

        # solve nonlinear system of equations for outputs
        results = nlsolve(df, y0, method=:newton, linesearch=BackTracking())

        if results.f_converged
            copyto!(y0, results.zero)
            copyto!(y, results.zero)
        else
            copyto!(y, NaN)
        end

        # return outputs
        return y
    end

    fdf = function(y, dydx, x)

        # converge residual function
        f(y, x)

        # get jacobian of the residual function with respect to the inputs
        drdx = residual_input_jacobian(component, x, y)

        # get jacobian of the residual function with respect to the outputs
        drdy = residual_output_jacobian(component, x, y)

        # get analytic sensitivities
        dydx .= -drdy\drdx

        return y, dydx
    end

    df = (dydx, x) -> fdf(y, dydx, x)[2]

    return ExplicitComponent(f, df, fdf, x_f, x_df, y, dydx)
end

"""
    ExplicitSystem(x0, y0, components, component_input_mapping, output_mapping)

Construct an explicit system component from a collection of explicit system
components.

# Arguments
 - `x0`: Initial values for inputs (used for size and type information)
 - `y0`: Initial values for outputs (used for size and type information)
 - `components::TC`: Collection of explicit components (see [`ExplicitComponents`](@ref))
    in calling order
 - `component_input_mapping::Vector{Vector{NTuple{2, Int}}`: Output to input mapping
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
 - `mode`: Mode used to calculate system derivatives.  May be either `Forward()` or
    `Reverse()`, defaults to `Reverse()`
"""
function ExplicitSystem(x0, y0, components, component_input_mapping, output_mapping;
    dydx = nothing, mode=Reverse(), sparsity=DensePattern())

    # number of components
    nc = length(components)

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
    input_mapping, component_output_mapping = output_to_input_mapping(x0,
        components, component_input_mapping, output_mapping)

    return ExplicitSystem(Tuple(components), collect(input_mapping),
        collect(component_output_mapping), collect(component_input_mapping),
        collect(output_mapping), x_f, x_df, y, dydx, mode)
end

"""
    output_to_input_mapping(inputs, components, component_input_mapping, output_mapping)

Returns (`input_mapping, component_output_mapping`) where `input_mapping` contains
a mapping from each system input to component inputs and/or system outputs and
`reversed_component_mapping` contains a mapping from each component output to
corresponding component inputs and/or system outputs.
"""
function output_to_input_mapping(inputs, components, component_input_mapping, output_mapping)

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

Construct a system component defined by the explicit vector-valued output
function: `y = f(x)`.

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
    inputs if it is not provided. Defaults to forward finite differencing.
 - `yderiv`: Method used to calculate the residual jacobian with respect to the
    outputs if it is not provided. Defaults to forward finite differencing.
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
    drdy = sparse(1:length(r), 1:length(y_f), ones(length(r)))

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

function ImplicitSystem(x0, y0, components, component_input_mapping, output_mapping;
    xsys = nothing, ysys = nothing, rsys = nothing, drdx = nothing,
    drdy = nothing, mode = Reverse())

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

    # construct forward mode mapping
    input_mapping, component_output_mapping = output_to_input_mapping(x0,
        components, component_input_mapping, output_mapping

    # allocate storage and initialize with NaNs (since values are not yet defined)
    x_f = vcat(eltype(x0)[], inputs.(components)...) .* NaN
    y_f = vcat(eltype(y0)[], outputs.(components)...) .* NaN
    x_dfdx = copy(x_f)
    y_dfdx = copy(y_f)
    x_dfdy = copy(x_f)
    y_dfdy = copy(y_f)
    r = copy(y_f)
    drdx = spzeros(promote_type(eltype(rsys), eltype(xsys)), length(rsys), length(xsys))
    drdy = spzeros(promote_type(eltype(rsys), eltype(ysys)), length(rsys), length(ysys))

    # construct index for accessing outputs/residuals for each component
    idx = vcat(0, cumsum(length.(residuals.(components))))

    return ImplicitSystem(Tuple(components), collect(input_mapping),
        collect(component_output_mapping), collect(component_input_mapping),
        collect(output_mapping), x_f, y_f, x_dfdx, y_dfdx, x_dfdy, y_dfdy, r,
        drdx, drdy, idx, mode)
end
