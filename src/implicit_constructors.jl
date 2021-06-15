
"""
    ImplicitComponent([func,] fin, fout; kwargs...)

Construct a system component defined by the in-place residual function
`func` with inputs corresponding to `fin` and state variables/outputs
corresponding to `fout`.

# Arguments
 - `func`: Template residual function, of the form `r = func(r, fin..., fout...)`
 - `fin`: Tuple of named or unnamed variables (see [`NamedVar`](@ref)) corresponding to inputs to `func`
 - `fout`: Tuple of named or unnamed variables (see [`NamedVar`](@ref)) corresponding to state variables/outputs from `func`

# Keyword Arguments
 - `f`: In-place residual function `f(r, x, y)`. `x` is an input vector containing
    values corresponding to all of the named variables in `fin`. `y` is an output
    vector containing values corresponding to all of the named variables in `fout`
 - `dfdx`: In-place residual jacobian function with respect to the inputs `dfdx(drdx, x, y)`
 - `dfdy`: In-place residual jacobian function with respect to the outputs `dfdy(drdy, x, y)`
 - `fdfdx`: In-place combined residual and jacobian with respect to the inputs function `fdfdx(r, drdx, x, y)`.
 - `fdfdy`: In-place combined residual and jacobian with respect to the outputs function `fdfdy(r, drdy, x, y)`.
 - `fdf`: In-place combined residual and jacobians function `fdf(r, drdx, drdy, x, y)`.
 - `r`: Vector which defines the size and type of the component's residuals.
    Defaults to the size and type of the output vector. It's length must
    correspond to the total number of component outputs/states.
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
"""
function ImplicitComponent(func, fin, fout;
    f=nothing, dfdx=nothing, dfdy=nothing, fdfdx=nothing, fdfdy=nothing, fdf=nothing,
    r=nothing, drdx=nothing, drdy=nothing, xderiv=ForwardFD(), yderiv=ForwardFD(),
    xsparsity = DensePattern(), ysparsity = DensePattern())

    # ensure the residual function is defined
    @assert any((!isnothing(func), !isnothing(f), !isnothing(fdfdx), !isnothing(fdfdy), !isnothing(fdf))) "Residual function not defined"

    # template function input/output arguments
    tin = fin
    tout = fout

    # template function default input/output values
    tin_val = get_values(tin)
    tout_val = get_values(tout)

    # component input/output argument indices (as a tuple)
    cin_idx = (findall(in(get_names(tin)), get_name.(tin))...,)
    cout_idx = (findall(in(get_names(tout)), get_name.(tout))...,)

    # component input/output arguments
    cin = getindices(fin, cin_idx)
    cout = getindices(fout, cout_idx)

    # component default input/output values
    cin_val = get_values(cin)
    cout_val = get_values(cout)

    # default component input/output vectors
    x0 = combine(cin_val)
    y0 = combine(cout_val)
    r0 = isnothing(r) ? similar(y0, promote_type(eltype(x0), eltype(y0))) : r

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

    # construct residual function (if necessary)
    if isnothing(f)
        if !isnothing(func)
            # construct residual function from `func`
            f = function(r, x, y)
                # expand component inputs and outputs to provided sizes/types
                fin_val = separate!(cin_val, x)
                fout_val = separate!(cout_val, y)
                # insert component inputs/outputs into template inputs/outputs
                fin_val = setindices(tin_val, fin_val, cin_idx)
                fout_val = setindices(tout_val, fout_val, cout_idx)
                # call function to update residual values
                r = func(r, fin_val..., fout_val...)
                # return result
                return r
            end
        elseif !isnothing(fdfdx)
            # construct residual function from fdfdx
            f = let drdx = copy(drdx)
                function(r, x, y)
                    fdfdx(r, drdx, x, y)
                    return y
                end
            end
        elseif !isnothing(fdfdy)
            # construct residual function from fdfdy
            f = let drdy = copy(drdy)
                function(r, x, y)
                    fdf(r, drdy, x, y)
                    return y
                end
            end
        else
            # construct residual function from fdf
            f = let drdx = copy(drdx), drdy = copy(drdy)
                function(r, x, y)
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
        dfdx = let r = copy(r), drdy = copy(drdy)
            function(drdx, x, y)
                fdf(r, drdx, drdy, x, y)
                return drdx
            end
        end

        # construct fdfdx from fdf
        fdfdx = let drdy = copy(drdy)
            function(r, drdx, x, y)
                fdf(r, drdx, drdy, x, y)
                return r, drdx
            end
        end
    elseif isnothing(dfdx)
        # construct dfdx from fdfdx
        dfdx = let r = copy(r)
            function(drdx, x, y)
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
        dfdy = let r = copy(r), drdx = copy(drdx)
            function(drdy, x, y)
                fdf(r, drdx, drdy, x, y)
                return drdy
            end
        end

        # construct fdfdy from fdf
        fdfdy = let drdx = copy(drdx)
            function(r, drdy, x, y)
                fdf(r, drdx, drdy, x, y)
                return r, drdy
            end
        end
    elseif isnothing(dfdy)
        # construct dfdy from fdfdy
        dfdy = let r = copy(r)
            function(drdy, x, y)
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
        y_dfdx, x_dfdy, y_dfdy, r, drdx, drdy, cin, cout)
end

ImplicitComponent(fin, fout; kwargs...) = ImplicitComponent(nothing, fin, fout; kwargs...)

"""
    make_implicit(component::ExplicitComponent)

Construct an implicit system component from an explicit system component.
"""
function make_implicit(component::ExplicitComponent)

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
        return drdy
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

make_implicit(component::AbstractImplicitComponent) = component

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
    component_input_mapping = system_component_mapping(argin, components)

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
    analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, mode)

Compute the sensitivities of an implicit system using the analytic sensitivity
equation in `Direct()` or `Adjoint()` mode.
"""
analytic_sensitivity_equation
analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, ::Direct) = dfdx - dfdu*(drdu\drdx)
analytic_sensitivity_equation(dfdx, dfdu, drdx, drdu, ::Adjoint) = dfdx - transpose(transpose(drdu)\transpose(dfdu))*drdx
