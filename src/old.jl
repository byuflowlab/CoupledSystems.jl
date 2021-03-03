





"""
    ExplicitComponentGroup{TX, TY, TJ, TC} <: AbstractComponent

Group of explicit components.

# Fields:
 - `x_f::TX`: Inputs used to find outputs
 - `x_df::TX`: Inputs used to find jacobian
 - `y::TY`: Cache for outputs
 - `dydx::TD`: Cache for jacobian
 - `components::TC`: Collection of explicit components (see [`ExplicitComponents`](@ref))
    in calling order
 - `component_mapping::Vector{Vector{NTuple{2, Int}}`: Output to input mapping
    for each component.
 - `output_mapping::Vector{NTuple{2, Int}}`: Output to input mapping
    for the combined system outputs
"""
struct ExplicitComponentGroup{TX, TY, TJ, TC} <: AbstractComponent
    x_f::TX
    x_df::TX
    y::TY
    dydx::TJ
    components::TC
    component_mapping::Vector{Vector{NTuple{2,Int}}}
    output_mapping::Vector{NTuple{2,Int}}
end



"""
    ImplicitComponentGroup{TX, TY, TR, TJX, TJY, TC} <: AbstractComponent

Group of explicit and/or implicit components to be solved implicitly.

# Fields:
 - `x::TX`: Inputs (scalar or vector)
 - `y::TY`: Outputs (scalar or vector)
 - `dfdx::TDX`: Derivative of `f` with respect to `x`
 - `dfdy::TDY`: Derivative of `f` with respect to `y`.
 - `components::TC`: Collection of components (see [`Components`](@ref)) in
    calling order
 - `component_mapping::Vector{Vector{NTuple{2, Int}}`: Output to input mapping
    for each component.
 - `output_mapping::Vector{NTuple{2, Int}}`: Output to input mapping
    for the combined system outputs
"""
struct ImplicitComponentGroup{TX, TY, TR, TJX, TJY, TC} <: AbstractComponent
    x_f::TX
    y_f::TY
    x_dfdx::TX
    y_dfdx::TY
    x_dfdy::TX
    y_dfdy::TY
    r::TR
    drdx::TJX
    drdy::TJY
    components::TC
    component_mapping::Vector{Vector{NTuple{2,Int}}}
    output_mapping::Vector{NTuple{2,Int}}
end

ExplicitComponent(component::ExplicitComponent) = component

function ExplicitComponent(component::ImplicitComponent, y0 = rand(length(component.y_f)))

    x = component.x_f
    y = component.y_f
    dydx = component.dydx

    f = function(y, x)

        # assemble inputs to nonlinear solver (without creating copies)
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

        copyto!(y0, results.zero)
        copyto!(y, results.zero)

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

    return ExplicitComponent(f, df, fdf, x, y, dydx)
end


function ExplicitComponentGroup(x, y, dydx, components, component_mapping, output_mapping)

    # number of components
    nc = length(components)

    # create storage
    x_f = copy(x)
    x_df = copy(x)
    y = copy(y)
    dydx = copy(dydx)

    # check that there is an input mapping for each component
    @assert length(components) == length(component_mapping) "There must be an input mapping for each component"
    for ic = 1:nc
        # check that there is an input mapping for each input to the component
        @assert length(components[ic].x) == length(component_mapping[i]) "Input mapping for component $ic must have the same length as the number of inputs to component $ic"
        for (jc, jx) in component_mapping[ic]
            # check that each input only uses available outputs
            @assert jc < nc "Input corresponds to non-existant component"
            @assert jc < ic "Implicit dependency in explicit system"
            @assert jx < length(components[ic].y) "Input corresponds to non-existant output"
        end
    end

    ExplicitComponentGroup(x_f, x_df, y, dydx, components, component_mapping, output_mapping)
end


function ImplicitComponent(f, dfdx, dfdy, fdf, x, y, r, drdx, drdy, dydx)

    # create storage
    x_f = copy(x)
    x_dfdx = copy(x)
    x_dfdy = copy(x)
    y_f = copy(y)
    y_dfdx = copy(y)
    y_dfdy = copy(y)
    r = copy(r)
    drdx = copy(drdx)
    drdy = copy(drdy)
    dydx = copy(dydx)

    return ImplicitComponent(f, dfdx, dfdy, fdf, x_f, y_f, x_dfdx, y_dfdx,
        x_dfdy, y_dfdy, r, drdx, drdy, dydx)
end


function ImplicitComponentGroup(x, y, r, drdx, drdy, components, component_mapping, output_mapping)

    # number of components
    nc = length(components)

    # create storage
    x_f = copy(x)
    y_f = copy(y)
    x_dfdx = copy(x)
    y_dfdx = copy(y)
    x_dfdy = copy(x)
    y_dfdy = copy(y)
    r = copy(r)
    drdx = copy(drdx)
    drdy = copy(drdy)

    # check that there is an input mapping for each component
    @assert length(components) == length(component_mapping) "There must be an input mapping for each component"
    for ic = 1:nc
        # check that there is an input mapping for each input to the component
        @assert length(components[ic].x) == length(component_mapping[i]) "Input mapping for component $ic must have the same length as the number of inputs to component $ic"
        for (jc, jx) in component_mapping[ic]
            # check that each input only uses available outputs
            @assert jc < nc "Input corresponds to non-existant component"
            @assert jx < length(components[ic].y) "Input corresponds to non-existant output"
        end
    end
    ImplicitComponentGroup(x_f, y_f, x_dfdx, y_dfdx, x_dfdy, y_dfdy, r, drdx,
        drdy, components, component_mapping, output_mapping)
end
