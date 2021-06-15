# Functions for AbstractComponent

inputs(component::ExplicitComponent) = component.x_f
inputs(component::ExplicitSystem) = component.x_f
outputs(component::ExplicitComponent) = component.y
outputs(component::ExplicitSystem) = component.y

# Functions for AbstractExplicitComponent

"""
    outputs(component::AbstractExplicitComponent, x)

Evaluate the outputs of an explicit system component.

This does *not* update any of the values stored in `component`
"""
outputs(component::AbstractExplicitComponent, x)

function outputs(component::ExplicitComponent, x)
    y = similar(component.y, promote_type(eltype(component.y), eltype(x)))
    return outputs!(component, y, x)
end

function outputs(component::ExplicitSystem, x)
    y = similar(component.y, promote_type(eltype(component.y), eltype(x)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    return outputs!(component, y, xsub, ysub, x)
end

"""
    outputs!(component::AbstractExplicitComponent, y, x)

Evaluate the outputs of an explicit system component and store the result in y.

This does *not* update any of the values stored in `component`
"""
outputs!(component::AbstractExplicitComponent, y, x)

function outputs!(component::ExplicitComponent, y, x)
    component.f(y, x)
    return y
end

function outputs!(component::ExplicitSystem, y, x)
    subcomponents = component.components
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in subcomponents]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in subcomponents]
    return outputs!(component, y, xsub, ysub, x)
end

"""
    outputs!(component::ExplicitSystem, y, xsub, ysub, x)

Evaluate the outputs of an explicit system and store the result in y.
Store the intermediate results in `xsub` and `ysub`.

This does *not* update any of the values stored in `component`
"""
function outputs!(component::ExplicitSystem, y, xsub, ysub, x)
    subcomponent_outputs!(component, xsub, ysub, x)
    update_system_outputs!(component, y, x, ysub)
    return y
end

"""
    outputs!(component::AbstractExplicitComponent, x)

Evaluate the outputs of an explicit system component.

Return the result and store in `component.y`.
"""
function outputs!(component::AbstractExplicitComponent, x)
    if x != component.x_f
        outputs!!(component, x)
    end
    return outputs(component)
end

"""
    outputs!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the outputs of an explicit system component.

Return the result and store in `component.y`.
"""
outputs!!(component::AbstractExplicitComponent, x)

function outputs!!(component::ExplicitComponent, x)
    copyto!(component.x_f, x)
    component.f(component.y, x)
    return outputs(component)
end

function outputs!!(component::ExplicitSystem, x)
    copyto!(component.x_f, x)
    subcomponent_outputs!(component, x)
    update_system_outputs!(component, x)
    return outputs(component)
end

"""
    outputs!!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the outputs of an explicit system component. Also force
(re-)evaluation of the outputs of the system's subcomponents.

Return the result and store in `component.y`.
"""
outputs!!!(component::AbstractExplicitComponent, x)

outputs!!!(component::ExplicitComponent, x) = outputs!!(component, x)

function outputs!!!(component::ExplicitSystem, x)
    copyto!(component.x_f, x)
    subcomponent_outputs!!(component, x)
    update_system_outputs!(component, x)
    return outputs(component)
end

"""
    jacobian(component::AbstractExplicitComponent)

Return the jacobian of the outputs with respect to the inputs stored in `component`
"""
jacobian(component::AbstractExplicitComponent)
jacobian(component::ExplicitComponent) = component.dydx
jacobian(component::ExplicitSystem) = component.dydx

"""
    jacobian(component::AbstractExplicitComponent, x)

Evaluate the jacobian of the outputs with respect to the inputs.

This does *not* update any of the values stored in `component`
"""
jacobian(component::AbstractExplicitComponent, x)

function jacobian(component::ExplicitComponent, x)
    dydx = similar(component.dydx, promote_type(eltype(component.dydx), eltype(x)))
    return jacobian!(component, dydx, x)
end

function jacobian(component::ExplicitSystem, x)
    dydx = similar(component.dydx, promote_type(eltype(component.dydx), eltype(x)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    return jacobian!(component, dydx, xsub, ysub, dsub, x)
end

"""
    jacobian!(component::AbstractExplicitComponent, dydx, x)

Evaluate the jacobian of the outputs with respect to the inputs and store the
result in `dydx`

This does *not* update any of the values stored in `component`
"""
jacobian!(component::AbstractExplicitComponent, dydx, x)

function jacobian!(component::ExplicitComponent, dydx, x)
    component.df(dydx, x)
    return dydx
end

jacobian!(component::ExplicitSystem, dydx, x) =
    jacobian!(component::ExplicitSystem, dydx, x, component.mode)

function jacobian!(component::ExplicitSystem, dydx, x, mode)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    jacobian!(component, dydx, xsub, ysub, dsub, x, mode)
    return dydx
end

"""
    jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, mode=component.mode)

Evaluate the jacobian of the outputs with respect to the inputs and store the
result in `dydx`.  Store the intermediate subcomponent `x`, `y` and `dydx` values
in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x) =
    jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, component.mode)

function jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Forward)
    subcomponent_outputs_and_jacobians!(component, xsub, ysub, dsub, x)
    forward_mode_jacobian!(component, dydx, dsub)
    return dydx
end

function jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Reverse)
    subcomponent_outputs_and_jacobians!(component, xsub, ysub, dsub, x)
    reverse_mode_jacobian!(component, dydx, dsub)
    return dydx
end

"""
    jacobian!(component::AbstractExplicitComponent, x)

Evaluate the jacobian of the outputs with respect to the inputs.

Return the result and store in `component.dydx`.
"""
function jacobian!(component::AbstractExplicitComponent, x)
    if x != component.x_df
        jacobian!!(component, x)
    end
    return jacobian(component)
end

jacobian!(component::ExplicitSystem, x) = jacobian!(component, x, component.mode)

function jacobian!(component::ExplicitSystem, x, mode::Union{Forward, Reverse})
    if x != component.x_df
        jacobian!!(component, x, mode)
    end
    return jacobian(component)
end

"""
    jacobian!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the jacobian of the outputs with respect to the inputs.

Return the result and store in `component.dydx`.
"""
jacobian!!(component::AbstractExplicitComponent, x)

function jacobian!!(component::ExplicitComponent, x)
    copyto!(component.x_df, x)
    component.df(component.dydx, x)
    return jacobian(component)
end

jacobian!!(component::ExplicitSystem, x) = jacobian!!(component::ExplicitSystem, x, component.mode)

function jacobian!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!(component, x)
    forward_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

function jacobian!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!(component, x)
    reverse_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

"""
    jacobian!!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the jacobian of the outputs with respect to the inputs.
Also force (re-)evaluation of the jacobians of the system's subcomponents.

Return the result and store in `component.dydx`.
"""
jacobian!!!(component::AbstractExplicitComponent, x) = jacobian!!(component, x)

jacobian!!!(component::ExplicitSystem, x) = jacobian!!!(component, x, component.mode)

function jacobian!!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!!(component, x)
    forward_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

function jacobian!!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!!(component, x)
    reverse_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

"""
    outputs_and_jacobian(component::AbstractExplicitComponent)

Return the outputs and their derivatives with respect to the inputs stored in `component`.
"""
outputs_and_jacobian(component) = outputs(component), jacobian(component)

"""
    outputs_and_jacobian(component::AbstractExplicitComponent, x)

Evaluate the outputs and their derivatives with respect to the inputs.

This does *not* update any of the values stored in `component`
"""
outputs_and_jacobian(component::AbstractExplicitComponent, x)

function outputs_and_jacobian(component::ExplicitComponent, x)
    y = similar(component.y, promote_type(eltype(component.y), eltype(x)))
    dydx = similar(component.dydx, promote_type(eltype(component.dydx), eltype(x)))
    return outputs_and_jacobian!(component, y, dydx, x)
end

function outputs_and_jacobian(component::ExplicitSystem, x)
    y = similar(component.y, promote_type(eltype(component.y), eltype(x)))
    dydx = similar(component.dydx, promote_type(eltype(component.dydx), eltype(x)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    return outputs_and_jacobian!(component, y, dydx, xsub, ysub, dsub, x)
end

"""
    outputs_and_jacobian!(component::AbstractExplicitComponent, y, dydx, x)

Evaluate the outputs and their derivatives with respect to the inputs and store
the results in `y` and `dydx`

This does *not* update any of the values stored in `component`
"""
outputs_and_jacobian!(component::AbstractExplicitComponent, y, dydx, x)

function outputs_and_jacobian!(component::ExplicitComponent, y, dydx, x)
    component.fdf(y, dydx, x)
    return y, dydx
end

function outputs_and_jacobian!(component::ExplicitSystem, y, dydx, x)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    return outputs_and_jacobian!(component, y, dydx, xsub, ysub, dsub, x)
end

"""
    outputs_and_jacobian!(component::ExplicitSystem, y, dydx, xsub, ysub, dsub,
        x, mode=component.mode)

Evaluate the outputs and their derivatives with respect to the inputs and store
the results in `y` and `dydx`.  Store the intermediate subcomponent `x`, `y` and
`dydx` values in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
outputs_and_jacobian!(component::ExplicitSystem, y, dydx, xsub, ysub,
    dsub, x) = outputs_and_jacobian!(component, y, dydx, xsub, ysub, dsub, x,
    component.mode)

function outputs_and_jacobian!(component::ExplicitSystem, y, dydx, xsub, ysub, dsub, x, ::Forward)
    subcomponent_outputs_and_jacobians!(component, xsub, ysub, dsub, x)
    forward_mode_jacobian!(component, dydx, dsub)
    update_system_outputs!(component, y, x, ysub)
    return y, dydx
end

function outputs_and_jacobian!(component::ExplicitSystem, y, dydx, xsub, ysub, dsub, x, ::Reverse)
    subcomponent_outputs_and_jacobians!(component, xsub, ysub, dsub, x)
    reverse_mode_jacobian!(component, dydx, dsub)
    update_system_outputs!(component, y, x, ysub)
    return y, dydx
end

"""
    outputs_and_jacobian!(component::AbstractExplicitComponent, x)

Evaluate the outputs and their derivatives with respect to the inputs.

Return the result and store in `component.y` and `component.dydx`
"""
function outputs_and_jacobian!(component::AbstractExplicitComponent, x)
    if (x != component.x_f) && (x != component.x_df)
        outputs_and_jacobian!!(component, x)
    elseif x != component.x_f
        outputs!!(component, x)
    elseif x != component.x_df
        jacobian!!(component, x)
    end
    return outputs_and_jacobian(component)
end

"""
    outputs_and_jacobian!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the outputs and their derivatives with respect to the
inputs.

Return the result and store in `component.y` and `component.dydx`
"""
outputs_and_jacobian!!(component::AbstractExplicitComponent, x)

function outputs_and_jacobian!!(component::ExplicitComponent, x)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    component.fdf(component.y, component.dydx, x)
    return outputs_and_jacobian(component)
end

outputs_and_jacobian!!(component::ExplicitSystem, x) =
    outputs_and_jacobian!!(component, x, component.mode)

function outputs_and_jacobian!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!(component, x)
    forward_mode_jacobian!(component, component.dydx)
    update_system_outputs!(component, x)
    return outputs_and_jacobian(component)
end

function outputs_and_jacobian!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!(component, x)
    reverse_mode_jacobian!(component, component.dydx)
    update_system_outputs!(component, x)
    return outputs_and_jacobian(component)
end

"""
    outputs_and_jacobian!!!(component::AbstractExplicitComponent, x)

Force (re-)evaluation of the outputs and their derivatives with respect to the
inputs for the system.  Also force (re-)evaluation of the outputs and jacobians
of the system's subcomponents.

Return the result and store in `component.y` and `component.dydx`
"""
outputs_and_jacobian!!!(component::AbstractExplicitComponent, x) =
    outputs_and_jacobian!!(component, x)

outputs_and_jacobian!!!(component::ExplicitSystem, x) =
    outputs_and_jacobian!!!(component, x, component.mode)

function outputs_and_jacobian!!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!!(component, x)
    forward_mode_jacobian!(component, component.dydx)
    update_system_outputs!(component, x)
    return outputs_and_jacobian(component)
end

function outputs_and_jacobian!!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_outputs_and_jacobians!!(component, x)
    reverse_mode_jacobian!(component, component.dydx)
    update_system_outputs!(component, x)
    return outputs_and_jacobian(component)
end
