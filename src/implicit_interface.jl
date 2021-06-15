# Functions for AbstractComponent

inputs(component::ImplicitComponent) = component.x_f
inputs(component::ImplicitSystem) = component.x_f
outputs(component::ImplicitComponent) = component.y_f
outputs(component::ImplicitSystem) = component.y_f

# Functions for AbstractImplicitComponent

"""
    residuals(component::AbstractImplicitComponent)

Return the residuals stored in `component`
"""
residuals(component::AbstractImplicitComponent)
residuals(component::ImplicitComponent) = component.r
residuals(component::ImplicitSystem) = component.r

"""
    residuals(component::AbstractImplicitComponent, x, y)

Evaluate the residuals of an implicit system component.

This does *not* update any of the values stored in `component`
"""
function residuals(component::AbstractImplicitComponent, x, y)
    TR = promote_type(eltype(component.r), eltype(x), eltype(y))
    r = similar(component.r, TR)
    return residuals!(component, r, x, y)
end

"""
    residuals!(component::AbstractImplicitComponent, r, x, y)

Evaluate the residuals of an implicit system component and store the result in `r`.

This does *not* update any of the values stored in `component`
"""
residuals!(component::AbstractImplicitComponent, r, x, y)

function residuals!(component::ImplicitComponent, r, x, y)
    component.f(r, x, y)
    return r
end

function residuals!(component::ImplicitSystem, r, x, y)
    subcomponents = component.components
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in subcomponents]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in subcomponents]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in subcomponents]
    return residuals!(component, r, xsub, ysub, rsub, x, y)
end

"""
    residuals!(component::ImplicitSystem, r, xsub, ysub, rsub, x, y)

Evaluate the residuals of an implicit system and store the result in y.
Store the intermediate results in `xsub`, `ysub`, and `rsub`.

This does *not* update any of the values stored in `component`
"""
function residuals!(component::ImplicitSystem, r, xsub, ysub, rsub, x, y)
    subcomponent_residuals!(component, xsub, ysub, rsub, x, y)
    update_system_residuals!(component, r, rsub)
    return r
end

"""
    residuals!(component::AbstractImplicitComponent, x, y)

Evaluate the residuals of an implicit system component.

Return the result and store in `component.r`
"""
function residuals!(component::AbstractImplicitComponent, x, y)
    if (x != component.x_f) || (y != component.y_f)
        residuals!!(component, x, y)
    end
    return residuals(component)
end

"""
    residuals!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residuals of an implicit system component.

Return the result and store in `component.r`
"""
residuals!!(component::AbstractImplicitComponent, x, y)

function residuals!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    component.f(component.r, x, y)
    return residuals(component)
end

function residuals!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    subcomponent_residuals!(component, x, y)
    update_system_residuals!(component)
    return residuals(component)
end

"""
    residuals!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residuals of an implicit system component. Also force
(re-)evaluation of the residuals of the system's subcomponents.

Return the result and store in `component.r`
"""
residuals!!!(component::AbstractImplicitComponent, x, y)

residuals!!!(component::ImplicitComponent, x, y) = residuals!!(component, x, y)

function residuals!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    subcomponent_residuals!!(component, x, y)
    update_system_residuals!(component)
    return residuals(component)
end

"""
    residual_input_jacobian(component::AbstractImplicitComponent)

Return the jacobian of the residuals with respect to the inputs stored in `component`.
"""
residual_input_jacobian(component::AbstractImplicitComponent)
residual_input_jacobian(component::ImplicitComponent) = component.drdx
residual_input_jacobian(component::ImplicitSystem) = component.drdx

"""
    residual_input_jacobian(component::AbstractImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the inputs,

This does *not* update any of the values stored in `component`
"""
residual_input_jacobian(component::AbstractImplicitComponent, x, y)

function residual_input_jacobian(component::AbstractImplicitComponent, x, y)
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    return residual_input_jacobian!(component, drdx, x, y)
end

function residual_input_jacobian(component::ExplicitSystem, x, y)
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    dsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    return residual_input_jacobian!(component, drdx, xsub, ysub, dsub, x, y)
end

"""
    residual_input_jacobian!(component::ImplicitComponent, drdx, x, y)

Evaluate the jacobian of the residuals with respect to the inputs and store
the result in `drdx`.

This does *not* update any of the values stored in `component`
"""
residual_input_jacobian!(component::ImplicitComponent, drdx, x, y)

function residual_input_jacobian!(component::ImplicitComponent, drdx, x, y)
    component.dfdx(drdx, x, y)
    return drdx
end

function residual_input_jacobian!(component::ImplicitSystem, drdx, x, y)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    dsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    residual_input_jacobian!(component, drdx, xsub, ysub, dsub, x, y)
    return drdx
end

"""
    residual_input_jacobian!(component::ImplicitSystem, drdx, xsub, ysub, rsub,
        dsub, x, y)

Evaluate the jacobian of the residual with respect to the inputs and store the
result in `drdx`.  Store the intermediate subcomponent `x`, `y` and `drdx` values
in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
function residual_input_jacobian!(component::ImplicitSystem, drdx, xsub, ysub,
    dsub, x, y)
    subcomponent_input_jacobians!(component, xsub, ysub, dsub, x, y)
    update_system_input_jacobian!(component, drdx, dsub)
    return drdx
end

"""
    residual_input_jacobian!(component::AbstractImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the inputs.

Return the result and store in `component.drdx`
"""
function residual_input_jacobian!(component::AbstractImplicitComponent, x, y)
    if (x != component.x_dfdx) || (y != component.y_dfdx)
        residual_input_jacobian!!(component, x, y)
    end
    return residual_input_jacobian(component)
end

"""
    residual_input_jacobian!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the jacobian of the residuals with respect to the inputs.

Return the result and store in `component.drdx`
"""
residual_input_jacobian!!(component::ImplicitComponent, x, y)

function residual_input_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    component.dfdx(component.drdx, x, y)
    return residual_input_jacobian(component)
end

function residual_input_jacobian!!(component::ImplicitSystem, x, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    subcomponent_input_jacobians!(component, x, y)
    update_system_input_jacobian!(component)
    return residual_input_jacobian(component)
end

"""
    residual_input_jacobian!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the jacobian of the residuals with respect to the inputs.
Also force (re-)evaluation of the jacobians of the system's subcomponents.

Return the result and store in `component.drdx`
"""
residual_input_jacobian!!!(component::AbstractImplicitComponent, x, y)

residual_input_jacobian!!!(component::ImplicitComponent, x, y) =
    residual_input_jacobian!!(component, x, y)

function residual_input_jacobian!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    subcomponent_input_jacobians!!(component, x, y)
    update_system_input_jacobian!(component)
    return residual_input_jacobian(component)
end

"""
    residuals_and_input_jacobian(component::AbstractImplicitComponent)

Return the residual and jacobian of the residuals with respect to the inputs stored in `component`.
"""
residuals_and_input_jacobian(component::AbstractImplicitComponent) =
    residuals(component), residual_input_jacobian(component)

"""
    residuals_and_input_jacobian(component::AbstractImplicitComponent, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs.

This does *not* update any of the values stored in `component`
"""
residuals_and_input_jacobian(component::AbstractImplicitComponent, x, y)

function residuals_and_input_jacobian(component::ImplicitComponent, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    return residuals_and_input_jacobian!(component, r, drdx, x, y)
end

function residuals_and_input_jacobian(component::ImplicitSystem, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_input_jacobian!(component, r, drdx, xsub, ysub, rsub, dsub, x, y)
    return r, drdx
end

"""
    residuals_and_input_jacobian!(component::AbstractImplicitComponent, r, drdx, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs
and store the result in `r` and `drdx`.

This does *not* update any of the values stored in `component`
"""
residuals_and_input_jacobian!(component::AbstractImplicitComponent, r, drdx, x, y)

function residuals_and_input_jacobian!(component::ImplicitComponent, r, drdx, x, y)
    component.fdfdx(r, drdx, x, y)
    return r, drdx
end

function residuals_and_input_jacobian!(component::ImplicitSystem, r, drdx, x, y)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_input_jacobian!(component, r, drdx, xsub, ysub, rsub, dsub, x, y)
    return r, drdx
end

"""
    residuals_and_input_jacobian!(component::ImplicitSystem, r, drdx, xsub,
        ysub, rsub, dsub, x, y)

Evaluate the residuals and their derivatives with respect to the inputs and store
the results in `r` and `drdx`.  Store the intermediate subcomponent `x`, `y`, `r`,
and `drdx` values in `xsub`, `ysub`, `rsub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
function residuals_and_input_jacobian!(component::ImplicitSystem, r, drdx, xsub,
    ysub, rsub, dsub, x, y)
    subcomponent_residuals_and_input_jacobians!(component, xsub, ysub, rsub, dsub, x, y)
    update_system_residuals!(component, r, rsub)
    update_system_input_jacobian!(component, drdx, dsub)
    return r, drdx
end

"""
    residuals_and_input_jacobian!(component::AbstractImplicitComponent, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs.

Return the result and store in `component.r` and `component.drdx`
"""
function residuals_and_input_jacobian!(component::AbstractImplicitComponent, x, y)
    if  (x != component.x_f) || (y != component.y_f) && (x != component.x_dfdx) || (y != component.y_dfdx)
        residuals_and_input_jacobian!!(component, x, y)
    elseif (x != component.x_f) || (y != component.y_f)
        residuals!!(component, x, y)
    elseif (x != component.x_dfdx) || (y != component.y_dfdx)
        residual_input_jacobian!!(component, x, y)
    end
    return residuals_and_input_jacobian(component)
end

"""
    residuals_and_input_jacobian!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residual and jacobian of the residuals with respect to the inputs.

Return the result and store in `component.r` and `component.drdx`
"""
residuals_and_input_jacobian!!(component::AbstractImplicitComponent, x, y)

function residuals_and_input_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    component.fdfdx(component.r, component.drdx, x, y)
    return residuals_and_input_jacobian(component)
end

function residuals_and_input_jacobian!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    subcomponent_residuals_and_input_jacobians!(component, x, y)
    update_system_residuals!(component)
    update_system_input_jacobian!(component)
    return residuals_and_input_jacobian(component)
end

"""
    residuals_and_input_jacobian!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residual and jacobian of the residuals with respect
to the inputs. Also force (re-)evaluation of each subcomponents residual and
jacobian with respect to its inputs.

Return the result and store in `component.r` and `component.drdx`
"""
residuals_and_input_jacobian!!!(component::AbstractImplicitComponent, x, y)

residuals_and_input_jacobian!!!(component::ImplicitComponent, x, y) =
    residuals_and_input_jacobian!!(component::ImplicitComponent, x, y)

function residuals_and_input_jacobian!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    subcomponent_residuals_and_input_jacobians!!(component, x, y)
    update_system_residuals!(component)
    update_system_input_jacobian!(component)
    return residuals_and_input_jacobian(component)
end

"""
    residual_output_jacobian(component::AbstractImplicitComponent)

Return the jacobian of the residuals with respect to the outputs stored in `component`.
"""
residual_output_jacobian(component::AbstractImplicitComponent)
residual_output_jacobian(component::ImplicitComponent) = component.drdy
residual_output_jacobian(component::ImplicitSystem) = component.drdy

"""
    residual_output_jacobian(component::AbstractImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the outputs,

This does *not* update any of the values stored in `component`
"""
residual_output_jacobian(component::AbstractImplicitComponent, x, y)

function residual_output_jacobian(component::ImplicitComponent, x, y)
    drdy = similar(component.drdy, promote_type(eltype(component.drdy), eltype(x), eltype(y)))
    return residual_output_jacobian!(component, drdy, x, y)
end

function residual_output_jacobian(component::ImplicitSystem, x, y)
    drdy = similar(component.drdy, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    return residual_output_jacobian!(component, drdy, xsub, ysub, dxsub, dysub, x, y)
end

"""
    residual_output_jacobian!(component::AbstractImplicitComponent, drdy, x, y)

Evaluate the jacobian of the residuals with respect to the outputs and store
the result in `drdy`

This does *not* update any of the values stored in `component`
"""
residual_output_jacobian!(component::AbstractImplicitComponent, drdy, x, y)

function residual_output_jacobian!(component::ImplicitComponent, drdy, x, y)
    component.dfdy(drdy, x, y)
    return drdy
end

function residual_output_jacobian!(component::ImplicitSystem, drdy, x, y)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    residual_output_jacobian!(component, drdy, xsub, ysub, dxsub, dysub, x, y)
    return drdy
end

"""
    residual_output_jacobian!(component::ImplicitSystem, drdy, xsub, ysub, dxsub,
        dysub, x, y)

Evaluate the jacobian of the residuals with respect to the outputs and store the
result in `drdx`.  Store the intermediate subcomponent `x`, `y` and `drdy` values
in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
function residual_output_jacobian!(component::ImplicitSystem, drdy, xsub, ysub,
    dxsub, dysub, x, y)
    subcomponent_input_jacobians!(component, xsub, ysub, dxsub, x, y)
    subcomponent_output_jacobians!(component, xsub, ysub, dysub, x, y)
    update_system_output_jacobian!(component, drdy, dxsub, dysub)
    return drdy
end

"""
    residual_output_jacobian!(component::AbstractImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the outputs.

Return the result and store in `component.drdy`
"""
function residual_output_jacobian!(component::AbstractImplicitComponent, x, y)
    if (x != component.x_dfdy) || (y != component.y_dfdy)
        residual_output_jacobian!!(component, x, y)
    end
    return residual_output_jacobian(component)
end

"""
    residual_output_jacobian!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the jacobian of the residuals with respect to the outputs.

Return the result and store in `component.drdy`
"""
residual_output_jacobian!!(component::AbstractImplicitComponent, x, y)

function residual_output_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    component.dfdy(component.drdy, x, y)
    return residual_output_jacobian(component)
end

function residual_output_jacobian!!(component::ImplicitSystem, x, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    subcomponent_input_jacobians!(component, x, y)
    subcomponent_output_jacobians!(component, x, y)
    update_system_output_jacobian!(component)
    return residual_output_jacobian(component)
end

"""
    residual_output_jacobian!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the jacobian of the residuals with respect to the outputs.
Also force (re-)evaluation of the jacobians of the system's subcomponents.

Return the result and store in `component.drdy`
"""
residual_output_jacobian!!!(component::AbstractImplicitComponent, x, y)

residual_output_jacobian!!!(component::ImplicitComponent, x, y) =
    residual_output_jacobian!!(component, x, y)

function residual_output_jacobian!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    subcomponent_input_jacobians!!(component, x, y)
    subcomponent_output_jacobians!!(component, x, y)
    update_system_output_jacobian!(component)
    return residual_output_jacobian(component)
end

"""
    residuals_and_output_jacobian(component::AbstractImplicitComponent)

Return the residuals and jacobian of the residuals with respect to the outputs
stored in `component`.
"""
residuals_and_output_jacobian(component::AbstractImplicitComponent) =
    residuals(component), residual_output_jacobian(component)

"""
    residuals_and_output_jacobian(component::AbstractImplicitComponent, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs.

This does *not* update any of the values stored in `component`
"""
residuals_and_output_jacobian(component::AbstractImplicitComponent, x, y)

function residuals_and_output_jacobian(component::ImplicitComponent, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdy = similar(component.drdy, promote_type(eltype(component.drdy), eltype(x), eltype(y)))
    return residuals_and_output_jacobian!(component, r, drdy, x, y)
end

function residuals_and_output_jacobian(component::ImplicitSystem, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdy = similar(component.drdy, promote_type(eltype(component.drdy), eltype(x), eltype(y)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_output_jacobian!(component, r, drdy, xsub, ysub, rsub, dxsub, dysub, x, y)
    return r, drdy
end

"""
    residuals_and_output_jacobian!(component::AbstractImplicitComponent, r, drdy, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs
and store the results in `r` and `drdy`

This does *not* update any of the values stored in `component`
"""
residuals_and_output_jacobian!(component::AbstractImplicitComponent, r, drdy, x, y)

function residuals_and_output_jacobian!(component::ImplicitComponent, r, drdy, x, y)
    component.fdfdy(r, drdy, x, y)
    return r, drdy
end

function residuals_and_output_jacobian!(component::ImplicitSystem, r, drdy, x, y)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_output_jacobian!(component, r, drdy, xsub, ysub, rsub, dxsub, dysub, x, y)
    return r, drdy
end

"""
    residuals_and_output_jacobian!(component::ImplicitSystem, r, drdy, xsub,
        ysub, rsub, dxsub, dysub, x, y)

Evaluate the residuals and their derivatives with respect to the outputs and store
the results in `r` and `drdy`.  Store the intermediate subcomponent `x`, `y`, `r`,
`drdx`, and `drdy` values in `xsub`, `ysub`, `rsub`, `dxsub`, and `dysub` respectively.

This does *not* update any of the values stored in `component`
"""
function residuals_and_output_jacobian!(component::ImplicitSystem, r, drdy, xsub,
    ysub, rsub, dxsub, dysub, x, y)
    subcomponent_residuals_and_jacobians!(component, xsub, ysub, rsub, dxsub, dysub, x, y)
    update_system_residuals!(component, r, rsub)
    update_system_output_jacobian!(component, drdy, dxsub, dysub)
    return r, drdy
end

"""
    residuals_and_output_jacobian!(component::AbstractImplicitComponent, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs.

Return the result and store in `component.r` and `component.drdy`
"""
function residuals_and_output_jacobian!(component::AbstractImplicitComponent, x, y)
    if  (x != component.x_f) || (y != component.y_f) && (x != component.x_dfdy) || (y != component.y_dfdy)
        residuals_and_output_jacobian!!(component, x, y)
    elseif (x != component.x_f) || (y != component.y_f)
        residuals!!(component, x, y)
    elseif (x != component.x_dfdy) || (y != component.y_dfdy)
        residual_output_jacobian!!(component, x, y)
    end
    return residuals_and_output_jacobian(component)
end

"""
    residuals_and_output_jacobian!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residuals and jacobian of the residuals with respect to the outputs.

Return the result and store in `component.r` and `component.drdy`
"""
residuals_and_output_jacobian!!(component::AbstractImplicitComponent, x, y)

function residuals_and_output_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    component.fdfdy(component.r, component.drdy, x, y)
    return residuals_and_output_jacobian(component)
end

function residuals_and_output_jacobian!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    subcomponent_residuals_and_jacobians!(component, x, y)
    update_system_residuals!(component)
    update_system_output_jacobian!(component)
    return residuals_and_output_jacobian(component)
end

"""
    residuals_and_output_jacobian!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residual and jacobian of the residuals with respect
to the outputs. Also force (re-)evaluation of each subcomponents residual and
jacobian with respect to its outputs.

Return the result and store in `component.r` and `component.drdy`
"""
residuals_and_output_jacobian!!!(component::AbstractImplicitComponent, x, y)

residuals_and_output_jacobian!!!(component::ImplicitComponent, x, y) =
    residuals_and_output_jacobian!!(component::ImplicitComponent, x, y)

function residuals_and_output_jacobian!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    subcomponent_residuals_and_jacobians!!(component, x, y)
    update_system_residuals!(component)
    update_system_output_jacobian!(component)
    return residuals_and_output_jacobian(component)
end

"""
    residuals_and_jacobians(component::AbstractImplicitComponent)

Return the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs stored in `component`.
"""
residuals_and_jacobians(component::AbstractImplicitComponent) = residuals(component),
    residual_input_jacobian(component), residual_output_jacobian(component)

"""
    residuals_and_jacobians(component::AbstractImplicitComponent, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs.

This does *not* update any of the values stored in `component`
"""
residuals_and_jacobians(component::AbstractImplicitComponent, x, y)

function residuals_and_jacobians(component::ImplicitComponent, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    drdy = similar(component.drdy, promote_type(eltype(component.drdy), eltype(x), eltype(y)))
    return residuals_and_jacobians!(component, r, drdx, drdy, x, y)
end

function residuals_and_jacobians(component::ImplicitSystem, x, y)
    r = similar(component.r, promote_type(eltype(component.r), eltype(x), eltype(y)))
    drdx = similar(component.drdx, promote_type(eltype(component.drdx), eltype(x), eltype(y)))
    drdy = similar(component.drdy, promote_type(eltype(component.drdy), eltype(x), eltype(y)))
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_jacobians!(component, r, drdx, drdy, xsub, ysub, rsub,
        dxsub, dysub, x, y)
    return r, drdx, drdy
end

"""
    residuals_and_jacobians!(component::AbstractImplicitComponent, r, drdx, drdy, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs and store the results in `r`, `drdx`, and `drdy`

This does *not* update any of the values stored in `component`
"""
residuals_and_jacobians!(component::AbstractImplicitComponent, r, drdx, drdy, x, y)

function residuals_and_jacobians!(component::ImplicitComponent, r, drdx, drdy, x, y)
    component.fdf(r, drdx, drdy, x, y)
    return r, drdx, drdy
end

function residuals_and_jacobians!(component::ImplicitSystem, r, drdx, drdy, x, y)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x), eltype(y))) for comp in component.components]
    ysub = [similar(comp.y_f, promote_type(eltype(comp.y_f), eltype(x), eltype(y))) for comp in component.components]
    rsub = [similar(comp.r, promote_type(eltype(comp.r), eltype(x), eltype(y))) for comp in component.components]
    dxsub = [similar(comp.drdx, promote_type(eltype(comp.drdx), eltype(x), eltype(y))) for comp in component.components]
    dysub = [similar(comp.drdy, promote_type(eltype(comp.drdy), eltype(x), eltype(y))) for comp in component.components]
    residuals_and_jacobians!(component, r, drdx, drdy, xsub, ysub, rsub,
        dxsub, dysub, x, y)
    return r, drdx, drdy
end

"""
    residuals_and_jacobians!(component::ImplicitSystem, r, drdx, drdy, xsub,
        ysub, rsub, dxsub, dysub, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs and store the results in `r`, `drdx`, and `drdy`
Store the intermediate subcomponent `x`, `y`, `r`, `drdx`, and `drdy` values in
`xsub`, `ysub`, `rsub`, `dxsub`, and `dysub` respectively.

This does *not* update any of the values stored in `component`
"""
function residuals_and_jacobians!(component::ImplicitSystem, r, drdx, drdy, xsub,
    ysub, rsub, dxsub, dysub, x, y)
    subcomponent_residuals_and_jacobians!(component, xsub, ysub, rsub, dxsub, dysub, x, y)
    update_system_residuals!(component, r, rsub)
    update_system_input_jacobian!(component, drdx, dxsub)
    update_system_output_jacobian!(component, drdy, dxsub, dysub)
    return r, drdx, drdy
end

"""
    residuals_and_jacobians!(component::AbstractImplicitComponent, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs.

Return the result and store in `component.r`, `component.drdx`, and `component.drdy`.
"""
function residuals_and_jacobians!(component::AbstractImplicitComponent, x, y)

    # calculate everything?
    if  (x != component.x_f) || (y != component.y_f) &&
        (x != component.x_dfdx) || (y != component.y_dfdx) &&
        (x != component.x_dfdy) || (y != component.y_dfdy)

        residuals_and_jacobians!!(component, x, y)
    else
        # calculate residuals?
        if (x != component.x_f) || (y != component.y_f)
            residuals!!(component, x, y)
        end
        # calculate input jacobian?
        if (x != component.x_dfdx) || (y != component.y_dfdx)
            residual_input_jacobian!!(component, x, y)
        end
        # calculate output jacobian?
        if (x != component.x_dfdy) || (y != component.y_dfdy)
            residual_output_jacobian!!(component, x, y)
        end
    end

    return residuals_and_jacobians(component)
end

"""
    residuals_and_jacobians!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residual, its jacobian with respect to the inputs,
and its jacobian with respect to the outputs.

Return the result and store in `component.r`, `component.drdx`, and `component.drdy`.
"""
residuals_and_jacobians!!(component::AbstractImplicitComponent, x, y)

function residuals_and_jacobians!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    component.fdf(component.r, component.drdx, component.drdy, x, y)
    return residuals_and_jacobians(component)
end

function residuals_and_jacobians!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    subcomponent_residuals_and_jacobians!(component, x, y)
    update_system_residuals!(component)
    update_system_input_jacobian!(component)
    update_system_output_jacobian!(component)
    return residuals_and_jacobians(component)
end

"""
    residuals_and_jacobians!!!(component::AbstractImplicitComponent, x, y)

Force (re-)evaluation of the residual, its jacobian with respect to the inputs,
and its jacobian with respect to the outputs. Also force (re-)evaluation of each
subcomponents residual and jacobians.

Return the result and store in `component.r`, `component.drdx`, and `component.drdy`.
"""
residuals_and_jacobians!!!(component::AbstractImplicitComponent, x, y)

residuals_and_jacobians!!!(component::ImplicitComponent, x, y) =
    residuals_and_jacobians!!(component, x, y)

function residuals_and_jacobians!!!(component::ImplicitSystem, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    subcomponent_residuals_and_jacobians!!(component, x, y)
    update_system_residuals!(component)
    update_system_input_jacobian!(component)
    update_system_output_jacobian!(component)
    return residuals_and_jacobians(component)
end
