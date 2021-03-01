abstract type AbstractComponent end

abstract type AbstractExplicitComponent end

"""
    ExplicitComponent{TX, TY, TJ} <: AbstractExplicitComponent

System component defined by the explicit vector-valued output function: `y = f(x)`

# Fields
 - `f`: In-place output function `f(y, x)`
 - `df`: In-place jacobian function `df(dydx, x)`
 - `fdf`: In-place combined output and jacobian function `fdf(y, dydx, x)`
 - `x_f::TX`: Inputs used to evaluate the outputs
 - `x_df::TX`: Inputs used to evaluate the jacobian
 - `y::TY`: Outputs
 - `dydx::TJ`: Jacobian
"""
struct ExplicitComponent{TX, TY, TJ} <: AbstractExplicitComponent
    f
    df
    fdf
    x_f::TX
    x_df::TX
    y::TY
    dydx::TJ
end

"""
    ExplicitSystem{TC, TX, TY, TJ, TD} <: AbstractExplicitComponent

Explicit system constructed from a chain of explicit system components called
sequentially.

# Fields
 - `components::TC`: Collection of components, in calling order
 - `input_mapping::Vector{NTuple{2,Vector{Int}}}`:
 - `component_output_mapping::Vector{Vector{NTuple{2,Vector{Int}}}}`:
 - `component_input_mapping::Vector{Vector{NTuple{2,Int}}}`:
 - `output_mapping::Vector{NTuple{2,Int}}`:
 - `x_f::TX`: Inputs used to evaluate the system outputs
 - `x_df::TX`: Inputs used to evaluate the system jacobian
 - `y::TY`: Storage for the system outputs
 - `dydx::TJ`: Storage for the system jacobian
 - `mode::TD`
"""
struct ExplicitSystem{TC, TX, TY, TJ, TD} <: AbstractExplicitComponent
    components::TC
    input_mapping::Vector{NTuple{2,Vector{Int}}}
    component_output_mapping::Vector{Vector{NTuple{2,Vector{Int}}}}
    component_input_mapping::Vector{Vector{NTuple{2,Int}}}
    output_mapping::Vector{NTuple{2,Int}}
    x_f::TX
    x_df::TX
    y::TY
    dydx::TJ
    mode::TD
end

"""
    outputs(component::AbstractExplicitComponent)

Return the outputs stored in `component`.
"""
outputs(component::AbstractExplicitComponent) = component.y

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

Evaluate the outputs of an explicit system component and store the result in y.
Store the intermediate results in `xsub` and `ysub`.

This does *not* update any of the values stored in `component`
"""
function outputs!(component::ExplicitSystem, y, xsub, ysub, x)
    subcomponent_outputs!(component, xsub, ysub, x)
    update_system_outputs!(component, y, x, ysub)
    return y
end

"""
    subcomponent_outputs!(component::ExplicitSystem, xsub, ysub, x)

Evaluate the inputs and outputs of an explicit systems subcomponents and store
the results in `xsub` and `ysub`.

This does *not* update any of the values stored in `component`
"""
function subcomponent_outputs!(component, xsub, ysub, x)
    # extract subcomponents
    subcomponents = component.components

    # loop through components
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                xsub[ic][ix] = x[jy]
            else
                xsub[ic][ix] = ysub[jc][jy]
            end
        end
        # update outputs for current subcomponent
        outputs!(subcomponent, ysub[ic], xsub[ic])
    end

    return xsub, ysub
end

"""
    update_system_outputs!(component, y, x, ysub=nothing)

Update the outputs `y` given the inputs `x` and the component outputs `ysub`.
If `ysub` is nothing use the value of the outputs stored in each subcomponent.
"""
function update_system_outputs!(component, y, x, ysub=nothing)
    # update system outputs
    for iy = 1:length(y)
        jc, jy = component.output_mapping[iy]
        if jc == 0
            y[iy] = x[jy]
        else
            if isnothing(ysub)
                y[iy] = outputs(component.components[jc])[jy]
            else
                y[iy] = ysub[jc][jy]
            end
        end
    end
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
    update_system_outputs!(component, component.y, x)
    return outputs(component)
end

"""
    subcomponent_outputs!(component::ExplicitSystem, x)

Evaluate the inputs and outputs of an explicit system's subcomponents.

Store in the `x_f` and `y` fields of each subcomponent.
"""
function subcomponent_outputs!(component, x)

    # unpack subcomponents
    subcomponents = component.components

    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # flag for whether to update a subcomponent
        update = false
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != x[jy]
                # set value from system input
                xsub[ix] = x[jy]
            else
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != outputs(subcomponents[jc])[jy]
                # set value from subcomponent output
                xsub[ix] = outputs(subcomponents[jc])[jy]
            end
        end
        # update outputs for current subcomponent (if necessary)
        if update
            outputs!!(subcomponent, xsub)
        end
    end

    return nothing
end

"""
    jacobian(component::AbstractExplicitComponent)

Return the jacobian of the outputs with respect to the inputs stored in `component`
"""
jacobian(component::AbstractExplicitComponent) = component.dydx

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

function jacobian!(component::ExplicitSystem, dydx, x)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    jacobian!(component, dydx, xsub, ysub, dsub, x)
    return dydx
end

"""
    jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x; mode=component.mode)

Evaluate the jacobian of the outputs with respect to the inputs and store the
result in `dydx`.  Store the intermediate subcomponent `x`, `y` and `dydx` values
in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x; mode=component.mode) =
    jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, mode)

function jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Forward)
    subcomponent_jacobians!(component, xsub, ysub, dsub, x)
    forward_mode_jacobian!(component, dydx, dsub)
    return dydx
end

function jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Reverse)
    subcomponent_jacobians!(component, xsub, ysub, dsub, x)
    reverse_mode_jacobian!(component, dydx, dsub)
    return dydx
end

"""
    subcomponent_jacobians!(component::ExplicitSystem, xsub, ysub, dsub, x)

Evaluate the inputs, outputs, and jacobians of an explicit system's subcomponents.
Store the results in xsub, ysub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_jacobians!(component::ExplicitSystem, xsub, ysub, dsub, x)

    # extract subcomponents
    subcomponents = component.components
    # loop through components
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                xsub[ic][ix] = x[jy]
            else
                xsub[ic][ix] = ysub[jc][jy]
            end
        end
        # update outputs and jacobians for current subcomponent
        outputs_and_jacobian!(subcomponent, ysub[ic], dsub[ic], xsub[ic])
    end

    return xsub, ysub, dsub
end

"""
    forward_mode_jacobian!(component::ExplicitSystem, dydx, dsub=nothing)

Fills in the system jacobian matrix `dydx` using the chain rule in forward mode.

If `dsub` is not provided, jacobian matrices will be taken from each subcomponent's
internal storage.
"""
function forward_mode_jacobian!(system::ExplicitSystem, dydx, dsub=nothing)

    # get output dimensions
    ny, nx = size(dydx)

    # initialize output to zeros
    dydx .= 0.0

    # loop through each input
    for ix = 1:nx
        # start new chain rule product
        cprod = 1
        # get the destinations for this input
        jc, jx = system.input_mapping[ix]
        # loop through each destination
        for k = 1:length(jc)
            # check the identity of the destination
            if jc[k] > length(system.components)
                # destination is a system output
                iy = jx[k]
                # add chain rule product to corresponding entry in jacobian
                dydx[iy, ix] += cprod
            else
                # destination is a subcomponent
                forward_mode_jacobian_branch!(system, dydx, ix, cprod, jc[k], jx[k], dsub)
            end
        end
    end

    return dydx
end

"""
    forward_mode_jacobian_branch!(system, dydx, ix, cprod, jc, jx, dsub=nothing)

Adds to the jacobian matrix using the chain rule in forward mode.

This function is called recursively to create new branches as necessary.

If `dsub` is not provided, jacobian matrices will be taken from each subcomponent's
internal storage.
"""
function forward_mode_jacobian_branch!(system, dydx, ix, cprod, jc, jx,
    dsub=nothing)

    # extract subcomponent
    subcomponent = system.components[jc]
    # loop through outputs from this subcomponent
    for jy = 1:length(subcomponent.y)
        # create new branch with new chain rule product
        new_cprod = cprod
        # multiply by current jacobian entry
        if isnothing(dsub)
            new_cprod *= jacobian(subcomponent)[jy, jx]
        else
            new_cprod *= dsub[jc][jy, jx]
        end
        # get the destinations for this output
        new_jc, new_jx = system.component_output_mapping[jc][jy]
        # loop through each destination
        for k = 1:length(new_jc)
            # check if we're done with this branch
            if new_jc[k] > length(system.components)
                # destination is a system output
                iy = new_jx[k]
                # add chain rule product to corresponding slot in jacobian
                dydx[iy, ix] += new_cprod
            elseif iszero(new_cprod)
                # stop computing branch because there is nothing to add
            else
                # move on to next subcomponent in chain
                forward_mode_jacobian_branch!(system, dydx, ix, new_cprod,
                    new_jc[k], new_jx[k], dsub)
            end
        end
    end
end

"""
    reverse_mode_jacobian!(component::ExplicitSystem, dydx, dsub=nothing)

Fills in the system jacobian matrix `dydx` using the chain rule in reverse mode.

If `dsub` is not provided, jacobian matrices will be taken from each subcomponent's
internal storage.
"""
function reverse_mode_jacobian!(component::ExplicitSystem, dydx, dsub = nothing)

    # get output dimensions
    ny, nx = size(dydx)

    # initialize output to zeros
    dydx .= 0.0

    # loop through each output
    for iy = 1:ny
        # start new chain rule product
        cprod = 1
        # get the source of this system output
        jc, jy = component.output_mapping[iy]
        # check the identity of the source
        if iszero(jc)
            # source is the system input
            ix = jy
            # add chain rule product to corresponding entry in jacobian
            dydx[iy, ix] += cprod
        else
            # source is a subcomponent, add its contribution to the jacobian
            reverse_mode_jacobian_branch!(component, dydx, iy, cprod, jc, jy, dsub)
        end
    end

    return dydx
end

"""
    reverse_mode_jacobian_branch!(component::ExplicitSystem, dydx, iy, cprod, jc,
        jy, dsub=nothing)

Adds to the jacobian matrix using the chain rule in reverse mode.

This function is called recursively to create new branches as necessary.

If `dsub` is not provided, jacobian matrices will be taken from each subcomponent's
internal storage.
"""
function reverse_mode_jacobian_branch!(system, dydx, iy, cprod, jc, jy, dsub=nothing)
    # extract subcomponent
    subcomponent = system.components[jc]
    # loop through inputs to this subcomponent
    for jx = 1:length(subcomponent.x_f)
        # create new branch with new chain rule product
        new_cprod = cprod
        # multiply by current jacobian entry
        if isnothing(dsub)
            new_cprod *= jacobian(subcomponent)[jy, jx]
        else
            new_cprod *= dsub[jc][jy, jx]
        end
        # get the source and index of this input
        new_jc, new_jy = system.component_input_mapping[jc][jx]
        # check if we're done
        if iszero(new_jc)
            # input corresponds to system input
            ix = new_jy
            # add chain rule product to corresponding slot in jacobian
            dydx[iy, ix] = new_cprod
        elseif iszero(new_cprod)
            # stop computing branch because there is nothing to add
        else
            # move on to next subcomponent in chain
            reverse_mode_jacobian_branch!(system, dydx, iy, new_cprod, new_jc, new_jy, dsub)
        end
    end
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

jacobian!!(component::ExplicitSystem, x; mode = component.mode) =
    jacobian!!(component::ExplicitSystem, x, mode)

function jacobian!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_df, x)
    subcomponent_jacobians!(component, x)
    forward_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

function jacobian!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_df, x)
    subcomponent_jacobians!(component, x)
    reverse_mode_jacobian!(component, component.dydx)
    return jacobian(component)
end

"""
    subcomponent_jacobians!(component::ExplicitSystem, x)

Evaluate the inputs, outputs, and jacobians of an explicit system's subcomponents.

Store in the `x_f`, `y`, and `dydx` fields of each subcomponent.
"""
function subcomponent_jacobians!(component::ExplicitSystem, x)

    # unpack subcomponents
    subcomponents = component.components

    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_df
        # flag for whether to update a subcomponent
        update = false
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != x[jy]
                # set value from system input
                xsub[ix] = x[jy]
            else
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != outputs(subcomponents[jc])[jy]
                # set value from subcomponent output
                xsub[ix] = outputs(subcomponents[jc])[jy]
            end
        end
        # update outputs and jacobians for current subcomponent (if necessary)
        if update
            outputs_and_jacobian!!(subcomponent, xsub)
        end
    end

    return nothing
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
    return dydx
end

function outputs_and_jacobian(component::ExplicitSystem, y, dydx, x)
    xsub = [similar(comp.x_f, promote_type(eltype(comp.x_f), eltype(x))) for comp in component.components]
    ysub = [similar(comp.y, promote_type(eltype(comp.y), eltype(x))) for comp in component.components]
    dsub = [similar(comp.dydx, promote_type(eltype(comp.dydx), eltype(x))) for comp in component.components]
    return outputs_and_jacobian!(component, y, dydx, xsub, ysub, dsub, x)
end

"""
    outputs_and_jacobian!(component::ExplicitSystem, y, dydx, xsub, ysub, dsub,
        x; mode=component.mode)

Evaluate the outputs and their derivatives with respect to the inputs and store
the results in `y` and `dydx`.  Store the intermediate subcomponent `x`, `y` and
`dydx` values in `xsub`, `ysub`, and `dsub` respectively.

This does *not* update any of the values stored in `component`
"""
outputs_and_jacobian!(component::AbstractExplicitComponent, y, dydx, xsub, ysub,
    dsub, x; mode=component.mode) = outputs_and_jacobian!(component, y, dydx,
    xsub, ysub, dsub, x, mode)

function outputs_and_jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Forward)
    subcomponent_jacobians!(component, xsub, ysub, dsub, x)
    forward_mode_jacobian!(component, dydx, dsub)
    update_system_outputs!(component, y, x, ysub)
    return y, dydx
end

function outputs_and_jacobian!(component::ExplicitSystem, dydx, xsub, ysub, dsub, x, ::Reverse)
    subcomponent_jacobians!(component, xsub, ysub, dsub, x)
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

outputs_and_jacobian!!(component::ExplicitSystem, x; mode=component.mode) =
    outputs_and_jacobian!!(component, x, mode)

function outputs_and_jacobian!!(component::ExplicitSystem, x, ::Forward)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_jacobians!(component, x)
    forward_mode_jacobian!(component, dydx)
    update_system_outputs!(component, y, x)
    return outputs_and_jacobian(component)
end

function outputs_and_jacobian!!(component::ExplicitSystem, x, ::Reverse)
    copyto!(component.x_f, x)
    copyto!(component.x_df, x)
    subcomponent_jacobians!(component, x)
    reverse_mode_jacobian!(component, dydx)
    update_system_outputs!(component, y, x)
    return outputs_and_jacobian(component)
end

"""
    ImplicitComponent{TX, TY, TR, TDRX, TDRY} <: AbstractComponent

System component defined by the vector-valued residual function: `0 = f(x, y)`

# Fields
 - `f`: In-place residual function `f(r, x, y)`.
 - `dfdx`: In-place residual jacobian function with respect to the inputs `dfdx(drdx, x, y)`
 - `dfdy`: In-place residual jacobian function with respect to the outputs `dfdy(drdy, x, y)`
 - `fdfdx`: In-place combined residual and jacobian with respect to the inputs function `fdfdx(r, drdx, x, y)`.
 - `fdfdy`: In-place combined residual and jacobian with respect to the outputs function `fdfdy(r, drdy, x, y)`.
 - `fdf`: In-place combined residual and jacobians function `fdf(r, drdx, drdy, x, y)`.
 - `x_f::TX`: `x` used to evaluate `f`
 - `y_f::TY`: `y` used to evaluate `f`
 - `x_dfdx::TX`: `x` used to evaluate `dfdx`
 - `y_dfdx::TY`: `y` used to evaluate `dfdx`
 - `x_dfdy::TX`: `x` used to evaluate `dfdy`
 - `y_dfdy::TY`: `y` used to evaluate `dfdy`
 - `r::TR`: cache for residual`
 - `drdx::TDRX`: cache for residual jacobian with respect to `x`
 - `drdy::TDRY`: cache for residual jacobian with respect to `y`
"""
struct ImplicitComponent{TX, TY, TR, TDRX, TDRY} <: AbstractComponent
    f
    dfdx
    dfdy
    fdfdx
    fdfdy
    fdf
    x_f::TX
    y_f::TY
    x_dfdx::TX
    y_dfdx::TY
    x_dfdy::TX
    y_dfdy::TY
    r::TR
    drdx::TDRX
    drdy::TDRY
end

"""
    residuals(component::ImplicitComponent)

Return the residuals stored in `component`
"""
residuals(component::ImplicitComponent) = component.r

"""
    residuals(component::ImplicitComponent, x, y)

Evaluate the residuals of an implicit system component.

This does *not* update any of the values stored in `component`
"""
function residuals(component::ImplicitComponent, x, y)
    TR = promote_type(eltype(component.r), eltype(x), eltype(y))
    r = similar(component.r, TR)
    return residuals!(component, r, x, y)
end

"""
    residuals!(component::ImplicitComponent, r, x, y)

Evaluate the residuals of an implicit system component and store the result in `r`.

This does *not* update any of the values stored in `component`
"""
function residuals!(component::ImplicitComponent, r, x, y)
    component.f(r, x, y)
    return r
end

"""
    residuals!(component::ImplicitComponent, x, y)

Evaluate the residuals of an implicit system component.

Return the result and store in `component.r`
"""
function residuals!(component::ImplicitComponent, x, y)
    if (x != component.x_f) || (y != component.y_f)
        residuals!!(component, x, y)
    end
    return residuals(component)
end

"""
    residuals!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the residuals of an implicit system component.

Return the result and store in `component.r`
"""
function residuals!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    component.f(component.r, x, y)
    return residuals(component)
end

"""
    residual_input_jacobian(component::ImplicitComponent)

Return the jacobian of the residuals with respect to the inputs stored in `component`.
"""
residual_input_jacobian(component::ImplicitComponent) = component.drdx

"""
    residual_input_jacobian(component::ImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the inputs,

This does *not* update any of the values stored in `component`
"""
function residual_input_jacobian(component::ImplicitComponent, x, y)
    TDRX = promote_type(eltype(component.drdx), eltype(x), eltype(y))
    drdx = similar(component.drdx, TDRX)
    return residual_input_jacobian!(component, drdx, x, y)
end

"""
    residual_input_jacobian!(component::ImplicitComponent, drdx, x, y)

Evaluate the jacobian of the residuals with respect to the inputs and store
the result in `drdx`.

This does *not* update any of the values stored in `component`
"""
function residual_input_jacobian!(component::ImplicitComponent, drdx, x, y)
    component.dfdx(drdx, x, y)
    return drdx
end

"""
    residual_input_jacobian!(component::ImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the inputs.

Return the result and store in `component.drdx`
"""
function residual_input_jacobian!(component::ImplicitComponent, x, y)
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
function residual_input_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    component.dfdx(component.drdx, x, y)
    return residual_input_jacobian(component)
end

"""
    residuals_and_input_jacobian(component::ImplicitComponent)

Return the residual and jacobian of the residuals with respect to the inputs stored in `component`.
"""
residuals_and_input_jacobian(component::ImplicitComponent) = component.r, component.drdx

"""
    residuals_and_input_jacobian(component::ImplicitComponent, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs.

This does *not* update any of the values stored in `component`
"""
function residuals_and_input_jacobian(component::ImplicitComponent, x, y)
    TR = promote_type(eltype(component.r), eltype(x), eltype(y))
    TDRX = promote_type(eltype(component.drdx), eltype(x), eltype(y))
    r = similar(component.r, TR)
    drdx = similar(component.drdx, TDRX)
    return residuals_and_input_jacobian!(component, r, drdx, x, y)
end

"""
    residuals_and_input_jacobian!(component::ImplicitComponent, r, drdx, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs
and store the result in `r` and `drdx`.

This does *not* update any of the values stored in `component`
"""
function residuals_and_input_jacobian!(component::ImplicitComponent, r, drdx, x, y)
    component.fdfdx(r, drdx, x, y)
    return r, drdx
end

"""
    residuals_and_input_jacobian!(component::ImplicitComponent, x, y)

Evaluate the residual and jacobian of the residuals with respect to the inputs.

Return the result and store in `component.r` and `component.drdx`
"""
function residuals_and_input_jacobian!(component::ImplicitComponent, x, y)
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
    residuals_and_input_jacobian!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the residual and jacobian of the residuals with respect to the inputs.

Return the result and store in `component.r` and `component.drdx`
"""
function residuals_and_input_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdx, x)
    copyto!(component.y_dfdx, y)
    component.fdfdx(component.r, component.drdx, x, y)
    return residuals_and_input_jacobian(component)
end

"""
    residual_output_jacobian(component::ImplicitComponent)

Return the jacobian of the residuals with respect to the outputs stored in `component`.
"""
residual_output_jacobian(component::ImplicitComponent) = component.drdy

"""
    residual_output_jacobian(component::ImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the outputs,

This does *not* update any of the values stored in `component`
"""
function residual_output_jacobian(component::ImplicitComponent, x, y)
    TDRY = promote_type(eltype(component.drdy), eltype(x), eltype(y))
    drdy = similar(component.drdy, TDRY)
    return residual_output_jacobian!(component, drdy, x, y)
end

"""
    residual_output_jacobian!(component::ImplicitComponent, drdy, x, y)

Evaluate the jacobian of the residuals with respect to the outputs and store
the result in `drdy`

This does *not* update any of the values stored in `component`
"""
function residual_output_jacobian!(component::ImplicitComponent, drdy, x, y)
    component.dfdy(drdy, x, y)
    return drdy
end

"""
    residual_output_jacobian!(component::ImplicitComponent, x, y)

Evaluate the jacobian of the residuals with respect to the outputs.

Return the result and store in `component.drdy`
"""
function residual_output_jacobian!(component::ImplicitComponent, x, y)
    if (x != component.x_dfdy) || (y != component.y_dfdy)
        residual_output_jacobian!!(component, x, y)
    end
    return residual_output_jacobian(component)
end

"""
    residual_output_jacobian!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the jacobian of the residuals with respect to the outputs.

Return the result and store in `component.drdy`
"""
function residual_output_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    component.dfdy(component.drdy, x, y)
    return residual_output_jacobian(component)
end

"""
    residuals_and_output_jacobian(component::ImplicitComponent)

Return the residuals and jacobian of the residuals with respect to the outputs
stored in `component`.
"""
residuals_and_output_jacobian(component::ImplicitComponent) = component.r, component.drdy

"""
    residuals_and_output_jacobian(component::ImplicitComponent, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs.

This does *not* update any of the values stored in `component`
"""
function residuals_and_output_jacobian(component::ImplicitComponent, x, y)
    TR = promote_type(eltype(component.r), eltype(x), eltype(y))
    TDRY = promote_type(eltype(component.drdy), eltype(x), eltype(y))
    r = similar(component.r, TR)
    drdy = similar(component.drdy, TDRY)
    return residuals_and_output_jacobian!(component, r, drdy, x, y)
end

"""
    residuals_and_output_jacobian!(component::ImplicitComponent, r, drdy, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs
and store the results in `r` and `drdy`

This does *not* update any of the values stored in `component`
"""
function residuals_and_output_jacobian!(component::ImplicitComponent, r, drdy, x, y)
    component.fdfdy(r, drdy, x, y)
    return r, drdy
end

"""
    residuals_and_output_jacobian!(component::ImplicitComponent, x, y)

Evaluate the residuals and jacobian of the residuals with respect to the outputs.

Return the result and store in `component.r` and `component.drdy`
"""
function residuals_and_output_jacobian!(component::ImplicitComponent, x, y)
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
    residuals_and_output_jacobian!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the residuals and jacobian of the residuals with respect to the outputs.

Return the result and store in `component.r` and `component.drdy`
"""
function residuals_and_output_jacobian!!(component::ImplicitComponent, x, y)
    copyto!(component.x_f, x)
    copyto!(component.y_f, y)
    copyto!(component.x_dfdy, x)
    copyto!(component.y_dfdy, y)
    component.fdfdy(component.r, component.drdy, x, y)
    return residuals_and_output_jacobian(component)
end

"""
    residuals_and_jacobians(component::ImplicitComponent)

Return the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs stored in `component`.
"""
residuals_and_jacobians(component::ImplicitComponent) = residuals(component),
    residual_input_jacobian(component), residual_output_jacobian(component)

"""
    residuals_and_jacobians(component::ImplicitComponent, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs.

This does *not* update any of the values stored in `component`
"""
function residuals_and_jacobians(component::ImplicitComponent, x, y)
    TR = promote_type(eltype(component.r), eltype(x), eltype(y))
    TDRX = promote_type(eltype(component.drdx), eltype(x), eltype(y))
    TDRY = promote_type(eltype(component.drdy), eltype(x), eltype(y))
    r = similar(component.r, TR)
    drdx = similar(component.drdx, TDRX)
    drdy = similar(component.drdy, TDRY)
    return residuals_and_jacobians!(component, r, drdx, drdy, x, y)
end

"""
    residuals_and_jacobians!(component::ImplicitComponent, r, drdx, drdy, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs and store the results in `r`, `drdx`, and `drdy`

This does *not* update any of the values stored in `component`
"""
function residuals_and_jacobians!(component::ImplicitComponent, r, drdx, drdy, x, y)
    component.fdf(r, drdx, drdy, x, y)
    return r, drdx, drdy
end

"""
    residuals_and_jacobians!(component::ImplicitComponent, x, y)

Evaluate the residual, its jacobian with respect to the inputs, and its jacobian
with respect to the outputs.

Return the result and store in `component.r`, `component.drdx`, and `component.drdy`.
"""
function residuals_and_jacobians!(component::ImplicitComponent, x, y)

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
    residuals_and_jacobians!!(component::ImplicitComponent, x, y)

Force (re-)evaluation of the residual, its jacobian with respect to the inputs,
and its jacobian with respect to the outputs.

Return the result and store in `component.r`, `component.drdx`, and `component.drdy`.
"""
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
