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
    subcomponent_outputs!!(component::ExplicitSystem, x)

Force re-evaluation of the inputs and outputs of an explicit system's subcomponents.

Store in the `x_f` and `y` fields of each subcomponent.
"""
function subcomponent_outputs!!(component, x)

    # unpack subcomponents
    subcomponents = component.components

    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from subcomponent output
                xsub[ix] = outputs(subcomponents[jc])[jy]
            end
        end
        # update outputs for current subcomponent
        outputs!!!(subcomponent, xsub)
    end

    return nothing
end

"""
    subcomponent_outputs_and_jacobians!(component::ExplicitSystem, xsub, ysub, dsub, x)

Evaluate the inputs, outputs, and jacobians of an explicit system's subcomponents.
Store the results in xsub, ysub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_outputs_and_jacobians!(component::ExplicitSystem, xsub, ysub, dsub, x)

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
    subcomponent_outputs_and_jacobians!(component::ExplicitSystem, x)

Evaluate the inputs, outputs, and jacobians of an explicit system's subcomponents.

Store in the `x_f`, `y`, and `dydx` fields of each subcomponent.
"""
function subcomponent_outputs_and_jacobians!(component::ExplicitSystem, x)

    # unpack subcomponents
    subcomponents = component.components

    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # flag for whether to update a subcomponent
        update_f = false
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # check if this input is different, update flag accordingly
                update_f = update_f || xsub[ix] != x[jy]
                # set value from system input
                xsub[ix] = x[jy]
            else
                # check if this input is different, update flag accordingly
                update_f = update_f || xsub[ix] != outputs(subcomponents[jc])[jy]
                # set value from subcomponent output
                xsub[ix] = outputs(subcomponents[jc])[jy]
            end
        end
        update_df = xsub != subcomponent.x_df
        # update outputs and jacobians for current subcomponent (if necessary)
        if update_f && update_df
            outputs_and_jacobian!!(subcomponent, xsub)
        elseif update_df
            jacobian!!(subcomponent, xsub)
        elseif update_f
            residuals!!(subcomponent, xsub)
        end
    end

    return nothing
end

"""
    subcomponent_outputs_and_jacobians!!(component::ExplicitSystem, x)

Force (re-)evaluation of inputs, outputs, and jacobians of an explicit system's
subcomponents.

Store in the `x_f`, `y`, and `dydx` fields of each subcomponent.
"""
function subcomponent_outputs_and_jacobians!!(component::ExplicitSystem, x)

    # unpack subcomponents
    subcomponents = component.components

    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_df
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from subcomponent output
                xsub[ix] = outputs(subcomponents[jc])[jy]
            end
        end
        # update outputs and jacobians for current subcomponent (if necessary)
        outputs_and_jacobian!!!(subcomponent, xsub)
    end

    return nothing
end

"""
    update_system_outputs!(component, x)

Update the outputs in `component` given the inputs `x`.
"""
function update_system_outputs!(component, x)
    # update system outputs
    for iy = 1:length(component.y)
        jc, jy = component.output_mapping[iy]
        if jc == 0
            component.y[iy] = x[jy]
        else
            component.y[iy] = outputs(component.components[jc])[jy]
        end
    end
    return outputs(component)
end

"""
    update_system_outputs!(component, y, x, ysub)

Update the outputs `y` given the inputs `x` and the component outputs `ysub`.
"""
function update_system_outputs!(component, y, x, ysub)
    # update system outputs
    for iy = 1:length(y)
        jc, jy = component.output_mapping[iy]
        if jc == 0
            y[iy] = x[jy]
        else
            y[iy] = ysub[jc][jy]
        end
    end
    return y
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
function forward_mode_jacobian_branch!(system, dydx, ix, cprod, jc, jx, dsub=nothing)
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
function reverse_mode_jacobian!(system::ExplicitSystem, dydx, dsub = nothing)

    # get output dimensions
    ny, nx = size(dydx)

    # initialize output to zeros
    dydx .= 0.0

    # loop through each output
    for iy = 1:ny
        # start new chain rule product
        cprod = 1
        # get the source of this system output
        jc, jy = system.output_mapping[iy]
        # check the identity of the source
        if iszero(jc)
            # source is the system input
            ix = jy
            # add chain rule product to corresponding entry in jacobian
            dydx[iy, ix] += cprod
        else
            # source is a subcomponent, add its contribution to the jacobian
            reverse_mode_jacobian_branch!(system, dydx, iy, cprod, jc, jy, dsub)
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
            dydx[iy, ix] += new_cprod
        elseif iszero(new_cprod)
            # stop computing branch because there is nothing to add
        else
            # move on to next subcomponent in chain
            reverse_mode_jacobian_branch!(system, dydx, iy, new_cprod, new_jc, new_jy, dsub)
        end
    end
end
