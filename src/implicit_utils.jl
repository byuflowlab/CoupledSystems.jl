"""
    subcomponent_residuals!(component::ImplicitSystem, xsub, ysub, rsub, x, y)

Evaluate the residuals of an implicit system's subcomponents and store
the results in `xsub`, `ysub`, and `rsub`.

This does *not* update any of the values stored in `component`
"""
function subcomponent_residuals!(component::ImplicitSystem, xsub, ysub, rsub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        residuals!(subcomponent, rsub[ic], xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_residuals!(component::ImplicitSystem, x, y)

Evaluate the residuals of an implicit system's subcomponents.

Store in the `x_f`, `y_f`, and `r` fields of each subcomponent.
"""
function subcomponent_residuals!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update = ysub != subcomponent.y_f
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != x[jy]
                # set value from system input
                xsub[ix] = x[jy]
            else
                # check if this input is different, update flag accordingly
                update = update || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        # update residuals for current subcomponent (if necessary)
        if update
            residuals!!(subcomponent, xsub, ysub)
        end
    end
    return nothing
end

"""
    subcomponent_residuals!!(component::ImplicitSystem, x, y)

Force re-evaluation of the residuals of an explicit system's subcomponents.

Store in the `x_f`, `y_f`, and `r` fields of each subcomponent.
"""
function subcomponent_residuals!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        # update residuals for current subcomponent
        residuals!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    subcomponent_input_jacobians!(component::ImplicitSystem, xsub, ysub, dsub, x, y)

Evaluate the input jacobians of an implicit system's subcomponents. Store the
results in xsub, ysub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_input_jacobians!(component::ImplicitSystem, xsub, ysub,
    dsub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        # update input jacobians for current subcomponent
        residual_input_jacobian!(subcomponent, dsub[ic], xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_input_jacobians!(component::ImplicitSystem, x, y)

Evaluate the input jacobians of an implicit system's subcomponents.

Store in the `x_dfdx`, `y_dfdx`, and `drdx` fields of each subcomponent.
"""
function subcomponent_input_jacobians!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdx
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update = ysub != subcomponent.y_dfdx
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
                update = update || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        # update input jacobians for current subcomponent (if necessary)
        if update
            residual_input_jacobian!!(subcomponent, xsub, ysub)
        end
    end
    return nothing
end

"""
    subcomponent_input_jacobians!!(component::ImplicitSystem, x, y)

Force (re-)evaluation the input jacobians of an implicit system's subcomponents.

Store in the `x_dfdx`, `y_dfdx`, and `drdx` fields of each subcomponent.
"""
function subcomponent_input_jacobians!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdx
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        residual_input_jacobian!!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    subcomponent_output_jacobians!(component::ImplicitSystem, xsub, ysub, dsub, x, y)

Evaluate the output jacobians of an implicit system's subcomponents. Store the
results in xsub, ysub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_output_jacobians!(component::ImplicitSystem, xsub, ysub,
    dsub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        # update input jacobians for current subcomponent
        residual_output_jacobian!(subcomponent, dsub[ic], xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_output_jacobians!(component::ImplicitSystem, x, y)

Evaluate the output jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, and `drdy` fields of each subcomponent.
"""
function subcomponent_output_jacobians!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdy
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update = ysub != subcomponent.y_dfdy
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
                update = update || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        # update input jacobians for current subcomponent (if necessary)
        if update
            residual_output_jacobian!!(subcomponent, xsub, ysub)
        end
    end
    return nothing
end

"""
    subcomponent_output_jacobians!!(component::ImplicitSystem, x, y)

Force (re-)evaluation the output jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, and `drdy` fields of each subcomponent.
"""
function subcomponent_output_jacobians!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdy
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        residual_output_jacobian!!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    subcomponent_residuals_and_input_jacobians!(component::ImplicitSystem, xsub,
        ysub, rsub, dsub, x, y)

Evaluate the residuals and input jacobians of an implicit system's subcomponents.
Store the results in xsub, ysub, rsub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_residuals_and_input_jacobians!(component::ImplicitSystem,
    xsub, ysub, rsub, dsub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        # update input jacobians for current subcomponent
        residuals_and_input_jacobian!(subcomponent, rsub[ic], dsub[ic], xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_residuals_and_input_jacobians!(component::ImplicitSystem, x, y)

Evaluate the residuals and input jacobians of an implicit system's subcomponents.

Store in the `x_dfdx`, `y_dfdx`, `r`, and `drdx` fields of each subcomponent.
"""
function subcomponent_residuals_and_input_jacobians!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update_f = ysub != subcomponent.y_f
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if iszero(jc)
                # check if this input is different, update flag accordingly
                update_f = update_f || xsub[ix] != x[jy]
                # set value from system input
                xsub[ix] = x[jy]
            else
                # check if this input is different, update flag accordingly
                update_f = update_f || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        update_dfdx = (xsub != subcomponent.x_dfdx) || (ysub != subcomponent.y_dfdx)
        # update residuals and input jacobians for current subcomponent (if necessary)
        if update_f && update_dfdx
            residuals_and_input_jacobian!!(subcomponent, xsub, ysub)
        elseif update_f
            residuals!!(subcomponent, xsub, ysub)
        elseif update_dfdx
            residual_input_jacobian!!(subcomponent, xsub, ysub)
        end
    end
    return nothing
end

"""
    subcomponent_residuals_and_input_jacobians!!(component::ImplicitSystem, x, y)

Force (re-)evaluation the residuals and input jacobians of an implicit system's subcomponents.

Store in the `x_dfdx`, `y_dfdx`, `r`, and `drdx` fields of each subcomponent.
"""
function subcomponent_residuals_and_input_jacobians!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdx
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        residuals_and_input_jacobian!!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    subcomponent_residuals_and_output_jacobians!(component::ImplicitSystem, xsub,
        ysub, rsub, dsub, x, y)

Evaluate the residuals and output jacobians of an implicit system's subcomponents.
Store the results in xsub, ysub, rsub, and dsub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_residuals_and_output_jacobians!(component::ImplicitSystem,
    xsub, ysub, rsub, dsub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        # update output jacobians for current subcomponent
        residuals_and_output_jacobian!(subcomponent, rsub[ic], dsub[ic], xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_residuals_and_output_jacobians!(component::ImplicitSystem, x, y)

Evaluate the residuals and output jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, `r`, and `drdy` fields of each subcomponent.
"""
function subcomponent_residuals_and_output_jacobians!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update_f = ysub != subcomponent.y_f
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
                update_f = update_f || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        update_dfdy = (xsub != subcomponent.x_dfdy) || (ysub != subcomponent.y_dfdy)
        # update residuals and output jacobians for current subcomponent (if necessary)
        if update_f && update_dfdy
            residuals_and_output_jacobian!!(subcomponent, xsub, ysub)
        elseif update_f
            residuals!!(subcomponent, xsub, ysub)
        elseif update_dfdy
            residual_output_jacobian!!(subcomponent, xsub, ysub)
        end
    end
    return nothing
end

"""
    subcomponent_residuals_and_output_jacobians!!(component::ImplicitSystem, x, y)

Force (re-)evaluation of the residuals and output jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, `r`, and `drdy` fields of each subcomponent.
"""
function subcomponent_residuals_and_output_jacobians!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdy
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        residuals_and_output_jacobian!!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    subcomponent_residuals_and_jacobians!(component::ImplicitSystem, xsub,
        ysub, rsub, dsub, x, y)

Evaluate the residuals and jacobians of an implicit system's subcomponents.
Store the results in xsub, ysub, rsub, dxsub, and dysub.

This does *not* update any of the values stored in `component`
"""
function subcomponent_residuals_and_jacobians!(component::ImplicitSystem,
    xsub, ysub, rsub, dxsub, dysub, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent outputs
        ysub[ic] = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub[ic])
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ic][ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ic][ix] = y[idx[jc]+jy]
            end
        end
        # update output jacobians for current subcomponent
        residuals_and_jacobians!(subcomponent, rsub[ic], dxsub[ic], dysub[ic],
            xsub[ic], ysub[ic])
    end
    return nothing
end

"""
    subcomponent_residuals_and_jacobians!(component::ImplicitSystem, x, y)

Evaluate the residuals and jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, `r`, `drdx`, and `drdy` fields of each subcomponent.
"""
function subcomponent_residuals_and_jacobians!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_f
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # flag for whether to update a subcomponent
        update_f = ysub != subcomponent.y_f
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
                update_f = update_f || xsub[ix] != y[idx[jc]+jy]
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        update_dfdx = (xsub != subcomponent.x_dfdx) || (ysub != subcomponent.y_dfdx)
        update_dfdy = (xsub != subcomponent.x_dfdy) || (ysub != subcomponent.y_dfdy)
        # calculate everything?
        if update_f && update_dfdx && update_dfdy
            residuals_and_jacobians!!(subcomponent, xsub, ysub)
        else
            # calculate residuals?
            if update_f
                residuals!!(subcomponent, xsub, ysub)
            end
            # calculate input jacobian?
            if update_dfdx
                residual_input_jacobian!!(subcomponent, xsub, ysub)
            end
            # calculate output jacobian?
            if update_dfdy
                residual_output_jacobian!!(subcomponent, xsub, ysub)
            end
        end
    end
    return nothing
end

"""
    subcomponent_residuals_and_jacobians!!(component::ImplicitSystem, x, y)

Force (re-)evaluation of the residuals and output jacobians of an implicit system's subcomponents.

Store in the `x_dfdy`, `y_dfdy`, `r`, and `drdy` fields of each subcomponent.
"""
function subcomponent_residuals_and_jacobians!!(component::ImplicitSystem, x, y)
    # unpack subcomponents
    subcomponents = component.components
    # unpack index for accessing outputs/residuals for each component
    idx = component.idx
    # update subcomponents sequentially
    for ic = 1:length(subcomponents)
        # extract current subcomponent
        subcomponent = subcomponents[ic]
        # subcomponent inputs
        xsub = subcomponent.x_dfdy
        # subcomponent outputs
        ysub = view(y, idx[ic]+1 : idx[ic+1])
        # map inputs or subcomponent outputs to current subcomponent inputs
        for ix = 1:length(xsub)
            jc, jy = component.component_input_mapping[ic][ix]
            if jc == 0
                # set value from system input
                xsub[ix] = x[jy]
            else
                # set value from provided subcomponent outputs
                xsub[ix] = y[idx[jc]+jy]
            end
        end
        residuals_and_jacobians!!!(subcomponent, xsub, ysub)
    end
    return nothing
end

"""
    update_system_residuals!(component)

Update the residuals in `component` to correspond to the subcomponent residuals.
"""
function update_system_residuals!(component)
    subcomponents = component.components
    idx = component.idx
    for ic = 1:length(subcomponents)
        subcomponent = subcomponents[ic]
        component.r[idx[ic]+1:idx[ic+1]] = residuals(subcomponent)
    end
    return residuals(component)
end

"""
    update_system_residuals!(component, r, rsub)

Update the residuals `r` to make them correspond to the subcomponent residuals
in `rsub`.
"""
function update_system_residuals!(component, r, rsub)
    subcomponents = component.components
    idx = component.idx
    for ic = 1:length(subcomponents)
        r[idx[ic]+1:idx[ic+1]] = rsub[ic]
    end
    return r
end

"""
    update_system_input_jacobian!(system::ImplicitSystem)

Update the input jacobian matrix in `component` to correspond to the subcomponent
residuals.
"""
function update_system_input_jacobian!(system::ImplicitSystem)
    system.drdx .= 0.0
    subcomponents = system.components
    component_input_mapping = system.component_input_mapping
    idx = system.idx
    for ic = 1:length(subcomponents)
        subcomponent = subcomponents[ic]
        dsub = residual_input_jacobian(subcomponent)
        nr, nx = size(dsub)
        for ix = 1:nx
            jc, jx = component_input_mapping[ic][ix]
            if iszero(jc)
                for ir = 1:nr
                    jr = idx[ic] + ir
                    system.drdx[jr,jx] = dsub[ir,ix]
                end
            end
        end
    end
    return residual_input_jacobian(system)
end

"""
    update_system_input_jacobian!(system::ImplicitSystem, drdx, dsub)

Update the input jacobian `drdx` to correspond to the subcomponent input jacobians
in dsub.
"""
function update_system_input_jacobian!(system::ImplicitSystem, drdx, dsub)
    drdx .= 0.0
    component_input_mapping = system.component_input_mapping
    idx = system.idx
    for ic = 1:length(dsub)
        nr, nx = size(dsub[ic])
        for ix = 1:nx
            jc, jx = component_input_mapping[ic][ix]
            if iszero(jc)
                for ir = 1:nr
                    jr = idx[ic] + ir
                    drdx[jr,jx] = dsub[ic][ir,ix]
                end
            end
        end
    end
    return drdx
end

"""
    update_system_output_jacobian!(system::ImplicitSystem, drdy, dsub)

Update the output jacobian `drdy` to correspond to the subcomponent output jacobians
in dsub.
"""
function update_system_output_jacobian!(system::ImplicitSystem, drdy, dxsub, dysub)
    drdy .= 0.0
    subcomponents = system.components
    component_input_mapping = system.component_input_mapping
    idx = system.idx
    # diagonal blocks
    for ic = 1:length(subcomponents)
        ir = iy = idx[ic]+1 : idx[ic+1]
        drdy[ir, iy] = dysub[ic]
    end
    # off diagonal blocks
    for ic = 1:length(subcomponents)
        nr, nx = size(dxsub[ic])
        for ix = 1:nx
            jc, jy = component_input_mapping[ic][ix]
            if !iszero(jc)
                for ir = 1:nr
                    kr = idx[ic] + ir
                    ky = idx[jc] + jy
                    drdy[kr, ky] = dxsub[ic][ir,ix]
                end
            end
        end
    end
    return drdy
end

"""
    update_system_output_jacobian!(system::ImplicitSystem)

Update the output jacobian in `system` to correspond to the subcomponent input
and output jacobians.
"""
function update_system_output_jacobian!(system::ImplicitSystem)
    system.drdy .= 0.0
    subcomponents = system.components
    component_input_mapping = system.component_input_mapping
    idx = system.idx
    # diagonal blocks
    for ic = 1:length(subcomponents)
        ir = iy = idx[ic]+1 : idx[ic+1]
        dysub = residual_output_jacobian(subcomponents[ic])
        system.drdy[ir, iy] = dysub
    end
    # off diagonal blocks
    for ic = 1:length(subcomponents)
        dxsub = residual_input_jacobian(subcomponents[ic])
        nr, nx = size(dxsub)
        for ix = 1:nx
            jc, jy = component_input_mapping[ic][ix]
            if !iszero(jc)
                for ir = 1:nr
                    kr = idx[ic] + ir
                    ky = idx[jc] + jy
                    system.drdy[kr, ky] = dxsub[ir,ix]
                end
            end
        end
    end
    return residual_output_jacobian(system)
end
