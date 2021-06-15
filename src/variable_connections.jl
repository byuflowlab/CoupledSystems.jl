"""
    system_component_mapping(argin, components)

Construct the mapping to subcomponent inputs from system inputs and/or subcomponent
outputs.

# Arguments
 - `argin`: Tuple of named variables (see [`NamedVar`](@ref)) corresponding to
    system inputs.
 - `components`: System subcomponents
"""
function system_component_mapping(argin, components)

    # names of subcomponent inputs/outputs
    input_names = [get_names(comp.argin) for comp in components]
    output_names = [get_names(comp.argout) for comp in components]

    # construct component input mapping vector
    component_input_mapping = Vector{Vector{NTuple{2,Int}}}(undef, length(components))
    for icomp = 1:length(components)
        # initialize mapping for this component
        current_mapping = Vector{NTuple{2,Int}}(undef, length(inputs(components[icomp])))
        # input variable starting index
        idx_in = 0
        # populate mapping for this component
        for i = 1:length(input_names[icomp])
            # number of values in this variable
            nval = vector_length(components[icomp].argin[i])
            # first check system inputs for match
            idx_sys = 0
            for j = 1:length(argin)
                if input_names[icomp][i] === get_name(argin[j])
                    # add to mapping
                    for k = 1:nval
                        current_mapping[idx_in+k] = (0, idx_sys+k)
                    end
                    # skip to next input variable
                    @goto next_input_variable
                else
                    # check next system input
                    idx_sys += vector_length(argin[j])
                end
            end
            # then check all components for match
            for jcomp = 1:length(components)
                # only check components other than self
                if icomp == jcomp
                    continue
                end
                # output variable starting index
                idx_out = 0
                # loop through all output variables
                for j = 1:length(output_names[jcomp])
                    # check if names match
                    if input_names[icomp][i] === output_names[jcomp][j]
                        # add to mapping
                        for k = 1:nval
                            current_mapping[idx_in+k] = (jcomp, idx_out+k)
                        end
                        # skip to next input variable
                        @goto next_input_variable
                    else
                        # check next output variable
                        idx_out += vector_length(components[jcomp].argout[j])
                    end
                end
            end
            error("No system input or component found for input `$(input_names[icomp][i])`")
            @label next_input_variable
            idx_in += nval
        end
        component_input_mapping[icomp] = current_mapping
    end

    return component_input_mapping
end

"""
    system_output_mapping(components, argin, argout)

Constructs the output mapping for a system based on input and output argument
names of the system and components.
"""
function system_output_mapping(components, argin, argout)

    # system output vector
    ysys = combine(argout)

    # names of input and output variables
    input_names = [get_names(comp.argin) for comp in components]
    output_names = [get_names(comp.argout) for comp in components]

    # construct system output mapping vector
    output_mapping = Vector{NTuple{2,Int}}(undef, length(ysys))
    # output variable starting index
    idx_sys = 0
    # populate mapping for this component
    for i = 1:length(argout)
        # number of values in variable
        nval = vector_length(argout[i])
        # first check system inputs for match
        idx_in = 0
        for j = 1:length(argin)
            if get_name(argout[i]) === get_name(argin[j])
                # add to mapping
                for k = 1:nval
                    output_mapping[idx_sys+k] = (0, idx_in+k)
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_in += vector_length(argin[j])
            end
        end
        # then check all components for match
        for jcomp = 1:length(components)
            # output variable starting index
            idx_out = 0
            # loop through all output variables
            for j = 1:length(output_names[jcomp])
                # check if names match
                if get_name(argout[i]) === output_names[jcomp][j]
                    # add to mapping
                    for k = 1:nval
                        output_mapping[idx_sys+k] = (jcomp, idx_out+k)
                    end
                    # skip to next input variable
                    @goto next_output_variable
                else
                    # check next output variable
                    idx_out += vector_length(components[jcomp].argout[j])
                end
            end
        end
        error("No system input or component output found for output `$(get_name(argout[i]))`")
        @label next_output_variable
        idx_sys += nval
    end

    return output_mapping
end


"""
    mapping_matrix(argin, argout)

Construct a matrix that maps the arguments in `argin` to the arguments in `argout`
"""
function mapping_matrix(argin::Tuple, argout::Tuple)

    # length of input and output vectors
    nx = vector_length(argin)
    ny = vector_length(argout)

    # initialize matrix
    dfdx = spzeros(ny, nx)

    # output variable starting index
    idx_out = 0
    # populate mapping for this component
    for i = 1:length(argout)
        # number of values in variable
        nval = vector_length(argout[i])
        # check all inputs for match
        idx_in = 0
        for j = 1:length(argin)
            if get_name(argout[i]) === get_name(argin[j])
                # add to mapping
                for k = 1:nval
                    dfdx[idx_out + k, idx_in + k] = 1
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_in += vector_length(argin[j])
            end
        end
        @label next_output_variable
        idx_out += nval
    end

    return dfdx
end

"""
    mapping_matrices(argin, argstate, argout)

Construct matrices that maps an implicit function's inputs and/or state variables
to specified output variables.
"""
function mapping_matrices(argin::Tuple, argstate::Tuple, argout::Tuple)

    # length of input, state, and output vectors
    nx = vector_length(argin)
    nu = vector_length(argstate)
    ny = vector_length(argout)

    # initialize matrices
    dfdx = BitArray(undef, ny, nx) .= 0
    dfdu = BitArray(undef, ny, nu) .= 0

    # output variable starting index
    idx_out = 0
    # populate mapping for this component
    for i = 1:length(argout)
        # number of values in variable
        nval = vector_length(argout[i])
        # first check system inputs for match
        idx_in = 0
        for j = 1:length(argin)
            if get_name(argout[i]) === get_name(argin[j])
                # add to mapping
                for k = 1:nval
                    dfdx[idx_out + k, idx_in + k] = 1
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_in += vector_length(argin[j])
            end
        end
        # then check system states for match
        idx_state = 0
        for j = 1:length(argstate)
            if get_name(argout[i]) === get_name(argstate[j])
                # add to mapping
                for k = 1:nval
                    dfdu[idx_out + k, idx_state + k] = 1
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_state += vector_length(argstate[j])
            end
        end
        error("No input or state variable found for output `$(get_name(argout[i]))`")
        @label next_output_variable
        idx_out += nval
    end

    return dfdx, dfdu
end
