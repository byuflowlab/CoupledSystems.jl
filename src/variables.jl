"""
    NamedVar{T}

Defines a named variable

# Field
 - `name::Symbol`: Variable name
 - `value::T`: Default variable value
"""
struct NamedVar{T}
    name::Symbol
    value::T
end

"""
    @var var = val

Macro which modifies the expression `var = val` to `var = NamedVar(:var, val)`
```julia-repl
julia> @var x = 5
  NamedVar{Int64}(:x, 5)

julia> x
  NamedVar{Int64}(:x, 5)
```
"""
macro var(ex)
    @assert (ex.head === :(=)) "Incorrect usage of @var macro"
    var = ex.args[1]
    name = string(var)
    val = ex.args[2]
    return esc(:($var = NamedVar(Symbol($name), $(val))))
end

"""
    name(var::NamedVar)

Return the name of a named variable.
"""
name(var::NamedVar) = var.name

"""
    value(var::NamedVar)

Return the value of a named variable.
"""
value(var::NamedVar) = var.value

"""
    value(val)

Return the value of a variable.
"""
value(val) = val

"""
    length(var::NamedVar)

Return the length of a named variable
"""
Base.length(var::NamedVar) = length(var.value)

"""
    size(var)

Return the size of a named variable
"""
Base.size(var::NamedVar) = size(var.value)

"""
    axes(var::NamedVar)

Return the axes of a named variable
"""
Base.axes(var::NamedVar) = axes(var.value)

"""
    combine(vars::Tuple)

Combine the values of all variables in `vars` into a single vector
"""
combine(vars) = vcat(vectorize.(value.(vars))...)

"""
    combine!(y, vars::Tuple)

In-place version of [`combine`](@ref)
"""
@inline function combine!(y, vars::Tuple, idx=0)
    var = first(vars)
    nval = length(var)
    y[idx+1:idx+nval] .= var
    idx += nval
    return combine!(y, tail(vars), idx)
end
@inline combine!(y, vars::Tuple{}, idx=0) = y

"""
    separate(vars::Tuple, values)

Return the values stored in `values` as separated variables rather than a single
output vector.  `vars` is a tuple of named variables corresponding to each
returned parameter.
"""
@inline function separate(vars::Tuple, values)
    # extract first variable
    n1 = length(first(vars))
    v1 = same_axes(value(first(vars)), view(values, 1:n1))
    # call recursively to extract remaining variables
    vars = tail(vars)
    values = view(values, n1+1:length(values))
    return (v1, separate(vars, values)...)
end
@inline separate(vars::Tuple{}, values) = ()

"""
    vectorize(x)

Converts `x` to a vector.
"""
vectorize
@inline vectorize(x::Number) = [x]
@inline vectorize(x::AbstractArray) = x[:]

"""
    same_axes(v1, v2)

Return `v2` with axes correponding to `v1`.  If `v1` is a scalar, return a scalar.
"""
same_axes
@inline same_axes(v1::AbstractArray, v2) = reshape(v2, axes(v1))
@inline same_axes(v1::Number, v2) = reshape(v2, axes(v1))[]

"""
    getindices(collection, indices::Tuple)

Custom version of `getindex` that allow collections to be accessed using a tuple
of indices
"""
@inline getindices(collection, indices::Tuple) = (collection[first(indices)],
    getindices(collection, tail(indices))...)
@inline getindices(collection, indices::Tuple{}) = ()

"""
    setindices(x, v::Tuple, indices::Tuple)

Creates a new collection similar to `x` with the values at the indices in `indices` set
to the values in `v`.
"""
@inline setindices(x, v::Tuple, indices::Tuple) = setindices(setindex(x, first(v), first(indices)), tail(v), tail(indices))
@inline setindices(x, v::Tuple{}, indices::Tuple{}) = x

"""
    system_component_mapping(components, argin)

Constructs the component input-output mapping for a system based on input and output
argument names.
"""
function system_component_mapping(components, argin)

    # names of input and output variables
    input_names = [name.(comp.argin) for comp in components]
    output_names = [name.(comp.argout) for comp in components]

    # construct component input mapping vector
    component_input_mapping = Vector{Vector{NTuple{2,Int}}}(undef, length(components))
    for icomp = 1:length(components)
        # initialize mapping for this component
        current_mapping = Vector{NTuple{2,Int}}(undef, length(inputs(components[icomp])))
        # input variable starting index
        idx_in = 0
        # populate mapping for this component
        for i = 1:length(input_names[icomp])
            # number of values in variable
            nval = length(components[icomp].argin[i])
            # first check system inputs for match
            idx_sys = 0
            for j = 1:length(argin)
                if input_names[icomp][i] === name(argin[j])
                    # add to mapping
                    for k = 1:nval
                        current_mapping[idx_in+k] = (0, idx_sys+k)
                    end
                    # skip to next input variable
                    @goto next_input_variable
                else
                    # check next system input
                    idx_sys += length(argin[j])
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
                        idx_out += length(components[jcomp].argout[j])
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
    input_names = [name.(comp.argin) for comp in components]
    output_names = [name.(comp.argout) for comp in components]

    # construct system output mapping vector
    output_mapping = Vector{NTuple{2,Int}}(undef, length(ysys))
    # output variable starting index
    idx_sys = 0
    # populate mapping for this component
    for i = 1:length(argout)
        # number of values in variable
        nval = length(argout[i])
        # first check system inputs for match
        idx_in = 0
        for j = 1:length(argin)
            if name(argout[i]) === name(argin[j])
                # add to mapping
                for k = 1:nval
                    output_mapping[idx_sys+k] = (0, idx_in+k)
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_in += length(argin[j])
            end
        end
        # then check all components for match
        for jcomp = 1:length(components)
            # output variable starting index
            idx_out = 0
            # loop through all output variables
            for j = 1:length(output_names[jcomp])
                # check if names match
                if name(argout[i]) === output_names[jcomp][j]
                    # add to mapping
                    for k = 1:nval
                        output_mapping[idx_sys+k] = (jcomp, idx_out+k)
                    end
                    # skip to next input variable
                    @goto next_output_variable
                else
                    # check next output variable
                    idx_out += length(components[jcomp].argout[j])
                end
            end
        end
        error("No system input or component output found for output `$(name(argout[i]))`")
        @label next_output_variable
        idx_sys += nval
    end

    return output_mapping
end

"""
    output_mapping_matrices(argin, argstate, argout)

Construct matrices that maps an implicit function's inputs and/or state variables
to specified output variables.
"""
function output_mapping_matrices(argin, argstate, argout)

    # length of input, state, and output vectors
    nx = sum(length.(argin))
    nu = sum(length.(argstate))
    ny = sum(length.(argout))

    # initialize matrices
    dfdx = BitArray(undef, ny, nx)
    dfdu = BitArray(undef, ny, nu)

    # output variable starting index
    idx_out = 0
    # populate mapping for this component
    for i = 1:length(argout)
        # number of values in variable
        nval = length(argout[i])
        # first check system inputs for match
        idx_in = 0
        for j = 1:length(argin)
            if name(argout[i]) === name(argin[j])
                # add to mapping
                for k = 1:nval
                    dfdx[idx_out + k, idx_in + k] = 1
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_in += length(argin[j])
            end
        end
        # then check system states for match
        idx_state = 0
        for j = 1:length(argstate)
            if name(argout[i]) === name(argstate[j])
                # add to mapping
                for k = 1:nval
                    dfdu[idx_out + k, idx_state + k] = 1
                end
                # skip to next input variable
                @goto next_output_variable
            else
                # check next system input
                idx_state += length(argin[j])
            end
        end
        error("No input or state variable found for output `$(name(argout[i]))`")
        @label next_output_variable
        idx_out += nval
    end

    return dfdx, dfdu
end
