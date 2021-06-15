"""
    NamedVar{T}

Defines a named variable

# Field
 - `name::Symbol`: Variable name
 - `value::T`: Variable value
"""
struct NamedVar{T}
    name::Symbol
    value::T
end

"""
    @named var = val

Macro which modifies the expression `var = val` to `var = NamedVar(:var, val)`
```julia-repl
julia> @named x = 5
  NamedVar{Int64}(:x, 5)

julia> x
  NamedVar{Int64}(:x, 5)
```
"""
macro named(ex)
    @assert (ex.head === :(=)) "Incorrect usage of @named macro"
    var = ex.args[1]
    name = string(var)
    val = ex.args[2]
    return esc(:($var = NamedVar(Symbol($name), $(val))))
end

"""
    get_name(var)

Return the name of a variable.  Return `nothing` if no name is assigned.
"""
get_name(var::NamedVar) = var.name
get_name(var) = nothing

"""
    get_names(vars::Tuple)

Return a tuple which contains the names of all named variables in `vars`
"""
@inline function get_names(vars::Tuple)
    # check identity of first named variable
    if typeof(first(vars)) <: NamedVar
        # if named variable store name in a tuple
        name = (get_name(first(vars)),)
    else
        # if not a named variable represent name with an empty tuple
        name = ()
    end
    # call recursively to extract names of remaining variables
    return (name..., get_names(tail(vars))...)
end
@inline get_names(vars::Tuple{}) = ()

"""
    get_value(var)

Return the value of a variable.
"""
get_value(var::NamedVar) = var.value
get_value(val) = val

"""
    get_values(vars::Tuple)

Return a tuple which contains the values of all variables in `vars`
"""
get_values(vars::Tuple) = (get_value(first(vars)), get_values(tail(vars))...)
get_values(vars::Tuple{}) = ()

"""
    vector_length(var)

Return the length of a variable when expressed as a vector
"""
vector_length(var::NamedVar) = vector_length(get_value(var))
vector_length(x) = add_vector_length(x, 0)

@inline add_vector_length(x::Number, nx) = nx + 1
@inline add_vector_length(x::NTuple{N,T}, nx) where {N,T<:Number} = nx + length(x)
@inline add_vector_length(x::AbstractArray{T,N}, nx) where {T<:Number,N} = nx + length(x)
@inline function add_vector_length(v, x::AbstractArray, nx)
    for xi in x
        nx = add_vector_length(xi, nx)
    end
    return nx
end
@inline function add_vector_length(x::Tuple, nx)
    nx = add_vector_length(first(x), nx)
    return add_vector_length(tail(x), nx)
end
@inline add_vector_length(x::Tuple{}, nx) = nx
@inline add_vector_length(var::NamedVar, nx) = add_vector_length(get_value(var), nx)


"""
    combine(x)

Combine the data in `x` into a vector.
"""
combine(x) = [x]
function combine(x::Union{<:Tuple, <:AbstractArray})
    x = collect(Iterators.flatten(x))
    while any(x->typeof(x) <: Union{<:Tuple, <:AbstractArray}, x)
        x = collect(Iterators.flatten(x))
    end
    return x
end
combine(x::NTuple{N, NamedVar}) where N = combine(get_values(x))
combine(x::Tuple{}) = Float64[]

"""
    combine!(v, x)

Combine the data in `x` into the vector `v`.
"""
@inline function combine!(v, x)
    vo, v = offset_combine!(v, x, firstindex(v))
    return v
end
@inline combine!(v, x::NTuple{N, NamedVar}) where N = combine!(v, get_values(x))
@inline combine!(v, x::Tuple{}) = v

"""
    offset_combine!(v, x, vo)

Copies the data in `x` into the vector `v` starting at offset `vo`.  Return the
new offset `vo` and the modified vector `v`.
"""
offset_combine!

@inline function offset_combine!(v, x::Number, vo)
    nx = length(x)
    copyto!(v, vo, x, 1, nx)
    vo += nx
    return vo, v
end
@inline function offset_combine!(v, x::NTuple{N,T}, vo) where {N, T<:Number}
    nx = length(x)
    copyto!(v, vo, x, 1, nx)
    vo += nx
    return vo, v
end
@inline function offset_combine!(v, x::AbstractArray{T,N}, vo) where {T<:Number, N}
    nx = length(x)
    copyto!(v, vo, x, 1, nx)
    vo += nx
    return vo, v
end
@inline function offset_combine!(v, x::AbstractArray, vo)
    for xi in x
        vo, v = offset_combine!(v, xi, vo)
    end
    return vo, v
end
@inline function offset_combine!(v, x::Tuple, vo)
    vo, v = offset_combine!(v, first(x), vo)
    return offset_combine!(v, tail(x), vo)
end
@inline offset_combine!(v, x::Tuple{}, vo) = vo, v

"""
    separate(x, v)

Return the data in the vector `v` with the shape of `x`.
"""
separate(x, v) = separate!(deepcopy(x), v)
separate(x::NTuple{N, T}, v) where {N,T<:NamedVar} = separate(get_values(x), v)
separate(x::Tuple{}, v) = v

"""
    separate!(x, v)

Return the data in the vector `v` with the shape of `x`.  Modifies `x` in order
to avoid allocations, if possible.
"""
@inline function separate!(x, v)
    vo, x = offset_separate!(x, v, firstindex(v))
    return x
end
@inline separate!(x::NTuple{N, NamedVar}, v) where N = separate!(get_values(x), v)
separate!(x::Tuple{}, v) = v

"""
    offset_separate!(x, v, vo)

Return the data in the vector `v` with the shape of `x` starting at offset `vo`.
Modify `x` in order to avoid allocations, if possible.
"""
offset_separate!

@inline function offset_separate!(x::Number, v, vo)
    x = v[vo]
    vo += 1
    return vo, x
end
@inline function offset_separate!(x::NTuple{N,T}, v, vo) where {N, T<:Number}
    x = NTuple{N,T}(view(v, vo : vo + N - 1))
    vo += N
    return vo, x
end
@inline function offset_separate!(x::AbstractArray{T,N}, v, vo) where {T<:Number, N}
    nx = length(x)
    x = reshape(view(v, vo : vo + nx - 1), size(x))
    vo += nx
    return vo, x
end
function offset_separate!(x::AbstractArray, v, vo)
    if ismutable(x)
        # modify in place (if possible)
        for i = 1:length(x)
            # extract new element type
            vo, xi = offset_separate!(xi, v, vo)
            # update element type of x if necessary
            TE = promote_type(typeof(xi), eltype(x))
            if !(TE <: eltype(x))
                x = convert.(TE, x)
            end
            # store result
            x[i] = xi
        end
    else
        # convert inputs to tuple
        xt = Tuple(x)
        # get outputs as a tuple
        vo, xt = offset_separate!(xt, v, vo)
        # update element type of x if necessary
        TE = promote_type(eltype(x), typeof.(xt)...)
        if !(TE <: eltype(x))
            x = convert.(TE, x)
        end
        # convert outputs to correct type
        x = typeof(x)(xt)
    end
    return vo, x
end
function offset_separate!(x::Tuple, v, vo)
    vo, xi = offset_separate!(first(x), v, vo)
    vo, x = offset_separate!(tail(x), v, vo)
    return vo, (xi, x...)
end
offset_separate!(x::Tuple{}, v, vo) = vo, ()
