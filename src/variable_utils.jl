



"""
    make_array(x)

Converts `x` to an array.
"""
@inline make_array(x::Number) = [x]
@inline make_array(x::AbstractArray) = x

"""
    same_axes(v1, v2)

Return `v2` with axes correponding to `v1`.  If `v1` is a scalar, return a scalar.
"""
same_axes
@inline same_axes(v1::Number, v2) = reshape(v2, axes(v1))[]
@inline same_axes(v1::AbstractArray, v2) = reshape(v2, axes(v1))

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
