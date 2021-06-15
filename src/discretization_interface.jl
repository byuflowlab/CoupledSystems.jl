abstract type AbstractDiscretization end

"""
    get_segment_ends(disc::AbstractDiscretization)

Return the segment ends of a given discretization
"""
get_segment_ends

"""
    get_segment_discretization_nodes(disc::AbstractDiscretization)

Return the segment discretization node locations for a given discretization
"""
get_segment_discretization_nodes

"""
    get_segment_collocation_nodes(disc::AbstractDiscretization)

Return the segment collocation node locations for a given discretization
"""
get_segment_collocation_nodes

"""
    get_segment_nodes(disc::AbstractDiscretization)

Return the segment discretization and collocation node locations for a given discretization
"""
get_segment_nodes

"""
    get_discretization_nodes(disc::AbstractDiscretization)

Return the nodes for a given discretization
"""
function get_discretization_nodes(disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_discretization_nodes(disc)
    compact = iscompact(disc)
    return combine_segments(segment_ends, segment_nodes, compact)
end

"""
    get_discretization_nodes(disc::AbstractDiscretization)

In-place version of [`get_discretization_nodes`](@ref)
"""
function get_discretization_nodes!(t, disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_discretization_nodes(disc)
    compact = iscompact(disc)
    return combine_segments!(t, segment_ends, segment_nodes, compact)
end

"""
    get_collocation_nodes(disc::AbstractDiscretization)

Return the collocation nodes for a given discretization
"""
function get_collocation_nodes(disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_collocation_nodes(disc)
    compact = iscompact(disc)
    return combine_segments(segment_ends, segment_nodes, compact)
end

"""
    get_collocation_nodes(disc::AbstractDiscretization)

In-place version of [`get_collocation_nodes`](@ref)
"""
function get_collocation_nodes!(t, disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_collocation_nodes(disc)
    return combine_segments!(t, segment_ends, segment_nodes, compact)
end

"""
    get_nodes(disc::AbstractDiscretization)

Return the nodes for a given discretization
"""
function get_nodes(disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_nodes(disc)
    compact = iscompact(disc)
    return combine_segments(segment_ends, segment_nodes, compact)
end

"""
    get_nodes(disc::AbstractDiscretization)

In-place version of [`get_nodes`](@ref)
"""
function get_nodes!(t, disc::AbstractDiscretization)
    segment_ends = get_segment_ends(disc)
    segment_nodes = get_segment_nodes(disc)
    compact = iscompact(disc)
    return combine_segments!(t, segment_ends, segment_nodes, compact)
end

"""
    number_of_segments(disc::AbstractDiscretization)

Return the number of segments in a given discretization
"""
number_of_segments(disc::AbstractDiscretization) = length(get_segment_ends(disc)) - 1

"""
    number_of_segment_discretization_nodes(disc::AbstractDiscretization)

Return the number of discretization nodes in a segment for a given discretization
"""
number_of_segment_discretization_nodes(disc::AbstractDiscretization) = length(get_segment_discretization_nodes(disc))

"""
    number_of_segment_collocation_nodes(disc::AbstractDiscretization)

Return the number of discretization nodes in a segment for a given discretization
"""
number_of_segment_collocation_nodes(disc::AbstractDiscretization) = length(get_segment_collocation_nodes(disc))

"""
    number_of_segment_nodes(disc::AbstractDiscretization)

Return the number of nodes in a segment for a given discretization
"""
number_of_segment_nodes(disc::AbstractDiscretization) = length(get_segment_nodes(disc))

"""
    number_of_discretization_nodes(disc::AbstractDiscretization)

Return the number of discretization nodes for a given discretization
"""
function number_of_discretization_nodes(disc::AbstractDiscretization)
    nseg = number_of_segments(disc)
    segment_nodes = get_segment_discretization_nodes(disc)
    nnodes = length(segment_nodes)
    compact = iscompact(disc) && iszero(segment_nodes[1]) && isone(segment_nodes[end])
    return (nseg)*(nnodes-compact)+compact
end

"""
    number_of_collocation_nodes(disc::AbstractDiscretization)

Return the number of collocation nodes for a given discretization
"""
function number_of_collocation_nodes(disc::AbstractDiscretization)
    nseg = number_of_segments(disc)
    segment_nodes = get_segment_collocation_nodes(disc)
    nnodes = length(segment_nodes)
    compact = iscompact(disc) && iszero(segment_nodes[1]) && isone(segment_nodes[end])
    return (nseg)*(nnodes-compact)+compact
end

"""
    number_of_nodes(disc::AbstractDiscretization)

Return the number of nodes for a given discretization
"""
function number_of_nodes(disc::AbstractDiscretization)
    nseg = number_of_segments(disc)
    segment_nodes = get_segment_nodes(disc)
    nnodes = length(segment_nodes)
    compact = iscompact(disc) && iszero(segment_nodes[1]) && isone(segment_nodes[end])
    return (nseg)*(nnodes-compact)+compact
end

"""
    set_collocation_node_values!(y, dy, disc::AbstractDiscretization)

Set the values of the collocation nodes in `y` provided that the values of the
disretization nodes in `y` and `dy` have already been calculated.
"""
set_collocation_node_values!

"""
    calculate_defects(y, dy, disc::AbstractDiscretization)

Calculates the defects of a polynomial fit given the states `y` and their state
rates `dy` at the discretization nodes.
"""
calculate_defects(y, dy, disc::AbstractDiscretization) = calculate_defects(similar(dy), y, dy, disc)

"""
    calculate_defects!(r, y, dy, disc::AbstractDiscretization)

In-place version of [`calculate_defects`](@ref)
"""
calculate_defects!

# --- Internal Functions --- #

"""
    combine_segments(segment_ends, segment_nodes, compact)

Combines multiple segments with ends in `segment_ends` and normalized node
positions in `segment_nodes` into a single vector
"""
function combine_segments(segment_ends, segment_nodes, compact)
    TF = promote_type(eltype(segment_ends), eltype(segment_nodes))
    nseg = length(segment_ends) - 1 # number of segments
    nnodes = length(segment_nodes) # number of nodes per segment
    compact = compact && iszero(segment_nodes[1]) && isone(segment_nodes[end])
    nt = (nseg)*(nnodes-compact)+compact
    t = zeros(TF, nt)
    return combine_segments!(t, segment_ends, segment_nodes, compact)
end

"""
    combine_segments!(segment_ends, segment_nodes, compact)

In-place version of [`combine_segments`](@ref)
"""
function combine_segments!(t, segment_ends, segment_nodes, compact)
    nseg = length(segment_ends) - 1 # number of segments
    nnodes = length(segment_nodes) # number of nodes per segment
    compact = compact && iszero(segment_nodes[1]) && isone(segment_nodes[end])
    it = 0 # discretization node vector index
    for iseg = 1:nseg
        # initial time, final time, and duration of this segment
        t0 = segment_ends[iseg]
        tf = segment_ends[iseg+1]
        dt = tf - t0
        # populate segment nodes
        for i = 1:nnodes
            t[it + i] = segment_nodes[i]*dt + t0
        end
        # move to next segment
        it = compact ? it + nnodes - compact : it + nnodes
    end
    return t
end
