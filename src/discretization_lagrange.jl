"""
   LagrangeDiscretization{TE, TN, TD} <: AbstractDiscretization

Discretization method that uses lagrange interpolation to define a polynomial fit
to an ODE.  The defect of the fit assessed at a subset of the nodes used to define
the interpolation.

# Fields:
 - `segment_ends::TE`: The ends of each segment, normalized to be between 0 and 1
 - `segment_nodes::TN`: The nodes corresponding to each segment, normalized to be
    between 0 and 1.
 - `collocation_nodes::Vector{Int}`: Indices of the collocation nodes
 - `derivative_matrix::TD`: Matrix used to determine the predicted
    state rates at the collocation nodes (using the polynomial fit)
 - `compact`: Flag indicating whether a compact representation of the system
    should be used.
"""
struct LagrangeDiscretization{TE, TN, TD, TC} <: AbstractDiscretization
    segment_ends::TE
    segment_nodes::TN
    collocation_nodes::Vector{Int}
    derivative_matrix::TD
    compact::TC
end

get_segment_ends(disc::LagrangeDiscretization) = disc.segment_ends

get_segment_discretization_nodes(disc::LagrangeDiscretization) = disc.segment_nodes

get_segment_collocation_nodes(disc::LagrangeDiscretization) = disc.segment_nodes[disc.collocation_nodes]

get_segment_nodes(disc::LagrangeDiscretization) = disc.segment_nodes

get_discretization_indices(disc::LagrangeDiscretization) = 1:length(disc.segment_nodes)

get_collocation_indices(disc::LagrangeDiscretization) = disc.collocation_nodes

iscompact(::LagrangeDiscretization{TE, TN, TD, Val{C}}) where {TE, TN, TD, C} = C

function set_collocation_node_values!(y, dy, tdseg, disc::LagrangeDiscretization)
    # nothing needs to be done
    return y
end

set_collocation_node_values_u!(y_u, dy_u, tdseg, disc::LagrangeDiscretization) =
    set_collocation_node_values(y_u, dy_u, tdseg, disc)

set_collocation_node_values_c!(y_c, dy_c, tdseg, disc::LagrangeDiscretization) =
    set_collocation_node_values(y_c, dy_c, tdseg, disc)

set_collocation_node_values_p!(y_p, dy_p, tdseg, disc::LagrangeDiscretization) =
    set_collocation_node_values(y_p, dy_p, tdseg, disc)

set_collocation_node_values_ti!(y_ti, dy_ti, tdseg, disc::LagrangeDiscretization) =
    set_collocation_node_values(y_ti, dy_ti, tdseg, disc)

function set_collocation_node_values_td!(y_td, dy_td, tdseg, dy, tdseg_td, disc::LagrangeDiscretization)
    # nothing needs to be done
    return y
end

@inline function calculate_defects!(r, y, dy, tdseg, disc::LagrangeDiscretization)
    # extract values at discretization nodes
    y_d = y
    # extract state rates at collocation nodes
    dy_c = view(dy, disc.collocation_nodes)
    # extract interpolation matrix
    A = disc.derivative_matrix
    # store state rates in residual vector
    copyto!(r, dy_c)
    # scale state rates to uniform segment length
    r .*= tdseg
    # subtract estimated state rates
    mul!(r, A, y_d, 1, -1)
    # Return residual: `r = dybar - dy` where `dybar = A*y`
    return r
end

# --- Constructors --- #

"""
    LagrangeDiscretization(segment_ends, segment_nodes, collocation_nodes,
        compact=Val(true))

Define a discretization that uses lagrange interpolation to define a polynomial fit
to an ODE.  The defect of each fit is assessed at the nodes specified in
`collocation_nodes`

# Arguments:
 - `segment_ends`: The ends of each segment, normalized to be between 0 and 1
 - `segment_nodes`: The nodes corresponding to each segment, normalized to be
    between 0 and 1.
 - `collocation_nodes`: Indices of the nodes at which the defect is assessed.
    The defect of each fit must be assessed at `n-1` nodes where `n` is the
    number of nodes used to define the fit.
 - `compact`: Flag indicating whether a compact representation of the system
    should be used.
"""
function LagrangeDiscretization(segment_ends, segment_nodes, collocation_nodes, compact=Val(true))
    # sort all arrays
    segment_ends = sort(segment_ends)
    segment_nodes = sort(segment_nodes)
    collocation_nodes = sort(collocation_nodes)
    # normalize arrays to be between 0 and 1
    segment_ends = (segment_ends .- first(segment_ends))/(last(segment_ends) - first(segment_ends))
    segment_nodes = (segment_nodes .- first(segment_nodes))/(last(segment_nodes) - first(segment_nodes))
    # get source and destination nodes
    source_nodes = segment_nodes
    destination_nodes = segment_nodes[collocation_nodes]
    # construct matrix for finding derivative of polynomial fit
    W = lagrange_derivative_interpolation_matrix(source_nodes, destination_nodes)
    # construct discretization object
    return LagrangeDiscretization(segment_ends, segment_nodes, collect(collocation_nodes), W, compact)
end

"""
    GaussRadau(segment_ends, nodes_per_segment, compact=Val(true))

Define a discretization for an ODE using the GaussRadau Pseudospectral method.

# Arguments:
 - `segment_ends`: The ends of each segment, normalized to be between 0 and 1
 - `nodes_per_segment`: The number of nodes corresponding to each segment.
 - `compact`: Flag indicating whether a compact representation of the system
    should be used.
"""
function GaussRadau(segment_ends, nodes_per_segment, compact=Val(true))
    segment_nodes, _ = gaussradau(nodes_per_segment-1)
    push!(segment_nodes, 1)
    collocation_nodes = 1:nodes_per_segment-1
    return LagrangeDiscretization(segment_ends, segment_nodes, collocation_nodes, compact)
end

# --- Internal Functions --- #

"""
    lagrange_interpolation_matrix(x, xx)

Construct a matrix `W` that returns the values of a polynomial interpolation of a
function at `xx`, given the function's values at `x`. (e.g. yy = W*y)
"""
function lagrange_interpolation_matrix(x, xx)
    W = xx .* x' .* 0.0
    for i = 1:length(xx)
        for j = 1:length(x)
            W[i, j] = 1.0
            for k = 1:length(x)
                if j != k
                    W[i, j] *= (xx[i] - x[k])/(x[j] - x[k])
                end
            end
        end
    end
    return W
end

"""
    lagrange_derivative_interpolation_matrix(x, xx)

Construct a matrix `W` that returns the first derivative of a polynomial interpolation
of a function at `xx`, given the function's values at `x`. (e.g. dyy = W*y)
"""
function lagrange_derivative_interpolation_matrix(x, xx)
    W = xx .* x' .* 0.0
    for i = 1:length(xx)
        for j = 1:length(x)
            W[i,j] = 0.0
            for k = 1:length(x)
                if j != k
                    tmp = 1/(x[j] - x[k])
                    for m = 1:length(x)
                        if m != j && m != k
                            tmp *= (xx[i] - x[m])/(x[j] - x[m])
                        end
                    end
                    W[i,j] += tmp
                end
            end
        end
    end
    return W
end
