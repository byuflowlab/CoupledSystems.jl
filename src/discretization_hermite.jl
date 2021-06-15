"""
   HermiteDiscretization{TE, TN, TC, TV, TD} <: AbstractDiscretization

Discretization method that uses hermite interpolation to define a polynomial fit
to an ODE.  The defect of the fit is assessed at interpolated nodes.

# Fields:
 - `segment_ends::TE`: The ends of each segment, normalized to be between 0 and 1
 - `segment_nodes::TN`: The nodes corresponding to each segment, normalized to be
    between 0 and 1.
 - `discretization_nodes::Vector{Int}`: Indices of the discretization nodes for each segment
 - `collocation_nodes::Vector{Int}`: Indices of the collocation nodes for each segment
 - `value_matrices::TV`:  Matrices used to determine the predicted values of the
    states at the collocation nodes (using the polynomial fit)
 - `derivative_matrices::TD`:  Matrices used to determine the predicted
    state rates at the collocation nodes (using the polynomial fit)
 - `compact::Bool`: Flag indicating whether the representation is compact
"""
struct HermiteDiscretization{TE, TN, TV, TD, TC} <: AbstractDiscretization
    segment_ends::TE
    segment_nodes::TN
    discretization_nodes::Vector{Int}
    collocation_nodes::Vector{Int}
    value_matrices::TV
    derivative_matrices::TD
    compact::TC
end

get_segment_ends(disc::HermiteDiscretization) = disc.segment_ends

get_segment_discretization_nodes(disc::HermiteDiscretization) = view(disc.segment_nodes, disc.discretization_nodes)

get_segment_collocation_nodes(disc::HermiteDiscretization) = view(disc.segment_nodes, disc.collocation_nodes)

get_segment_nodes(disc::HermiteDiscretization) = disc.segment_nodes

get_discretization_indices(disc::HermiteDiscretization) = disc.discretization_nodes

get_collocation_indices(disc::HermiteDiscretization) = disc.collocation_nodes

iscompact(::HermiteDiscretization{TE, TN, TV, TD, Val{C}}) where {TE, TN, TV, TD, C} = C

function set_collocation_node_values!(y, dy, tdseg, disc::HermiteDiscretization)
    # interpolate values from discretization nodes to collocation nodes
    y_d = view(y, disc.discretization_nodes)
    dy_d = view(y, disc.discretization_nodes)
    y_c = view(y, disc.collocation_nodes)
    M, dM = disc.value_matrices
    # y_c = M*y_d + dM*dy_d*tdseg
    mul!(mul!(y_c, M, y_d), dM, dy_d, tdseg, 1)
    return y
end

set_collocation_node_values_u!(y_u, dy_u, tdseg, disc::HermiteDiscretization) =
    set_collocation_node_values(y_u, dy_u, tdseg, disc)

set_collocation_node_values_c!(y_c, dy_c, tdseg, disc::HermiteDiscretization) =
    set_collocation_node_values(y_c, dy_c, tdseg, disc)

set_collocation_node_values_p!(y_p, dy_p, tdseg, disc::HermiteDiscretization) =
    set_collocation_node_values(y_p, dy_p, tdseg, disc)

set_collocation_node_values_ti!(y_ti, dy_ti, tdseg, disc::HermiteDiscretization) =
    set_collocation_node_values(y_ti, dy_ti, tdseg, disc)

function set_collocation_node_values_td!(y_td, dy_td, tdseg, dy, tdseg_td, disc::HermiteDiscretization)
    # interpolate values from discretization nodes to collocation nodes
    dy_d = view(y, disc.discretization_nodes)
    y_td_d = view(y_td, disc.discretization_nodes)
    dy_td_d = view(dy_td, disc.discretization_nodes)
    y_td_c = view(y_td, disc.collocation_nodes)
    M, dM = disc.value_matrices
    # y_td_c = M*y_td_d + dM*dy_td_d * tdseg + dM*dy_d * tdseg_td
    mul!(mul!(mul!(dy_td_c, M, y_td_d), dM, dy_td_d, tdseg, 1), dM, dy_d, tdseg_td, 1)
    return y_td
end

function calculate_defects!(r, y, dy, tdseg, disc::HermiteDiscretization)
    # extract values and rates at discretization nodes
    y_d = view(y, disc.discretization_nodes)
    dy_d = view(dy, disc.discretization_nodes)
    # extract state rates at collocation nodes
    dy_c = view(dy, disc.collocation_nodes)
    # extract interpolation matrices
    A, B = disc.derivative_matrices
    # construct residual expression: `r = dybar - dy` where `dybar = A*y + B*dy`
    copyto(r, view(dy, disc.collocation_nodes))
    r .*= tdseg # scale state rates to uniform segment length
    mul!(mul!(ybar, A, y_d, 1, -1), B, dy_d, tdseg, 1)
    return r
end

"""
    HermiteDiscretization(segment_ends, segment_nodes, collocation_nodes, compact=Val(true))

Constructs a discretization of a function into polynomials given the segment
ends and the nodes corresponding to each segment.
"""
function HermiteDiscretization(segment_ends, segment_nodes, collocation_nodes, compact=Val(true))
    # sort arrays
    segment_ends = sort(segment_ends)
    segment_nodes = sort(segment_nodes)
    # normalize to be between 0 and 1
    segment_ends = (segment_ends .- first(segment_ends))/(last(segment_ends) - first(segment_ends))
    segment_nodes = (segment_nodes .- first(segment_nodes))/(last(segment_nodes) - first(segment_nodes))
    # discretization nodes are all nodes not assigned to collocation nodes
    discretization_nodes = findall(in(collocation_nodes), 1:length(segment_nodes))
    # get source and destination nodes
    source_nodes = segment_nodes[discretization_nodes]
    destination_nodes = segment_nodes[collocation_nodes]
    # construct matrices for polynomial interpolation
    value_matrices = hermite_interpolation_matrices(source_nodes, destination_nodes)
    derivative_matrices = hermite_derivative_interpolation_matrices(source_nodes, destination_nodes)
    # construct discretization object
    return HermiteDiscretization(segment_ends, segment_nodes, collect(discretization_nodes),
        collect(collocation_nodes), value_matrices, derivative_matrices, compact)
end

"""
    GaussLobatto(segment_ends, nodes_per_segment; compact=Val(true))

Define a discretization for an ODE using Legendre-Gauss-Lobatto collocation.

# Arguments:
 - `segment_ends`: The ends of each segment, normalized to be between 0 and 1
 - `nodes_per_segment`: The number of nodes corresponding to each segment.
 - `compact`: Flag indicating whether a compact representation of the system
    should be used.
"""
function GaussLobatto(segment_ends, nodes_per_segment, compact=Val(true))
    segment_nodes, _ = gausslobatto(2*nodes_per_segment-1)
    collocation_nodes = 2:2:length(segment_nodes)
    return HermiteDiscretization(segment_ends, segment_nodes, collocation_nodes, compact)
end

"""
    interpolate_nodes(y, dy, disc::HermiteDiscretization, scaling=1.0)

Interpolates states from the discretization nodes to the collocation nodes using
polynomial interpolation.
"""
function interpolate_nodes(y, dy, disc::HermiteDiscretization, scaling=1.0)
    W, Wd = disc.hermite_interpolation_matrices
    return W*y + Wd*dy/scaling
end

"""
    calculate_defects(y, dy, disc::HermiteDiscretization, scaling=1.0)

Calculate the defects of a given polynomial fit given the states `y` (which are
used to generate the fit) and the state rates `dy` (which are used to test the fit)
at each of the nodes.  The output is scaled by multiplying by `scaling`.
"""
function calculate_defects(y, dy, disc::HermiteDiscretization)
    W, Wd = disc.interpolation_derivative_matrix
    dybar = W*y
    return (dy - dybar)*scaling
end

"""
    hermite_interpolation_matrices(x, xx)

Construct two matrices `W` and `Wd` that together return the values of a polynomial
interpolation of a function at `xx`, given the function's values and derivatives
at `x`. (e.g. yy = W*y + Wd*dy)
"""
function hermite_interpolation_matrices(x, xx)
    W = xx .* x' .* 0.0 # basis polynomials
    Wd = xx .* x' .* 0.0 # basis polynomials
    for i = 1:length(xx)
        for j = 1:length(x)
            lj = 1 # lagrange basis polynomial
            dlj = 0 # derivative of lagrange basis polynomial
            for k = 1:length(x)
                if j != k
                    lj *= (xx[i]-x[k])/(x[j]-x[k]);
                    dlj += 1/(x[j]-x[k]);
                end
            end
            W[i,j] = (1 - 2*(xx[i]-x[j])*dlj)*lj^2
            Wd[i,j] = (xx[i]-x[j])*lj^2
        end
    end
    return W, Wd
end

"""
    hermite_derivative_interpolation_matrix(x, xx)

Construct two matrices `W` and `Wd` that together return the values of a polynomial
interpolation of a function at `xx`, given the function's values and derivatives
at `x`.  (e.g. dyy = W*y + Wd*dy)
"""
function hermite_derivative_interpolation_matrices(x, xx)
    W = xx .* x' .* 0.0 # basis polynomials
    Wd = xx .* x' .* 0.0 # basis polynomials
    for i = 1:length(xx)
        for j = 1:length(x)
            lj = 1 # lagrange basis polynomial
            dlj = 0 # derivative of lagrange basis polynomial
            lj_xx = 0 # partial of lagrange basis polynomila wrt xx
            for k = 1:length(x)
                if j != k
                    lj *= (xx[i]-x[k])/(x[j]-x[k])
                    dlj += 1/(x[j]-x[k])
                    tmp = 1/(x[j]-x[k])
                    for m = 1:length(x)
                        if m != j && m != k
                            tmp *= (xx[i]-x[m])/(x[j]-x[m])
                        end
                    end
                    lj_xx += tmp
                end
            end
            W[i,j] = -2*dlj*lj^2 + (1 - 2*(xx[i]-x[j])*dlj)* 2 * lj * lj_xx
            Wd[i,j] = lj^2 + (xx[i]-x[j]) * 2 * lj * lj_xx
        end
    end
    return W, Wd
end
