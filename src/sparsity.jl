# NOTE: The code in this file comes from SNOW.jl, eventually it might be removed
# from here and imported from there.

# --- sparsity patterns --- #

"""
    AbstractSparsityPattern

Abstract type representing the sparsity pattern of a jacobian matrix
"""
abstract type AbstractSparsityPattern end

"""
    DensePattern <: AbstractSparsityPattern

Type which indicates that a jacobian is dense
"""
struct DensePattern <: AbstractSparsityPattern end

"""
    SparsePattern{TI} <: AbstractSparsityPattern

Type which indicates that a jacobian is sparse and indicates the non-zero rows
and columns

# Fields:
 - `rows::Vector{TI}`: Row of each non-zero matrix element
 - `cols::Vector{TI}`: Column of each non-zero matrix element
"""
struct SparsePattern{TI} <: AbstractSparsityPattern
    rows::Vector{TI}
    cols::Vector{TI}
end

"""
    SparsePattern(A::SparseMatrixCSC)
construct sparse pattern from representative sparse matrix
# Arguments
- `A::SparseMatrixCSC`: sparse jacobian
"""
function SparsePattern(A::SparseMatrixCSC)
    rows, cols, _ = findnz(A)
    return SparsePattern(rows, cols)
end

"""
    SparsePattern(A::Matrix)
construct sparse pattern from representative matrix
# Arguments
- `A::Matrix`: sparse jacobian
"""
function SparsePattern(A::Matrix)
    return SparsePattern(sparse(A))
end

"""
    SparsePattern(::ForwardAD, func!, ng, x1, x2, x3)
detect sparsity pattern by computing derivatives (using forward AD)
at three different locations. Entries that are zero at all three
spots are assumed to always be zero.
# Arguments
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `x1,x2,x3::Vector{Float}`:: three input vectors.
"""
function SparsePattern(::ForwardAD, func!, ng, x1, x2, x3)

    g = zeros(ng)
    config = ForwardDiff.JacobianConfig(func!, g, x1)
    J1 = ForwardDiff.jacobian(func!, g, x1, config)
    J2 = ForwardDiff.jacobian(func!, g, x2, config)
    J3 = ForwardDiff.jacobian(func!, g, x3, config)
    @. J1 = abs(J1) + abs(J2) + abs(J3)
    Jsp = sparse(J1)

    return SparsePattern(Jsp)
end

"""
    SparsePattern(::AbstractFD, func!, ng, x1, x2, x3)
detect sparsity pattern by computing derivatives (using finite differencing)
at three different locations. Entries that are zero at all three
spots are assumed to always be zero.
# Arguments
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `x1,x2,x3::Vector{Float}`:: three input vectors.
"""
function SparsePattern(dtype::AbstractFD, func!, ng,
    x1, x2, x3)

    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(x1, zeros(ng), fdtype)

    nx = length(x1)
    J1 = zeros(ng, nx)
    J2 = zeros(ng, nx)
    J3 = zeros(ng, nx)
    FiniteDiff.finite_difference_jacobian!(J1, func!, x1, cache)
    FiniteDiff.finite_difference_jacobian!(J2, func!, x2, cache)
    FiniteDiff.finite_difference_jacobian!(J3, func!, x3, cache)

    @. J1 = abs(J1) + abs(J2) + abs(J3)
    Jsp = sparse(J1)

    return SparsePattern(Jsp)
end

"""
    SparsePattern(diffmethod, func!, ng, lx, ux)
detect sparsity pattern by computing derivatives
at three randomly generating locations w/in bounds.
Entries that are zero at all three spots are assumed to always be zero.
# Arguments
- `diffmethod::AbstractDiffMethod`: method to compute derivatives
- `func!::Function`: function of form f = func!(g, x)
- `ng::Int`: number of constraints
- `lx::Vector{Float}`: lower bounds on x
- `ux::Vector{Float}`: upper bounds on x
"""
function SparsePattern(diffmethod, func!, ng, lx, ux)
    r = rand(3)
    x1 = @. (1-r[1])*lx + r[1]*ux
    x2 = @. (1-r[2])*lx + r[2]*ux
    x3 = @. (1-r[3])*lx + r[3]*ux
    return SparsePattern(diffmethod, func!, ng, x1, x2, x3)
end

"""
    getsparsity(::DensePattern, nx, nf)

Get rows and cols for dense jacobian
"""
function getsparsity(::DensePattern, nx, nf)
    len = nf*nx
    rows = [i for i = 1:nf, j = 1:nx][:]
    cols = [j for i = 1:nf, j = 1:nx][:]
    return rows, cols
end

"""
    getsparsity(sp::SparsePattern, nx, nf)

Get rows and cols for sparse jacobian
"""
getsparsity(sp::SparsePattern, nx, nf) = sp.rows, sp.cols
