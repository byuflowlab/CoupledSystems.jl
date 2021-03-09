"""
    AbstractDiffMethod

Abstract type representing different differentiation methods
"""
abstract type AbstractDiffMethod end

# --- differentiation method --- #

"""
    AbstractAD

Abstract type representing different automatic differentation methods
"""
abstract type AbstractAD <: AbstractDiffMethod end

"""
    ForwardAD{T} <: AbstractAD

Forward automatic differentiation using the ForwardDiff package.

# Fields:
 - `chunk::T`: Chunk size
"""
struct ForwardAD{T} <: AbstractAD
    chunk::T
end

# use default chunk size if chunk is not specified
ForwardAD() = ForwardAD(nothing)

"""
    ReverseAD <: AbstractAD

Reverse automatic differentiation using the ReverseDiff package.

"""
struct ReverseAD <: AbstractAD end

"""
    AbstractFD

Abstract type representing different finite difference methods
"""
abstract type AbstractFD <: AbstractDiffMethod end

"""
    ForwardFD <: AbstractFD

Forward finite differencing as implemented by the FiniteDiff package
"""
struct ForwardFD <: AbstractFD end

"""
    CentralFD <: AbstractFD

Central finite differencing as implemented by the FiniteDiff package
"""
struct CentralFD <: AbstractFD end

"""
    ComplexFD <: AbstractFD

Complex step finite differencing as implemented by the FiniteDiff package
"""
struct ComplexFD <: AbstractFD end

"""
    AbstractChainRuleMode

Type representing direction in which to to apply the chain rule when computing
the derivatives of an explicit system.
"""
abstract type AbstractChainRuleMode end

"""
    Forward <: AbstractChainRuleMode

Type indicating that the chain rule should be applied in forward mode
"""
struct Forward <: AbstractChainRuleMode end

"""
    Reverse <: AbstractChainRuleMode

Type indicating that the chain rule should be applied in reverse mode
"""
struct Reverse <: AbstractChainRuleMode end

"""
    AbstractSensitivityMode

Abstract type representing mode by which to apply the analytic sensitivity
equations when computing derivatives of an explicit component created by
combining an implicit component, output component, and solver.
"""
abstract type AbstractSensitivityMode end

"""
    Direct <: AbstractSensitivityMode

Type indicating that the analytic sensitivity equations should be applied using
the direct method.
"""
struct Direct <: AbstractSensitivityMode end

"""
    Adjoint <: AbstractSensitivityMode

Type indicating that the analytic sensitivity equations should be applied using
the adjoint method.
"""
struct Adjoint <: AbstractSensitivityMode end

"""
    finitediff_type(mode::AbstractFD)

Converts the types used by this package to the inputs expected by the FiniteDiff
package.
"""
finitediff_type
finitediff_type(::ForwardFD) = Val(:forward)
finitediff_type(::CentralFD) = Val(:central)
finitediff_type(::ComplexFD) = Val(:complex)

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

# --- jacobian function creation --- #

"""
    create_output_jacobian_functions(f!, x0, y0, dtype, sp)

Creates functions to compute the jacobian using the specified derivative
type and sparsity pattern.

# Arguments
- `f!`: In-place version of the output function: `y = f(x)`
- `x0::AbstractVector`: Initial values for inputs (used for size and type information)
- `y0::AbstractVector`: Initial values for outputs (used for size and type information)
- `dtype::AbstractDiffMethod`: Method used to calculate derivatives
- `sparsity::AbstractSparsityPattern`: Jacobian sparsity structure
"""
create_output_jacobian_functions

function create_output_jacobian_functions(f!, x0, y0, dtype::ForwardAD, sp::DensePattern)

    # set chunk size
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(x0) : dtype.chunk

    # configure ForwardDiff
    cfg = ForwardDiff.JacobianConfig(f!, y0, x0, chunk)

    # check tag now since we disable it when running our function
    ForwardDiff.checktag(cfg, f!, x0)

    # define jacobian function
    ycache = copy(y0)
    df! = function(dydx, x)
        ForwardDiff.jacobian!(dydx, f!, ycache, x, cfg, Val{false}())
        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        ForwardDiff.jacobian!(dydx, f!, y, x, cfg, Val{false}())
        return y, dydx
    end

    return df!, fdf!
end

function create_output_jacobian_functions(f!, x0, y0, dtype::ReverseAD, sp::DensePattern)

    f_tape = ReverseDiff.JacobianTape(f!, y0, x0)
    cache = ReverseDiff.compile(f_tape)
    result = DiffResults.JacobianResult(y0, x0)

    # define jacobian function
    df! = function(dydx, x)
        ReverseDiff.jacobian!(result, cache, x)
        copyto!(dydx, DiffResults.jacobian(result))
        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        ReverseDiff.jacobian!(result, cache, x)
        copyto!(y, DiffResults.value(result))
        copyto!(dydx, DiffResults.jacobian(result))
        return y, dydx
    end

    return df!, fdf!
end

function create_output_jacobian_functions(f!, x0, y0, dtype::AbstractFD, sp::DensePattern)

    # Construct FiniteDiff Cache
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(copy(x0), copy(y0), copy(y0), fdtype)

    # define jacobian function
    ycache = copy(y0)
    df! = function(dydx, x)
        if eltype(x) <: eltype(ycache)
            # don't allocate because we can store the result
            f!(ycache, x)
            FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache, ycache)
        else
            # allocate because we can't store the result
            FiniteDiff.finite_difference_jacobian!(dydx, f!, x)
        end

        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        if eltype(y) <: eltype(ycache)
            # don't allocate because we can store the result
            FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache, y)
        else
            # allocate because we can't store the result
            FiniteDiff.finite_difference_jacobian!(dydx, f!, x)
        end
        return y, dydx
    end

    return df!, fdf!
end

function create_output_jacobian_functions(f!, x0, y0, dtype::ForwardAD, sp::SparsePattern)

    # get chunk size for ForwardDiff
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(x0) : dtype.chunk

    # construct cache
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = SparseDiffTools.ForwardColorJacCache(f!, x0, chunk; dx=y0,
        colorvec=colorvec, sparsity=sparsity)

    # define jacobian function
    df! = function(dydx, x)
        SparseDiffTools.forwarddiff_color_jacobian!(dydx, f!, x, cache)
        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        SparseDiffTools.forwarddiff_color_jacobian!(dydx, f!, x, cache)
        return y, dydx
    end

    return df!, fdf!
end

function create_output_jacobian_functions(f!, x0, y0, dtype::AbstractFD, sp::SparsePattern)

    # Construct FiniteDiff Cache
    fdtype = FD <: CentralFD ? Val(:central) : Val(:reverse)
    cache = FiniteDiff.JacobianCache(copy(x0), copy(y0), copy(y0), fdtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)

    # define jacobian function
    ycache = copy(y0)
    df! = function(dydx, x)
        f!(ycache, x)
        FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache, ycache)
        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache, y)
        return y, dydx
    end

    return df!, fdf!
end


"""
    create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype, sp)

Creates functions to compute the jacobian of the residual with respect to the
inputs using the specified derivative type and sparsity pattern.

# Arguments
- `f!`: In-place residual function `f(r, x, y)`.
- `x0`: Initial values for inputs (used for size and type information)
- `y0`: Initial values for outputs (used for size and type information)
- `r0`: Initial values for residuals (used for size and type information)
- `dtype`: Method to use to calculate derivatives
- `sparsity`: Jacobian sparsity structure
"""
create_residual_input_jacobian_functions

function create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype::ForwardAD, sp::DensePattern)

    # construct function to take derivative of
    ycache = copy(y0)
    fx! = (r, x) -> f!(r, x, ycache)

    # set chunk size
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(x0) : dtype.chunk

    # configure ForwardDiff
    cfg = ForwardDiff.JacobianConfig(fx!, r0, x0, chunk)

    # check tag now since we disable it when running our function
    ForwardDiff.checktag(cfg, fx!, x0)

    # define jacobian function
    rcache = copy(r0)
    dfdx! = function(drdx, x, y)
        copyto!(ycache, y)
        ForwardDiff.jacobian!(drdx, fx!, rcache, x, cfg, Val{false}())
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        copyto!(ycache, y)
        ForwardDiff.jacobian!(drdx, fx!, r, x, cfg, Val{false}())
        return r, drdx
    end

    return dfdx!, fdfdx!
end

function create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype::ReverseAD, sp::DensePattern)

    f_tape = ReverseDiff.JacobianTape(f!, r0, (x0, y0))
    cache = ReverseDiff.compile(f_tape)
    xres = DiffResults.JacobianResult(r0, x0)
    yres = DiffResults.JacobianResult(r0, y0)

    # define jacobian function
    dfdx! = function(drdx, x, y)
        ReverseDiff.jacobian!((xres, yres), cache, (x, y))
        copyto!(drdx, DiffResults.jacobian(xres))
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        ReverseDiff.jacobian!((xres, yres), cache, (x, y))
        copyto!(r, DiffResults.value(xres))
        copyto!(drdx, DiffResults.jacobian(xres))
        return r, drdx
    end

    return dfdx!, fdfdx!
end

function create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype::AbstractFD, sp::DensePattern)

    # construct function to take derivative of
    ycache = copy(y0)
    fx! = (r, x) -> f!(r, x, ycache)

    # Construct FiniteDiff Cache
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(copy(x0), copy(r0), copy(r0), fdtype)

    # define jacobian function
    rcache = copy(r0)
    dfdx! = function(drdx, x, y)
        if eltype(y) <: eltype(ycache)
            # don't allocate because we can store the result
            copyto!(ycache, y)
            f!(rcache, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache, rcache)
        else
            # allocate because we can't store the result
            fx_new! = (r, x) -> f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx!, x)
        end
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        if eltype(y) <: eltype(ycache)
            # don't allocate because we can store the result
            copyto!(ycache, y)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache, r)
        else
            # allocate because we can't store the result
            fx_new! = (r, x) -> f!(r, x, y)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx_new!, x)
        end
        return r, drdx
    end

    return dfdx!, fdfdx!
end

function create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype::ForwardAD, sp::SparsePattern)

    # construct function to take derivative of
    ycache = copy(y0)
    fx! = (r, x) -> f!(r, x, ycache)

    # get chunk size for ForwardDiff
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(x0) : dtype.chunk

    # construct cache
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = SparseDiffTools.ForwardColorJacCache(fx!, x0, chunk; dx=r0,
        colorvec=colorvec, sparsity=sparsity)

    # define jacobian function
    dfdx! = function(drdx, x, y)
        copyto!(ycache, y)
        SparseDiffTools.forwarddiff_color_jacobian!(drdx, fx!, x, cache)
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        copyto!(ycache, y)
        f!(r, x, y)
        SparseDiffTools.forwarddiff_color_jacobian!(drdx, fx!, x, cache)
        return r, drdx
    end

    return dfdx!, fdfdx!
end

function create_residual_input_jacobian_functions(f!, x0, y0, r0, dtype::AbstractFD, sp::SparsePattern)

    # construct function to take derivative of
    ycache = copy(y0)
    fx! = (r, x) -> f!(r, x, ycache)

    # Construct FiniteDiff Cache
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(copy(x0), copy(r0), copy(r0), fdtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)

    # define jacobian function
    rcache = copy(r0)
    dfdx! = function(drdx, x, y)
        copyto!(ycache, y)
        f!(rcache, x, y)
        FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache, rcache)
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        copyto!(ycache, y)
        f!(r, x, y)
        FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache, r)
        return r, drdx
    end

    return dfdx!, fdfdx!
end

"""
    create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype, sp)

Creates functions to compute the jacobian of the residual with respect to the
inputs using the specified derivative type and sparsity pattern.

# Arguments
- `f!`: In-place residual function `f(r, x, y)`.
- `x0`: Initial values for inputs (used for size and type information)
- `y0`: Initial values for outputs (used for size and type information)
- `r0`: Initial values for residuals (used for size and type information)
- `dtype`: Method to use to calculate derivatives
- `sparsity`: Jacobian sparsity structure
"""
create_residual_output_jacobian_functions

function create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype::ForwardAD, sp::DensePattern)

    # construct function to take derivative of
    xcache = copy(x0)
    fy! = (r, y) -> f!(r, xcache, y)

    # set chunk size
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(y0) : dtype.chunk

    # configure ForwardDiff
    cfg = ForwardDiff.JacobianConfig(fy!, r0, y0, chunk)

    # check tag now since we disable it when running our function
    ForwardDiff.checktag(cfg, fy!, y0)


    # define jacobian function
    rcache = copy(r0)
    dfdy! = function(drdy, x, y)
        copyto!(xcache, x)
        ForwardDiff.jacobian!(drdy, fy!, rcache, y, cfg, Val{false}())
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        copyto!(xcache, x)
        ForwardDiff.jacobian!(drdy, fy!, r, y, cfg, Val{false}())
        return r, drdy
    end

    return dfdy!, fdfdy!
end

function create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype::ReverseAD, sp::DensePattern)

    f_tape = ReverseDiff.JacobianTape(f!, r0, (x0, y0))
    cache = ReverseDiff.compile(f_tape)
    xres = DiffResults.JacobianResult(r0, x0)
    yres = DiffResults.JacobianResult(r0, y0)

    # define jacobian function
    dfdy! = function(drdy, x, y)
        ReverseDiff.jacobian!((xres, yres), cache, (x, y))
        copyto!(drdy, DiffResults.jacobian(yres))
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        ReverseDiff.jacobian!((xres, yres), cache, (x, y))
        copyto!(r, DiffResults.value(yres))
        copyto!(drdy, DiffResults.jacobian(yres))
        return r, drdy
    end

    return dfdy!, fdfdy!
end

function create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype::AbstractFD, sp::DensePattern)

    # construct function to take derivative of
    xcache = copy(x0)
    fy! = (r, y) -> f!(r, xcache, y)

    # Construct FiniteDiff Cache
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(copy(y0), copy(r0), copy(r0), fdtype)

    # define jacobian function
    rcache = copy(r0)
    dfdy! = function(drdy, x, y)
        if eltype(x) <: eltype(xcache)
            # don't allocate because we can store the result
            copyto!(xcache, x)
            f!(rcache, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache, rcache)
        else
            # allocate because we can't store the result
            fy_new! = (r, y) -> f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y)
        end
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        if eltype(x) <: eltype(xcache)
            # don't allocate because we can store the result
            copyto!(xcache, x)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache, r)
        else
            # allocate because we can't store the result
            fy_new! = (r, y) -> f!(r, x, y)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y)
        end
        return r, drdy
    end

    return dfdy!, fdfdy!
end

function create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype::ForwardAD, sp::SparsePattern)

    # construct function to take derivative of
    xcache = copy(x0)
    fy! = (r, y) -> f!(r, xcache, y)

    # get chunk size for ForwardDiff
    chunk = isnothing(dtype.chunk) ? ForwardDiff.Chunk(y0) : dtype.chunk

    # construct cache
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = SparseDiffTools.ForwardColorJacCache(fy!, y0, chunk; dx=r0,
        colorvec=colorvec, sparsity=sparsity)

    # define jacobian function
    dfdy! = function(drdy, x, y)
        copyto!(xcache, x)
        SparseDiffTools.forwarddiff_color_jacobian!(drdy, fy!, y, cache)
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        copyto!(xcache, x)
        f!(r, x, y)
        SparseDiffTools.forwarddiff_color_jacobian!(drdy, fy!, y, cache)
        return r, drdy
    end

    return dfdy!, fdfdy!
end

function create_residual_output_jacobian_functions(f!, x0, y0, r0, dtype::AbstractFD, sp::SparsePattern)

    # construct function to take derivative of
    xcache = copy(x0)
    fy! = (r, y) -> f!(r, xcache, y)

    # Construct FiniteDiff Cache
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(copy(y0), copy(r0), copy(r0), fdtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)

    # define jacobian function
    rcache = copy(r0)
    dfdy! = function(drdy, x, y)
        copyto!(xcache, x)
        f!(rcache, x, y)
        FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache, rcache)
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        copyto!(xcache, x)
        f!(r, x, y)
        FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache, r)
        return r, drdy
    end

    return dfdy!, fdfdy!
end
