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
    xcache = copy(x0)
    ycache = copy(y0) # this needs to be updated before calling FiniteDiff
    ycache1 = copy(y0)
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(xcache, ycache, ycache1, fdtype)

    # define jacobian function
    df! = function(dydx, x)
        TF = promote_type(eltype(x), eltype(ycache))
        if TF <: eltype(ycache)
            # don't allocate because we can store the result
            f!(ycache, x)
            copyto!(xcache, x)
            FiniteDiff.finite_difference_jacobian!(dydx, f!, xcache, cache, ycache)
        else
            # allocate because we can't store the result
            FiniteDiff.finite_difference_jacobian!(dydx, f!, x)
        end

        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        TF = promote_type(eltype(y), eltype(ycache))
        if TF <: eltype(ycache)
            # don't allocate because we can store the result
            copyto!(xcache, x)
            copyto!(ycache, y)
            FiniteDiff.finite_difference_jacobian!(dydx, f!, xcache, cache)
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
    xcache = copy(x0)
    ycache = copy(y0)
    ycache1 = copy(y0)
    fdtype = finitediff_type(dtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = FiniteDiff.JacobianCache(xcache, ycache, ycache1, fdtype, sparsity,
        colorvec)

    # define jacobian function
    df! = function(dydx, x)
        f!(ycache, x)
        FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache)
        return dydx
    end

    # define combined output and jacobian function
    fdf! = function(y, dydx, x)
        f!(y, x)
        copyto!(ycache, y)
        FiniteDiff.finite_difference_jacobian!(dydx, f!, x, cache)
        return y, dydx
    end

    return df!, fdf!
end
