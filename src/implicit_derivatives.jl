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
    xcache = copy(x0)
    rcache = copy(r0) # this needs to be updated
    rcache1 = copy(r0)
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(xcache, rcache, rcache1, fdtype)

    # define jacobian function
    dfdx! = function(drdx, x, y)
        TF = promote_type(eltype(y), eltype(ycache))
        if TF <: eltype(ycache)
            # don't allocate because we can store the result
            f!(rcache, x, y)
            copyto!(xcache, x)
            copyto!(ycache, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx!, xcache, cache)
        else
            # allocate because we can't store the result
            fx_new! = (r, x) -> f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx_new!, x, fdtype)
        end
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        TF = promote_type(eltype(y), eltype(ycache))
        if TF <: eltype(ycache)
            # don't allocate because we can store the result
            f!(r, x, y)
            copyto!(xcache, x)
            copyto!(ycache, y)
            copyto!(rcache, r)
            FiniteDiff.finite_difference_jacobian!(drdx, fx!, xcache, cache)
        else
            # allocate because we can't store the result
            fx_new! = (r, x) -> f!(r, x, y)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdx, fx_new!, x, fdtype)
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
    xcache = copy(x0)
    rcache = copy(r0) # this needs to be updated before calling FiniteDiff
    rcache1 = copy(r0)
    fdtype = finitediff_type(dtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = FiniteDiff.JacobianCache(xcache, rcache, rcache1, fdtype, sparsity,
        colorvec)

    # define jacobian function
    dfdx! = function(drdx, x, y)
        copyto!(ycache, y)
        f!(rcache, x, y)
        FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache)
        return drdx
    end

    # define combined output and jacobian function
    fdfdx! = function(r, drdx, x, y)
        f!(r, x, y)
        copyto!(ycache, y)
        copyto!(rcache, r)
        FiniteDiff.finite_difference_jacobian!(drdx, fx!, x, cache)
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
    ycache = copy(y0)
    rcache = copy(r0) # this needs to be updated
    rcache1 = copy(r0)
    fdtype = finitediff_type(dtype)
    cache = FiniteDiff.JacobianCache(ycache, rcache, rcache1, fdtype)

    # define jacobian function
    dfdy! = function(drdy, x, y)
        TF = promote_type(eltype(y), eltype(ycache))
        if TF <: eltype(xcache)
            # don't allocate because we can store the result
            f!(rcache, x, y)
            copyto!(xcache, x)
            copyto!(ycache, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, ycache, cache)
        else
            # allocate because we can't store the result
            fy_new! = (r, y) -> f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, fdtype)
        end
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        TF = promote_type(eltype(x), eltype(xcache))
        if TF <: eltype(xcache)
            # don't allocate because we can store the result
            f!(r, x, y)
            copyto!(xcache, x)
            copyto!(ycache, y)
            copyto!(rcache, r)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, ycache, cache)
        else
            # allocate because we can't store the result
            fy_new! = (r, y) -> f!(r, x, y)
            f!(r, x, y)
            FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, fdtype)
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
    ycache = copy(y0)
    rcache = copy(r0) # this needs to be updated before calling FiniteDiff
    rcache1 = copy(r0)
    fdtype = finitediff_type(dtype)
    sparsity = sparse(sp.rows, sp.cols, ones(length(sp.rows)))
    colorvec = SparseDiffTools.matrix_colors(sparsity)
    cache = FiniteDiff.JacobianCache(ycache, rcache, rcache1, fdtype, sparsity,
        colorvec)

    # define jacobian function
    dfdy! = function(drdy, x, y)
        copyto!(xcache, x)
        f!(rcache, x, y)
        FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache)
        return drdy
    end

    # define combined output and jacobian function
    fdfdy! = function(r, drdy, x, y)
        f!(r, x, y)
        copyto!(xcache, x)
        copyto!(rcache, r)
        FiniteDiff.finite_difference_jacobian!(drdy, fy!, y, cache)
        return r, drdy
    end

    return dfdy!, fdfdy!
end
