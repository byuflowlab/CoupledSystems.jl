"""
    AbstractSolver

Supertype for solvers that may be used to solve implicit systems of equations.
"""
abstract type AbstractSolver end

"""
    create_mutating_solver_cache(component::AbstractImplicitComponent, x, solver::AbstractSolver)

Construct a cache for a nonlinear solver which uses mutating functions.  `x` is
a pointer to an array that will be modified to contain the correct input values.
"""
create_mutating_solver_cache(component::AbstractImplicitComponent, x, ::AbstractSolver) = component

"""
    create_nonmutating_solver_cache(component::AbstractImplicitComponent, x, solver::AbstractSolver)

Construct a cache for a nonlinear solver which uses nonmutating functions.  `x` is
a pointer to an array that will be modified to contain the correct input values.
"""
create_nonmutating_solver_cache(component::AbstractImplicitComponent, x, ::AbstractSolver) = component


# --- Newton Solver --- #

"""
    Newton{T1, T2, T3, T4, T5}

# Fields
 - `xtol::T1`: Step size tolerance for ending iterations
 - `ftol::T2`: Function tolerance for ending iterations
 - `iterations::T3`: Maximum number of iterations
 - `linesearch::T4`: Linesearch algorithm
 - `linsolve::T5`: Linear solver algorithm
"""
struct Newton{T1, T2, T3, T4, T5}
    xtol::T1
    ftol::T2
    iterations::T3
    linesearch::T4
    linsolve::T5
end

"""
    Newton(; kwargs...)

# Keyword Arguments
 - `xtol = 0.0`: Step size tolerance for ending iterations
 - `ftol = 1e-8`: Function tolerance for ending iterations
 - `iterations = 1000`: Maximum number of iterations
 - `linesearch = LineSearches.BackTracking(maxstep=1e6)`: Linesearch algorithm
 - `linsolve = (x, A, b) -> copyto!(x, A\b)`: Linear solver function
"""
function Newton(;
    xtol=0.0,
    ftol=1e-8,
    iterations=1000,
    linesearch=BackTracking(maxstep=1e6),
    linsolve = (x, A, b) -> copyto!(x, A\b))
    return Newton(xtol, ftol, iterations, linesearch, linsolve)
end

function create_mutating_solver_cache(component::AbstractImplicitComponent, x, ::Newton)
    f! = function(r, y)
        residuals!(component, x, y)
        copyto!(r, residuals(component))
        return r
    end
    j! = function(drdy, y)
        residual_output_jacobian!(component, x, y)
        copyto!(drdy, residual_output_jacobian(component))
        return drdy
    end
    fj! = function(r, drdy, y)
        residuals_and_output_jacobian!(component, x, y)
        copyto!(r, residuals(component))
        copyto!(drdy, residual_output_jacobian(component))
        return r, drdy
    end
    r = residuals(component)
    drdy = residual_output_jacobian(component)
    y_f = component.y_f
    y_df = component.y_dfdy
    f_calls = [0,]
    df_calls = [0,]
    return OnceDifferentiable(f!, j!, fj!, r, drdy, y_f, y_df, f_calls, df_calls)
end

function create_nonmutating_solver_cache(component::AbstractImplicitComponent, x, ::Newton)
    f! = (r, y) -> residuals!(component, r, x, y)
    j! = (drdy, y) -> residual_output_jacobian!(component, drdy, x, y)
    fj! = (r, drdy, y) -> residuals_and_output_jacobian!(component, r, drdy, x, y)
    r = residuals(component)*one(eltype(x))
    drdy = residual_output_jacobian(component)*one(eltype(x))
    y_f = component.y_f*one(eltype(x))
    y_df = component.y_dfdy*one(eltype(x))
    f_calls = [0,]
    df_calls = [0,]
    return OnceDifferentiable(f!, j!, fj!, r, drdy, y_f, y_df, f_calls, df_calls)
end

# function create_nonmutating_solver_cache(component::AbstractImplicitComponent, x, ::Newton)
#     TF = eltype(x)
#     f! = (r, y) -> residuals!(component, r, x, y)
#     j! = (drdy, y) -> residual_output_jacobian!(component, drdy, x, y)
#     fj! = function(r, drdy, y)
#         residuals_and_output_jacobian!(component, r, drdy, x, y)
#         return r, drdy
#     end
#     y = TF.(component.y_f)
#     r = TF.(residuals(component))
#     return OnceDifferentiable(f!, j!, fj!, y, r)
# end

function (solver::Newton)(cache, x, y0)
    # solve nonlinear system
    results = nlsolve(cache, y0, method=:newton,
        xtol = solver.xtol,
        ftol = solver.ftol,
        iterations = solver.iterations,
        linesearch = solver.linesearch,
        linsolve = solver.linsolve)
    # return NaNs if not converged
    if !results.f_converged
        copyto!(results.zero, NaN)
    end
    # return results
    return results.zero
end
