"""
    TrustRegion{T} <: AbstractSolver

Trust region solver, as provided by the NLsolve package.  Stores the keyword
arguments which are passed to NLsolve in the field `kwargs`
"""
struct TrustRegion{T} <: AbstractSolver
    kwargs::T
end

"""
    TrustRegion(; kwargs...)

Trust region solver, as provided by the NLsolve package.  Uses the same default
arguments as NLsolve.

The relevant keyword arguments and their default values are repeated here for
convenience.

# Keyword Arguments
 - `xtol = 0.0`: Step size tolerance for ending iterations
 - `ftol = 1e-8`: Function tolerance for ending iterations
 - `iterations = 1000`: Maximum number of iterations
 - `factor = 1.0`: Factor for determining initial size of trust region
 - `autoscale = true`: Autoscale variables?
 - `linsolve = (x, A, b) -> copyto!(x, A\b)`: Linear solver function
"""
function TrustRegion(; kwargs...)
    return TrustRegion(kwargs)
end

function (solver::TrustRegion)(cache, x, y0)
    # solve nonlinear system
    results = nlsolve(cache, y0, method=:trust_region; solver.kwargs...)
    # return NaNs if not converged
    if !results.f_converged
        copyto!(results.zero, NaN)
    end
    # return results
    return results.zero
end

# NLsolve uses the same cache for TrustRegion and Newton
function create_mutating_solver_cache(component::AbstractImplicitComponent, x, ::TrustRegion)
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

# NLsolve uses the same cache for TrustRegion and Newton
function create_nonmutating_solver_cache(component::AbstractImplicitComponent, x, ::TrustRegion)
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
