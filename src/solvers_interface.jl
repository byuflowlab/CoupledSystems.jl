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
