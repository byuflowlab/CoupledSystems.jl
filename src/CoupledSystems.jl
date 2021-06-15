module CoupledSystems

using LinearAlgebra
using SparseArrays, SparseDiffTools
using FiniteDiff, ForwardDiff, ReverseDiff, DiffResults
using NLsolve
using DifferentialEquations
using FastGaussQuadrature

import Base.setindex, Base.front, Base.tail

include("component_types.jl")
include("differentiation_types.jl")
include("sparsity.jl")

include("variable_interface.jl")
include("variable_connections.jl")
include("variable_utils.jl")

include("solvers_interface.jl")
include("solvers_newton.jl")
include("solvers_trustregion.jl")

include("discretization_interface.jl")
include("discretization_lagrange.jl")
include("discretization_hermite.jl")

include("explicit_constructors.jl")
include("explicit_derivatives.jl")
include("explicit_interface.jl")
include("explicit_utils.jl")

include("implicit_constructors.jl")
include("implicit_derivatives.jl")
include("implicit_interface.jl")
include("implicit_utils.jl")

# Variable Interface
export @named
export NamedVar
export named
export value
export combine
export combine!
export separate
export separate!

# Differentiation Types
export AbstractDiffMethod
export AbstractAD, ForwardAD, ReverseAD
export AbstractFD, ForwardFD, CentralFD, ComplexFD
export Forward, Reverse
export Direct, Adjoint

# Sparsity types and functions
export AbstractSparsityPattern, DensePattern, SparsePattern
export get_sparsity

# Component Types
export AbstractComponent
export AbstractExplicitComponent, ExplicitComponent, ExplicitSystem
export AbstractImplicitComponent, ImplicitComponent, ImplicitSystem
export make_explicit, make_implicit, add_solver

# Solver Types
export AbstractSolver, Newton, TrustRegion

# Discretization Types
export AbstractDiscretization
export LagrangeDiscretization, GaussRadau
export HermiteDiscretization, GaussLobatto

# Explicit Component Interface
export outputs, outputs!, outputs!!, outputs!!!
export jacobian, jacobian!, jacobian!!, jacobian!!!
export outputs_and_jacobian, outputs_and_jacobian!, outputs_and_jacobian!!, outputs_and_jacobian!!!

# Implicit Component Interface
export residuals, residuals!, residuals!!, residuals!!!
export residual_input_jacobian, residual_input_jacobian!, residual_input_jacobian!!, residual_input_jacobian!!!
export residual_output_jacobian, residual_output_jacobian!, residual_output_jacobian!!, residual_output_jacobian!!!
export residuals_and_input_jacobian, residuals_and_input_jacobian!, residuals_and_input_jacobian!!, residuals_and_input_jacobian!!!
export residuals_and_output_jacobian, residuals_and_output_jacobian!, residuals_and_output_jacobian!!, residuals_and_output_jacobian!!!
export residuals_and_jacobians, residuals_and_jacobians!, residuals_and_jacobians!!, residuals_and_jacobians!!!

end
