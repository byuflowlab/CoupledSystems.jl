module CoupledSystems

using LinearAlgebra
using SparseArrays, SparseDiffTools
using FiniteDiff, ForwardDiff, ReverseDiff, DiffResults
using NLsolve

export AbstractDiffMethod
export AbstractAD, ForwardAD, ReverseAD
export AbstractFD, ForwardFD, CentralFD, ComplexFD
export Forward, Reverse
export Direct, Adjoint

export AbstractSparsityPattern, DensePattern, SparsePattern

export AbstractComponent
export AbstractExplicitComponent, ExplicitComponent, ExplicitSystem
export AbstractImplicitComponent, ImplicitComponent, ImplicitSystem

export AbstractSolver, Newton

export outputs, outputs!, outputs!!, outputs!!!
export jacobian, jacobian!, jacobian!!, jacobian!!!
export outputs_and_jacobian, outputs_and_jacobian!, outputs_and_jacobian!!, outputs_and_jacobian!!!
export residuals, residuals!, residuals!!, residuals!!!
export residual_input_jacobian, residual_input_jacobian!, residual_input_jacobian!!, residual_input_jacobian!!!
export residual_output_jacobian, residual_output_jacobian!, residual_output_jacobian!!, residual_output_jacobian!!!
export residuals_and_input_jacobian, residuals_and_input_jacobian!, residuals_and_input_jacobian!!, residuals_and_input_jacobian!!!
export residuals_and_output_jacobian, residuals_and_output_jacobian!, residuals_and_output_jacobian!!, residuals_and_output_jacobian!!!
export residuals_and_jacobians, residuals_and_jacobians!, residuals_and_jacobians!!, residuals_and_jacobians!!!

# functions for computing derivatives
include("derivatives.jl")

# definition of components and calling functions
include("components.jl")

# functions for finding jacobian matrix sparsity
include("sparsity.jl")

# constructors for explicit and implicit components
include("constructors.jl")

# collection of built-in solvers for converting implicit components to explicit components
include("solvers.jl")

end
