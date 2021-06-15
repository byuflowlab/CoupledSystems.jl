#################################
# --- Differentiation Types --- #
#################################

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
    finitediff_type(mode::AbstractFD)

Converts the types used by this package to the inputs expected by the FiniteDiff
package.
"""
finitediff_type
finitediff_type(::ForwardFD) = Val(:forward)
finitediff_type(::CentralFD) = Val(:central)
finitediff_type(::ComplexFD) = Val(:complex)

############################
# --- Chain Rule Types --- #
############################

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

####################################################
# --- Analytic Sensitivity Equation Mode Types --- #
####################################################

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
