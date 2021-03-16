# CoupledSystems

[![](https://img.shields.io/badge/docs-stable-blue.svg)](https://flow.byu.edu/CoupledSystems.jl/stable)
[![](https://img.shields.io/badge/docs-dev-blue.svg)](https://flow.byu.edu/CoupledSystems.jl/dev)
![](https://github.com/byuflowlab/CoupledSystems.jl/workflows/Run%20tests/badge.svg)

*Easily model and obtain analytic derivatives of arbitrarily complex coupled systems*

Author: Taylor McDonnell

**CoupledSystems** is a modeling framework for easily modeling and obtaining exact derivatives of arbitrarily complex coupled systems.  It is similar in nature to OpenMDAO, but relies on a different theoretical foundation for propagating derivatives (and is written in Julia!).  It is also designed to automatically interface with a variety of packages in the Julia ecosystem to make obtaining derivatives and solving nonlinear systems of equations relatively painless.

## Package Features
 - Can be used to:
    - Perform mixed-mode automatic differentiation
    - Construct objective and constraint functions (with exact derivatives) for gradient-based optimization.
    - Easily construct monolithic coupled systems of equations from an arbitrary number of systems of equations.
    - Perform efficient sensitivity analyses and obtain exact derivatives (to machine precision)
 - Automatically calculates partial derivatives using:
    - Finite Differencing (Implemented using [FiniteDiff](https://github.com/JuliaDiff/FiniteDiff.jl))
      - Forward
      - Central
      - Complex Step
    - Automatic Differentiation
      - Forward Mode (Implemented using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl))
      - Reverse Mode (Implemented using [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl))
    - User-defined analytic calculations
 - Automatically calculates total derivatives using the
    - Chain Rule (for explicit systems)
      - Forward
      - Reverse
    - Analytic Sensitivity Equations (for implicit systems)
      - Direct
      - Adjoint
 - Multiple solvers available for solving implicit systems
    - Trust Region (Implemented using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl))
    - Newton's Method with Line Search (Implemented using [NLsolve](https://github.com/JuliaNLSolvers/NLsolve.jl))
    - User-defined solvers
 - Efficient and convenient calling interface
    - Output only, derivative only, and/or combined output and derivative calculation functions
    - Non-allocating interface to reduce/eliminate run-time allocations
    - Allocating interface for derivative verification using [ForwardDiff](https://github.com/JuliaDiff/ForwardDiff.jl)
    - Only calculates new values/derivatives for each component/subcomponent if inputs have been updated (unless otherwise specified)

## Installation

Enter the package manager by typing `]` and then run the following:

```julia
pkg> add CoupledSystems
```

## Usage

See the [documentation](https://flow.byu.edu/CoupledSystems.jl/dev)

## References
<a id="1">[1]</a>
Martins, Joaquim R. R. A., and Ning, Andrew, Engineering Design Optimization, Cambridge University Press, 2021.
