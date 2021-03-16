# Library

```@contents
Pages = ["library.md"]
Depth = 3
```

## Public API

### System Variables
```@docs
@var
NamedVar
name
value
combine
combine!
separate
```

### Differentiation Methods

```@docs
AbstractDiffMethod
AbstractAD
ForwardAD
ReverseAD
AbstractFD
ForwardFD
CentralFD
ComplexFD
Forward
Reverse
Direct
Adjoint
```

### Sparsity

```@docs
AbstractSparsityPattern
DensePattern
SparsePattern
```

### Constructors

```@docs
AbstractComponent
AbstractExplicitComponent
ExplicitComponent
ExplicitSystem
AbstractImplicitComponent
ImplicitComponent
ImplicitSystem
```

### Solvers

```@docs
AbstractSolver
Newton
```

### Calling Interface

```@docs
outputs
outputs!
outputs!!
outputs!!!
jacobian
jacobian!
jacobian!!
jacobian!!!
outputs_and_jacobian
outputs_and_jacobian!
outputs_and_jacobian!!
outputs_and_jacobian!!!
residuals
residuals!
residuals!!
residuals!!!
residual_input_jacobian
residual_input_jacobian!
residual_input_jacobian!!
residual_input_jacobian!!!
residual_output_jacobian
residual_output_jacobian!
residual_output_jacobian!!
residual_output_jacobian!!!
residuals_and_input_jacobian
residuals_and_input_jacobian!
residuals_and_input_jacobian!!
residuals_and_input_jacobian!!!
residuals_and_output_jacobian
residuals_and_output_jacobian!
residuals_and_output_jacobian!!
residuals_and_output_jacobian!!!
residuals_and_jacobians
residuals_and_jacobians!
residuals_and_jacobians!!
residuals_and_jacobians!!!
```

### Convenience Functions
```
combine
separate

```

## Private API

```@autodocs
Modules = [CoupledSystems]
Public = false
``

## Index

```@index
```
