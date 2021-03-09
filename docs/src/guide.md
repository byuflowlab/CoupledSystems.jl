# Getting Started: The Sellar Problem

The Sellar problem is is a two-discipline toy problem introduced by R.S. Sellar, S.M. Batill, and J.E. Renaud in "Response Surface Based, Concurrent Subspace Optimization for Multidisciplinary System Design".  The problem itself is non-physical, but it provides a simple example with which to demonstrate multidisciplinary coupling and analysis capabilities.

Each of the two disciplines in the Sellar problem consists of a single explicit equation.  The output of each discipline is used as an input to the other discipline.  Together, the two disciplines form a nonlinear system of equations which must be solved in order to obtain valid outputs.  The outputs of the coupled system of equations are used to construct objective and constraint functions for an optimization. The following is an XDSM diagram of the problem structure.

![](sellar-xdsm1.svg)

Our goal is to manipulate this problem in order to create a single explicit function/component which defines the objective and constraint values as well as their derivatives with respect to the design variables (the inputs).  In other words, we would like to take the diagram shown above and convert it to the following one-component system, and propagate derivatives while doing so.

![](sellar-xdsm-final.svg)

## Constructing Explicit Components

Let's start by loading the package, and then constructing each of the components of the Sellar problem individually.  Since all of the components (when considered in isolation) are explicit, we model them as such.

```@example guide
using CoupledSystems

# --- Define Discipline 1 --- #

# vector valued input-output function
function f_d1(outputs, inputs)
    z1 = inputs[1]
    z2 = inputs[2]
    x = inputs[3]
    y2 = inputs[4]
    outputs[1] = z1^2 + z2 + x - 0.2*y2
end

# define arrays that define size and type of inputs and outputs
inputs_d1 = zeros(4)
outputs_d1 = zeros(1)

# construct explicit component for discipline 1
d1 = ExplicitComponent(inputs_d1, outputs_d1; f=f_d1, deriv=ForwardFD())

# --- Define Discipline 2 --- #

# vector valued input-output function
function f_d2(outputs, inputs)
    z1 = inputs[1]
    z2 = inputs[2]
    y1 = inputs[3]
    outputs[1] = sqrt(y1) + z1 + z2
end

# define arrays that define size and type of inputs and outputs
inputs_d2 = zeros(3)
outputs_d2 = zeros(1)

# construct explicit component for discipline 2
d2 = ExplicitComponent(inputs_d2, outputs_d2; f=f_d2, deriv=ForwardFD())

# --- Define Objective --- #

function f_obj(outputs, inputs)
    z1 = inputs[1]
    x = inputs[2]
    y1 = inputs[3]
    y2 = inputs[4]
    outputs[1] = x^2 + z1 + y1 + exp(-y2) # objective
end

# define arrays that define size and type of inputs and outputs
inputs_obj = zeros(4)
outputs_obj = zeros(1)

# construct explicit component for objective
obj = ExplicitComponent(inputs_obj, outputs_obj; f=f_obj, deriv=ForwardFD())

# --- Define Constraint 1 --- #

function f_c1(outputs, inputs)
    y1 = inputs[1]
    outputs[1] = 3.16 - y1
end

# define arrays that define size and type of inputs and outputs
inputs_c1 = zeros(1)
outputs_c1 = zeros(1)

# construct explicit component for constraint 1
c1 = ExplicitComponent(inputs_c1, outputs_c1; f=f_c1, deriv=ForwardFD())

# --- Define Constraint 2 --- #

function f_c2(outputs, inputs)
    y2 = inputs[1]
    outputs[1] = y2 - 24.0
end

# define arrays that define size and type of inputs and outputs
inputs_c2 = zeros(1)
outputs_c2 = zeros(1)

# construct explicit component for constraint 2
c2 = ExplicitComponent(inputs_c2, outputs_c2; f=f_c2, deriv=ForwardFD())

nothing #hide
```

Note that rather than providing the jacobians of each function directly, we use forward finite differencing to estimate the jacobian of each component. This is the default behavior of CoupledSystems.  Other provided jacobian calculation methods include [`CentralFD()`](@ref), [`ComplexFD()`](@ref), [`ForwardAD()`](@ref), and [`ReverseAD()`](@ref).  It's also possible to provide your own jacobian function through the `df` keyword argument.

## Converting Explicit Components into Implicit Components

We would like to be able to converge all interdependent components simultaneously.  This can be done with the help of a nonlinear solver.  Let's go ahead and add one to the XDSM diagram.

![](sellar-xdsm2.svg)

Notice that the two disciplinary components have been converted to implicit components so that they can provide residual vectors to the nonlinear solver.  The nonlinear solver then drives these residuals to zero and provides the final states/outputs `y_1` and `y_2` to the rest of the system.  Let's go ahead and convert our explicit components to implicit components so that they can be used to provide residual vectors (and associated derivatives) to the nonlinear solver.

```@example guide
# convert discipline 1 into an implicit component
d1i = ImplicitComponent(d1)

# convert discipline 2 into an implicit component
d2i = ImplicitComponent(d2)

nothing #hide
```

## Combining Implicit Components into an Implicit System

At this point we combine the two implicit components to create a single set of residual equations.  The state variables (which are also the system outputs) are also combined.

```@example guide

# define input mapping for discipline 1
mapping_1 = [
    (0, 1), # first input (z1) is index 1 of the system inputs
    (0, 2), # second input (z2) is index 2 of the system inputs
    (0, 3), # third input (x) is index 3 of the system inputs
    (2, 1), # fourth input (y2) is the first output from discipline 2
]

# define input mapping for discipline 2
mapping_2 = [
    (0, 1), # first input (z1) is index 1 of the system inputs
    (0, 2), # second input (z2) is index 2 of the system inputs
    (1, 1), # third input is the first output from the first component
]

# array that defines size and types of the inputs
inputs_isys = zeros(3)

# components in the implicit system
components_isys = [d1i, d2i]

# input mapping for components in the implicit system
component_mapping_isys = [mapping_1, mapping_2]

# implicit system construction
isys = ImplicitSystem(inputs_isys, components_isys, component_mapping_isys)

nothing #hide
```

Let's now update the XDSM diagram to account for the newly combined system.

![](sellar-xdsm3.svg)

## Coupling an Implicit Component with a Solver to Construct an Explicit Component

Now that we have a single implicit system of equations, we can couple it with the nonlinear solver, which will effectively convert it into an explicit component.  The outputs of the resulting explicit component will be equal to the outputs of each of the implicit system's subcomponents, concatenated.

```@example guide
mda = ExplicitComponent(isys, solver=Newton())
nothing #hide
```

Here is the XDSM diagram of the Sellar problem after combining the implicit system of equations with the nonlinear solver to create an explicit component.

![](sellar-xdsm4.svg)

## Combining Explicit Components into an Explicit System

Since there are no longer any model interdependencies, each component may be called sequentially to generate the final output.  We can express this sequence of explicit components as a single explicit system with its own set of inputs and outputs

```@example guide

# define input mapping for the multidisciplinary analysis
mapping_mda = [
    (0, 1), # first input (z1) is index 1 of the system inputs
    (0, 2), # second input (z2) is index 2 of the system inputs
    (0, 3), # third input (x) is index 3 of the system inputs
]

# define input mapping for the objective
mapping_obj = [
    (0, 1), # first input (z1) is index 1 of the system inputs
    (0, 3), # second input (x) is index 3 of the system inputs
    (1, 1), # third input (y1) is the first output from the multidisciplinary analysis
    (1, 2), # fourth input (y2) is the second output from the multidisciplinary analysis
]

# define input mapping for the first constraint
mapping_g1 = [
    (1, 1), # first input (y1) is the first output from the multidisciplinary analysis
]

# define input mapping for the second constraint
mapping_g2 = [
    (1, 2), # first input (y2) is the second output from the multidisciplinary analysis
]

# array defining size and type of inputs to the explicit system
inputs_sellar = zeros(3)

# components in the explicit system
components_sellar = [mda, obj, c1, c2]

# input mapping for each of the components
component_mapping_sellar = [mapping_mda, mapping_obj, mapping_g1, mapping_g2]

# array defining size and type of outputs from the explicit system
outputs_sellar = zeros(3)

# output mapping for the combined system
output_mapping_sellar = [
    (2, 1), # first output is the first output from the second component
    (3, 1), # second output is the first output from the third component
    (4, 1), # third output is the first output from the fourth component
]

sellar = ExplicitSystem(inputs_sellar, outputs_sellar, components_sellar,
    component_mapping_sellar, output_mapping_sellar; mode=Reverse())

nothing #hide
```

At this point we have achieved our goal of representing the Sellar problem as an explicit one-component system.

![](sellar-xdsm-final.svg)

## Querying Explicit Components and/or Systems

Now that our entire system has been reduced down into a single explicit component, we can easily obtain the outputs and their derivatives with respect to the design variables for any set of design variables.

```@example guide
# inputs to the Sellar problem
x = rand(3)

# outputs from the Sellar problem
y = outputs!(sellar, x)

# jacobian of the outputs with respect to the inputs
dydx = jacobian!(sellar, x)

# combined evaluation of outputs and jacobian
y, dydx = outputs_and_jacobian!(sellar, x)

nothing #hide
```

## Verifying Derivatives

Derivatives can be verified easily using finite differencing.

```@example guide
using FiniteDiff

# Verify using forward finite differencing
f = (x) -> outputs(sellar, x)
dydx_fd = FiniteDiff.finite_difference_jacobian(f, x)
println("Maximum Error: ", maximum(abs.(dydx - dydx_fd)))
nothing #hide
```

For an even better derivative check, the jacobians can be verified against exact derivatives computed using forward mode automatic differentiation.

```@example guide
using ForwardDiff

# Verify using forward mode automatic differentiation
dydx_ad = ForwardDiff.jacobian(f, x)
error = dydx - dydx_ad
println("Maximum Error: ", maximum(abs.(dydx - dydx_ad)))
nothing #hide
```

Since this package uses analytic expressions to propagate derivatives, the derivatives computed by this package combined with finite differencing are actually more accurate than those computed using finite differencing alone.

```@example guide
error = dydx - dydx_ad
error_fd = dydx_fd - dydx_ad
println("Maximum Error using Finite Differencing: ",
    maximum(abs.(dydx_fd - dydx_ad)))
println("Maximum Error using Finite Differencing and CoupledSystems: ",
    maximum(abs.(dydx - dydx_ad)))
nothing #hide
```

## Final Notes

While the intended use of CoupledSystems is typically to reduce an arbitrarily complex system down to a single explicit component, there are multiple ways in which this may be accomplished.  For example, an alternative approach to modeling the Sellar problem would be to construct a single implicit system from all components and then couple that system with a nonlinear solver (which is effectively the approach used in OpenMDAO).  

Additionally, there are multiple ways in which any given problem may be divided up into components.  For example, we could have defined the objective and constraint functions as a single vector-valued function rather than three.  Ultimately, while these choices may seem arbitrary, different combinations of choices may be more computationally efficient than others, especially for derivative computations.
