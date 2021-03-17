# Examples

## Modeling Ordinary Differential Equations: The Lorenz Equations

In this example we solve the Lorenz equations:

```math
\begin{aligned}
\frac{dx}{dt} &= σ(y-x) \\
\frac{dy}{dt} &= x(ρ-z) - y \\
\frac{dz}{dt} &= xy - βz \\
\end{aligned}
```

While these equations [can be modeled in DifferentialEquations directly](https://diffeq.sciml.ai/stable/tutorials/ode_example/#Example-2:-Solving-Systems-of-Equations), for purposes of demonstration we will model these equations using CoupledSystems as an [`ExplicitComponent`](@ref) and then convert the explicit component into an [`ODEFunction`](https://diffeq.sciml.ai/stable/features/performance_overloads/#ODEFunction) which may be used with the [`DifferentialEquations`](https://github.com/SciML/DifferentialEquations.jl) package.

```@example lorenz
using CoupledSystems, DifferentialEquations

# define state variables (and default values)
@var x = 1.0
@var y = 0.0
@var z = 0.0
uvar = (x, y, z)

# define parameters (and default values)
@var σ = 10.0
@var ρ = 28.0
@var β = 8/3
pvar = (σ, ρ, β)

# define time variable (and default value)
@var t = 0.0
tvar = t

# define derivatives of the state variables with respect to time
@var xdot = 0.0
@var ydot = 0.0
@var zdot = 0.0
duvar = (xdot, ydot, zdot)

# define inputs and outputs from template function
fin = (x, y, z, σ, ρ, β, t) # inputs
fout = (xdot, ydot, zdot) # outputs
foutin = () # in-place outputs

# construct template function for the ODE
function lorenz(x, y, z, σ, ρ, β, t)
    xdot = σ*(y-x)
    ydot = x*(ρ-z)-y
    zdot = x*y-β*z
    return xdot, ydot, zdot
end

# model the ODE as an explicit component
comp = ExplicitComponent(lorenz, fin, fout, foutin)

# convert the `ExplicitComponent` into an `ODEFunction`
f = ODEFunction(comp, uvar, duvar, pvar, tvar)

# define and solve the ODE problem using DifferentialEquations
u0 = combine(uvar) # set initial state to default values set above
tspan = (0.0, 100.0) # time span
p = combine(pvar) # set parameters to default values set above
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob)
nothing #hide
```

It may be desirable, however, to incorporate the outputs from the solution to a set of differential equation into part of a larger assembly of components.  In this case, one can wrap the differential equation solution as a single explicit component.  The inputs to this component correspond to the initial states, parameters, and time frame over which to solve the ODE.  The outputs from this component correspond to the time history of the selected states and/or the component inputs.  Unless otherwise specified through the keyword argument `saveat`, the returned states correspond to the values of the states at the end of the specified time frame.

```@example lorenz
odecomp = ExplicitComponent(
    comp, # explicit subcomponent which defines the ODE
    uvar, # subcomponent variables corresponding to states
    duvar, # subcomponent variables corresponding to time derivatives of states
    pvar, # subcomponent variables corresponding to parameters
    tvar, # subcomponent variable corresponding to time
    u0, # values or variables corresponding to initial states
    p, # values or variables corresponding to parameters
    tspan, # values or variables corresponding to initial and final time values
    argin, # input variables to this component (passed to `u0`, `p` and `tspan`)
    argout, # output variables from this component (comes from time history of `uvar` or `argin`)
    )
nothing #hide
```

The resulting explicit component may be used just like any other explicit component.

```@example lorenz
X = combine(odein)

Y = outputs(odecomp, X)

dYdX = jacobian(odecomp, X)

Y, dYdX = outputs_and_jacobian(odecomp, X)

x, y, z = separate(odeout, Y)

nothing #hide
```

Alternatively, we can model the solution to the ordinary differential equation as an implicit component, and let an optimizer converge the residuals.  In this case, we must specify the locations in time at which to converge the residuals.

```@example lorenz
odecomp = ImplicitComponent(
    comp, # explicit subcomponent which defines the ODE
    uvar, # subcomponent variables corresponding to states
    duvar, # subcomponent variables corresponding to time derivatives of states
    pvar, # subcomponent variables corresponding to parameters
    tvar, # subcomponent variable corresponding to time
    u0, # values or variables corresponding to initial states
    p, # values or variables corresponding to parameters
    tspan, # values or variables corresponding to initial and final time values
    argin, # input variables to this component (passed to `u0`, `p` and `tspan`)
    argout, # output variables from this component (comes from time history of `uvar` or `argin`)
    )
nothing #hide
```

This implicit formulation of the ODE problem can be helpful for solving optimal control problems in a manner similar to that used by [Dymos](https://openmdao.github.io/dymos/).
