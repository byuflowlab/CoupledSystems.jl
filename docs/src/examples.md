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
@named x = 1.0
@named y = 0.0
@named z = 0.0
uvar = (x, y, z)

# define parameters (and default values)
@named σ = 10.0
@named ρ = 28.0
@named β = 8/3
pvar = (σ, ρ, β)

# define time variable (and size/type)
@named t = 0.0
tvar = t

# define derivatives of the state variables with respect to time (and size/type)
@named xdot = 0.0
@named ydot = 0.0
@named zdot = 0.0
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
f = ODEFunction(comp, duvar, uvar, pvar, tvar)

# define and solve the ODE problem using DifferentialEquations
u0 = combine(uvar) # set initial state to default values set above
tspan = (0.0, 100.0) # time span
p = combine(pvar) # set parameters to default values set above
prob = ODEProblem(f, u0, tspan, p)
sol = solve(prob)
nothing #hide
```

It may be desirable, however, to incorporate the outputs from the solution of a set of differential equation into part of a larger assembly of components.  In this case, one can simply wrap the solution to the differential equation as an explicit component.  To maximize the accuracy and efficiency of the derivative calculations, make sure to make use of the features provided by the DiffEqSensitivity package (see https://diffeq.sciml.ai/stable/analysis/sensitivity/) and to use automatic differentiation.

```@example lorenz
using DiffEqSensitivity

# define inputs (and size/type)
@named p = zeros(3)

# define outputs (and size/type)
@named xh = zeros(11)
@named yh = zeros(11)
@named zh = zeros(11)

# define inputs and outputs from template function
fin = (p,)
fout = (xh, yh, zh)
finplace = ()

# define template function
func = function(p)
    # define the ODE problem
    prob = ODEProblem(f, u0, tspan, p)

    # solve the ODE
    sol = solve(prob, sensealg=ForwardSensitivity(), saveat = 0.0:10.0:100.0)

    # separate outputs
    xh = view(sol, 1, :)
    yh = view(sol, 2, :)
    zh = view(sol, 3, :)

    return xh, yh, zh
end

xode = ExplicitComponent(func, fin, fout, foutin; deriv=ForwardAD())

nothing #hide
```

The resulting explicit component may be used just like any other explicit component.

```@example lorenz
X = rand(3)

Y = outputs(xode, X)

dYdX = jacobian(xode, X)

Y, dYdX = outputs_and_jacobian(xode, X)

x, y, z = separate(fout, Y)

nothing #hide
```

Alternatively we may want to represent the ODE problem as an implicit system of equations with an associated set of residuals.  This can be helpful when one wishes to converge the residuals of an ODE problem and another set of residuals and/or constraints simultaneously (such as in optimal control problems).  In this case we can model the solution to the ODE as a chain of polynomials which are constrained to satisfy the governing equations at a fixed number of collocation points.

To use this representation of the ODE problem, we first must separate the dynamic parameters (or controls) from the static parameters.  Since all parameters are static in this problem, no changes are necessary.  

```@example lorenz
# create empty tuple representing dynamic parameters
cvar = ()
nothing #hide
```

We also need to discretize time into polynomial segments.  The collocation methods available in Dymos are also available in CoupledSystems (Gauss-Raudau collocation and Gauss-Lobatto collacation).  We will use Gauss-Radau collocation for this problem with eleven evenly spaced segments.  As in Dymos, the scaling of `segment_ends` is irrelevant as it will be automatically scaled to be between 0 and 1.

```@example lorenz
# use Gauss-Radau discretization
segment_ends = range(0, 10, length=11)
nodes_per_segment = 4
disc = GaussRadau(segment_ends, nodes_per_segment)
nothing #hide
```

Finally, we need to specify the initial condition and duration of our simulation.

```@example lorenz
@named x0 = 1.0
@named y0 = 0.0
@named z0 = 0.0
initial_states = (x0, y0, z0)

@named td = 100.0
duration = td

nothing #hide
```

We can now construct our implicit representation of the ODE problem.

```@example lorenz
iode = ImplicitComponent(comp, duvar, uvar, cvar, pvar, tvar, disc, duration, initial_states)
nothing #hide
```

This implicit formulation of the ODE problem can be helpful for solving optimal control problems in a manner similar to that used by [Dymos](https://openmdao.github.io/dymos/) as is demonstrated in [Optimal Model Control: The Brachistochrone Problem](@ref).

## Optimal Model Control: The Brachistochrone Problem
