using CoupledSystems

# --- Define the ODE Function as an Explicit Component --- #

# define the names and the size/type of the state variables
@named y = 0.0
@named v = 0.0
uvar = (y, v)

# define the names and the size/type of the state rates
@named ydot = 0.0
@named vdot = 0.0
duvar = (ydot, vdot)

# define the names and the size/type of the control variables
cvar = ()

# define the names and the size/type of the (static) parameters
@named g = 9.80665
pvar = (g,)

# define the name and the size/type of the time variable
@named t = 0.0
tvar = t

# define the inputs and outputs from the ODE function
fin = (y, v, g) # inputs
fout = (ydot, vdot) # outputs
foutin = () # in-place outputs

# define the ODE function
function func(y, v, g)
    ydot = v
    vdot = -g
    return ydot, vdot
end

# model the ODE function as an explicit component
xcomp = ExplicitComponent(func, fin, fout, foutin)

# --- Define the ODE Problem as an Implicit Component --- #

# define the name, size/type, and default values of the simulation duration
@named td = 5.0
duration = td

# define the names, size/type, and default values of the initial state variables
@named y0 = 100.0
@named v0 = 0.0
initial_states = (y0, v0)

# define the discretization of the problem into polynomial segments
segment_ends = [0.0, 1.0] # use only one segment
nodes_per_segment = 4 # use four nodes per segment
discretization = GaussRadau(segment_ends, nodes_per_segment) # use Gauss-Radau nodes

# define the names and order of the implicit component inputs
component_inputs = (:g, :td, :y0, :v0)

# model the ODE problem as an implicit component
icomp = ImplicitComponent(xcomp, duvar, uvar, cvar, pvar, tvar, discretization,
    duration, initial_states; component_inputs = component_inputs)

# --- Test the Residuals of the Implicit Component --- #

# use the default inputs
X = combine((g, td, y0, v0))

# use the final state variables (after the residuals are converged)
Y = combine(([100.0, 84.54702037909698, 12.483154620903008, -22.583125],
[0.0, -17.409305706967675, -41.43059429303232, -49.03325]))

# calculate the residuals
r = residuals(icomp, X, Y)
