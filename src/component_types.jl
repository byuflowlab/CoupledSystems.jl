"""
    AbstractComponent

Supertype for all components
"""
abstract type AbstractComponent end

"""
    AbstractExplicitComponent <: AbstractComponent

Supertype for components defined by the vector valued output function `y = f(x)`
"""
abstract type AbstractExplicitComponent <: AbstractComponent end

"""
    AbstractImplicitComponent <: AbstractComponent

Supertype for components defined by the vector valued residual function `0 = f(x, y)`
"""
abstract type AbstractImplicitComponent <: AbstractComponent end

"""
    ExplicitComponent{TX, TY, TJ, TI, TO} <: AbstractExplicitComponent

System component defined by the explicit vector-valued output function: `y = f(x)`

# Fields
 - `f`: In-place output function `f(y, x)`
 - `df`: In-place jacobian function `df(dydx, x)`
 - `fdf`: In-place combined output and jacobian function `fdf(y, dydx, x)`
 - `x_f::TX`: Inputs used to evaluate the outputs
 - `x_df::TX`: Inputs used to evaluate the jacobian
 - `y::TY`: Outputs
 - `dydx::TJ`: Jacobian
 - `argin::TI`: Tuple of named variables corresponding to component inputs
 - `argout::TO`: Tuple of named variables corresponding to component outputs
"""
struct ExplicitComponent{TX, TY, TJ, TI, TO} <: AbstractExplicitComponent
    f
    df
    fdf
    x_f::TX
    x_df::TX
    y::TY
    dydx::TJ
    argin::TI
    argout::TO
end

"""
    ExplicitSystem{TC, TX, TY, TJ, TD} <: AbstractExplicitComponent

Explicit system constructed from a chain of explicit system components, called
sequentially.

# Fields
 - `components::TC`: Collection of explicit components, in calling order
 - `input_mapping::Vector{NTuple{2,Vector{Int}}}`: Mapping from system inputs to
    component inputs and/or system outputs.
 - `component_output_mapping::Vector{Vector{NTuple{2,Vector{Int}}}}`: Mapping from
    component outputs to component inputs and/or system outputs
 - `component_input_mapping::Vector{Vector{NTuple{2,Int}}}`: Mapping to component
    inputs from system inputs and/or component outputs
 - `output_mapping::Vector{NTuple{2,Int}}`: Mapping to system outputs from system
    inputs and/or component outputs
 - `x_f::TX`: Inputs used to evaluate the system outputs
 - `x_df::TX`: Inputs used to evaluate the system jacobian
 - `y::TY`: Storage for the system outputs
 - `dydx::TJ`: Storage for the system jacobian
 - `argin::TI`: Tuple of named inputs to output function, defines size, type, and default values of inputs
 - `argout::TO`: Tuple of named outputs from output function, defines size and type of outputs
 - `mode::TD`: Default direction in which to apply the chain rule (`Forward()` or `Reverse()`)
"""
struct ExplicitSystem{TC, TX, TY, TJ, TI, TO, TD} <: AbstractExplicitComponent
    components::TC
    input_mapping::Vector{NTuple{2,Vector{Int}}}
    component_output_mapping::Vector{Vector{NTuple{2,Vector{Int}}}}
    component_input_mapping::Vector{Vector{NTuple{2,Int}}}
    output_mapping::Vector{NTuple{2,Int}}
    x_f::TX
    x_df::TX
    y::TY
    dydx::TJ
    argin::TI
    argout::TO
    mode::TD
end

"""
    ImplicitComponent{TX, TY, TR, TDRX, TDRY, TI, TO} <: AbstractImplicitComponent

System component defined by the vector-valued residual function: `0 = f(x, y)`

# Fields
 - `f`: In-place residual function `f(r, x, y)`.
 - `dfdx`: In-place residual jacobian function with respect to the inputs `dfdx(drdx, x, y)`
 - `dfdy`: In-place residual jacobian function with respect to the outputs `dfdy(drdy, x, y)`
 - `fdfdx`: In-place combined residual and jacobian with respect to the inputs function `fdfdx(r, drdx, x, y)`.
 - `fdfdy`: In-place combined residual and jacobian with respect to the outputs function `fdfdy(r, drdy, x, y)`.
 - `fdf`: In-place combined residual and jacobians function `fdf(r, drdx, drdy, x, y)`.
 - `x_f::TX`: `x` used to evaluate `f`
 - `y_f::TY`: `y` used to evaluate `f`
 - `x_dfdx::TX`: `x` used to evaluate `dfdx`
 - `y_dfdx::TY`: `y` used to evaluate `dfdx`
 - `x_dfdy::TX`: `x` used to evaluate `dfdy`
 - `y_dfdy::TY`: `y` used to evaluate `dfdy`
 - `r::TR`: cache for residual`
 - `drdx::TDRX`: cache for residual jacobian with respect to `x`
 - `drdy::TDRY`: cache for residual jacobian with respect to `y`
 - `argin::TI`: Tuple of named inputs to output function, defines size, type, and default values of inputs
 - `argout::TO`: Tuple of named outputs from output function, defines size, type, and default values of outputs
"""
struct ImplicitComponent{TX, TY, TR, TDRX, TDRY, TI, TO} <: AbstractImplicitComponent
    f
    dfdx
    dfdy
    fdfdx
    fdfdy
    fdf
    x_f::TX
    y_f::TY
    x_dfdx::TX
    y_dfdx::TY
    x_dfdy::TX
    y_dfdy::TY
    r::TR
    drdx::TDRX
    drdy::TDRY
    argin::TI
    argout::TO
end

"""
    ImplicitSystem{TC, TX, TY, TR, TDRX, TDRY, TI, TO} <: AbstractImplicitComponent

Implicit system constructed from interdependent explicit and/or implicit system
components.

# Fields
 - `components::TC`: Collection of components, in preferred calling order
 - `component_input_mapping::Vector{Vector{NTuple{2,Int}}}`:
 - `x_f::TX`: `x` used to evaluate `f`
 - `y_f::TY`: `y` used to evaluate `f`
 - `x_dfdx::TX`: `x` used to evaluate `dfdx`
 - `y_dfdx::TY`: `y` used to evaluate `dfdx`
 - `x_dfdy::TX`: `x` used to evaluate `dfdy`
 - `y_dfdy::TY`: `y` used to evaluate `dfdy`
 - `r::TR`: cache for residual`
 - `drdx::TDRX`: cache for residual jacobian with respect to `x`
 - `drdy::TDRY`: cache for residual jacobian with respect to `y`
 - `argin::TI`: Tuple of named inputs to output function, defines size, type, and default values of inputs
 - `argout::TO`: Tuple of named outputs from output function, defines size, type, and default values of outputs
 - `idx::Vector{Int}`: Index used for accessing outputs/residuals for each component
"""
struct ImplicitSystem{TC, TX, TY, TR, TDRX, TDRY, TI, TO} <: AbstractImplicitComponent
    components::TC
    component_input_mapping::Vector{Vector{NTuple{2,Int}}}
    x_f::TX
    y_f::TY
    x_dfdx::TX
    y_dfdx::TY
    x_dfdy::TX
    y_dfdy::TY
    r::TR
    drdx::TDRX
    drdy::TDRY
    argin::TI
    argout::TO
    idx::Vector{Int}
end

"""
    inputs(component::AbstractComponent)

Return the inputs stored in `component`
"""
inputs(component::AbstractComponent)

"""
    outputs(component::AbstractComponent)

Return the outputs stored in `component`.
"""
outputs(component::AbstractComponent)
