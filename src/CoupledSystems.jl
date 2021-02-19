module CoupledSystems

export CoupledSystem

"""
    CoupledSystem{TF, TDFX, TDFU, TG, TDGX, TDGU}

Represents a system constructed out of multiple systems coupled together

# Fields:
 - `fg!`: functions which compute residuals and outputs for each system
 - `dfg!`: functions which compute partials for each system
 - `fgdfg!`: functions which compute residuals, outputs, and partials for each system
 - `F::TF`: cache for residuals for each system
 - `DFX::TDFX`: cache for residual partials wrt state variables for each system
 - `DFU::TDFU`: cache for residual partials wrt input variables for each system
 - `G::TG`: cache for outputs for each system
 - `DGX::TDGX`: cache for output partials wrt state variables for each system
 - `DGU::TDGU`: cache for output partials wrt input variables for each system
 - `mapping_system::Vector{Vector{Int}}`: output system for each input
 - `mapping_index::Vector{Vector{Int}}`: output index for each input
"""
struct CoupledSystem{TF, TDFX, TDFU, TG, TDGX, TDGU}
    fg!
    dfg!
    fgdfg!
    F::TF
    DFX::TDFX
    DFU::TDFU
    G::TG
    DGX::TDGX
    DGU::TDGU
    mapping_system::Vector{Vector{Int}}
    mapping_index::Vector{Vector{Int}}
end

"""
    CoupledSystem(f0, g0, dfdx0, dfdu0, dgdx0, dgdu0, mapping_system,
        mapping_index; kwargs...)

Construct a coupled system of equations.

Arguments `f0`, `g0`, `dfdx0`, `dfdu0`, `dgdx0`, and `dgdu0` are used solely to
provide size and type information for the outputs from each system's equations.
`mapping_system` and `mapping_index` are used to describe the output-input
correspondance for each system.  The residual and output equations for each
system must be provided to this constructor as keyword arguments.  It is also
recommended that the partial derivatives of the residual and output equations
with respect to each system's state and output variables are provided as keyword
arguments, though these functions can be approximated if necessary.

# Arguments
 - `f0`: (required) Collection of vectors that provide residual output size and
    type for each system
 - `g0`: (required) Collection of vectors that provide output value size and
    type for each system
 - `dfdx0`: Collection of matrices that provide the size and type of the
    jacobian matrices that hold the partial derivative of each system's residual
    equations with respect to its state variables.
 - `dfdu0`: Collection of matrices that provide the size and type of the
    jacobian matrices that hold the partial derivative of each system's residual
    equations with respect to its input variables.
 - `dgdx0`: Collection of matrices that provide the size and type of the
    jacobian matrices that hold the partial derivative of each system's output
    equations with respect to its state variables.
 - `dfdu0`: Collection of matrices that provide the size and type of the
    jacobian matrices that hold the partial derivative of each system's output
    equations with respect to its input variables.
 - `mapping_system`: Collection of vectors that indicate which system each input
    is taken from.
 - `mapping_index`: Collection of vectors that indicate (together with
    `mapping_system`) which system output each input corresponds to.

# Keyword Arguments
 - `f!`: Functions which compute the residuals of each system
 - `g!`: Functions which compute the outputs of each system
 - `dfdx!`: Functions which compute the derivative of the residuals with respect
    to the state variables for each system
 - `dfdu!`: Functions which compute the derivative of the residuals with respect
    to the input variables for each system
 - `dgdx!`: Functions which compute the derivative of the outputs with respect
    to the state variables for each system
 - `dgdu!`: Functions which compute the derivative of the outputs with respect
    to the input variables for each system
 - `fg!`: Functions which computes the residuals and outputs simultaneously for
    each system
 - `dfg!`: Functions which compute all partials simultaneously for each system
 - `fgdfg!`: Functions which compute the residuals, outputs, and partials
    simultaneously for each system
"""
function CoupledSystem(f0, g0, dfdx0, dfdu0, dgdx0, dgdu0, mapping_system, mapping_index;
    f! = nothing, g! = nothing,
    dfdx! = nothing, dfdu! = nothing, dgdx! = nothing, dgdu! = nothing,
    fg! = nothing, dfg! = nothing,
    fgdfg! = nothing)

    # ensure function arguments correspond to consistent number of systems
    @assert length(f0) == length(g0) == length(dfdx0) == length(dfdu0) ==
        length(dgdx0) == length(dgdu0) == length(mapping_system) ==
        length(mapping_index)

    # ensure `f` is defined
    @assert any((!isnothing(f!), !isnothing(fg!), !isnothing(fgdfg!))) "Residual function not defined"

    # ensure `g` is defined
    @assert any((!isnothing(g!), !isnothing(fg!), !isnothing(fgdfg!))) "Output function not defined"

    # number of systems
    nsys = length(f0)

    # initialize residual function storage
    F = deepcopy(f0)
    DFX = deepcopy(dfdx0)
    DFU = deepcopy(dfdu0)

    # initialize output function storage
    G = deepcopy(g0)
    DGX = deepcopy(dgdx0)
    DGU = deepcopy(dgdu0)

    # define fg! if undefined
    if isnothing(fg!)
        if !isnothing(f!) && !isnothing(g!)
            # construct fg! from f! and g!
            fg! = Vector{Function}(undef, nsys)
            for i = 1:nsys
                fi!, gi! = f![i], g![i]
                fg![i] = function(F, G, x, u)
                    fi!(F, x, u)
                    gi!(G, x, u)
                    return nothing
                end
            end
        else
            # call fgdfg! in place of fg!
            fg! = Vector{Function}(undef, nsys)
            for i = 1:nsys
                fgdfgi! = fgdfg![i]
                DFXi, DFUi, DGXi, DGUi = DFX[i], DFU[i], DGX[i], DGU[i]
                fg![i] = function(F, G, x, u)
                    fgdfgi!(F, G, DFXi, DFUi, DGXi, DGUi, x, u)
                    return nothing
                end
            end
        end
    end

    # define dfg! if undefined
    if isnothing(dfg!)
        if !isnothing(dfdx!) && !isnothing(dfdu!) && !isnothing(dgdx!) && !isnothing(dgdu!)
            # construct dfg! from dfdx!, dfdu!, dgdx!, and dgdu!
            dfg! = Vector{Function}(undef, nsys)
            for i = 1:nsys
                dfdxi!, dfdui! = dfdx![i], dfdu![i]
                dgdxi!, dgdui! = dgdx![i], dgdu![i]
                dfg![i] = function(DFX, DFU, DGX, DGU, x, u)
                    dfdxi!(DFX, x, u)
                    dfdui!(DFU, x, u)
                    dgdxi!(DGX, x, u)
                    dgdui!(DGU, x, u)
                    return nothing
                end
            end
        elseif !isnothing(fgdfg!)
            # call dfgdxu! in place of dfg!
            dfg! = Vector{Function}(undef, nsys)
            for i = 1:nsys
                fgdfgi! = fgdfg![i]
                Fi, Gi = F[i], G[i]
                dfg! = function(DFX, DFU, DGX, DGU, x, u)
                    fgdfgi!(Fi, Gi, DFX, DFU, DGX, DGU, x, u)
                    return nothing
                end
            end
        else
            # calculate partials of each function (ForwardDiff or FiniteDifference)
            error("Derivative approximations not available yet")
        end
    end

    # define fgdfg! if undefined
    if isnothing(fgdfg!)
        # construct fgdfg! from fg! and dfg!
        fgdfg! = Vector{Function}(undef, nsys)
        for i = 1:nsys
            fgi! = fg![i]
            dfgi! = dfg![i]
            fgdfg![i] = function(F, G, DFX, DFU, DGX, DGU, x, u)
                fgi!(F, G, x, u)
                dfgi!(DFX, DFU, DGX, DGU, x, u)
                return nothing
            end
        end
    end

    return CoupledSystem(fg!, dfg!, fgdfg!, F, DFX, DFU, G, DGX, DGU,
        mapping_system, mapping_index)
end

function jacobian!(system::CoupledSystem, x, u)

    return LinearMap()
end

end
