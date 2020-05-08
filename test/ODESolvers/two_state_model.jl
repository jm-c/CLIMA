using StaticArrays
using CLIMA.VariableTemplates
using CLIMA.DGmethods.NumericalFluxes:
    NumericalFluxNonDiffusive, NumericalFluxDiffusive, NumericalFluxGradient

import CLIMA.DGmethods:
    BalanceLaw,
    vars_state,
    init_state!,
    vars_aux,
    init_aux!,
    LocalGeometry,
    vars_diffusive,
    vars_gradient,
    flux_nondiffusive!,
    flux_diffusive!,
    source!,
    boundary_state!,
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

struct FastODE{T} <: BalanceLaw
    ω::T
end
struct SlowODE{T} <: BalanceLaw
    ω::T
end
struct Alt_ODE{M} <: BalanceLaw
    slowM::M
    function Alt_ODE(slowM::M) where {M}
        return new{M}(slowM)
    end
end

FT = Float64
vars_state(::FastODE, FT) = @vars(U::FT)
vars_state(::SlowODE, FT) = @vars(u::FT)
vars_state(m::Alt_ODE, FT) = vars_state(m.slowM, FT)

function init_state!(m::FastODE, Q::Vars, A::Vars, coords, t::Real)
    Q.U = -0

    return nothing
end

function init_state!(m::SlowODE, Q::Vars, A::Vars, coords, t::Real)
    Q.u = -0

    return nothing
end

vars_aux(::FastODE, FT) = @vars(Uc::FT)
vars_aux(::SlowODE, FT) = @vars(delta_u::FT)
vars_aux(::Alt_ODE, FT) = @vars()

function init_aux!(m::FastODE, A::Vars, geom::LocalGeometry)
    A.Uc = 0
    return nothing
end

function init_aux!(m::SlowODE, A::Vars, geom::LocalGeometry)
    A.delta_u = 0
    return nothing
end

function init_aux!(m::Alt_ODE, A::Vars, geom::LocalGeometry)
    return nothing
end

vars_diffusive(::FastODE, FT) = @vars()
vars_diffusive(::SlowODE, FT) = @vars()
vars_diffusive(::Alt_ODE, FT) = @vars()
vars_gradient(::FastODE, FT) = @vars()
vars_gradient(::SlowODE, FT) = @vars()
vars_gradient(::Alt_ODE, FT) = @vars()

flux_nondiffusive!(::FastODE, _...) = nothing
flux_nondiffusive!(::SlowODE, _...) = nothing
flux_nondiffusive!(::Alt_ODE, _...) = nothing
flux_diffusive!(::FastODE, _...) = nothing
flux_diffusive!(::SlowODE, _...) = nothing
flux_diffusive!(::Alt_ODE, _...) = nothing

@inline function source!(m::FastODE, S, Q, D, A, t)
    @inbounds begin
        U = Q.U

        S.U += FT(1)

        return nothing
    end
end

@inline function source!(m::SlowODE, S, Q, D, A, t)
    @inbounds begin
        u = Q.u

        S.u += FT(1)

        return nothing
    end
end

@inline function source!(m::Alt_ODE, S, Q, D, A, t)
    @inbounds begin
      # u = Q.u
      # S.u += FT(1)

        return nothing
    end
end

function boundary_state!(nf, m::FastODE, _...)
    return nothing
end

function boundary_state!(nf, m::SlowODE, _...)
    return nothing
end

function boundary_state!(nf, m::Alt_ODE, _...)
    return nothing
end

@inline function initialize_fast_state!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
)
    dgFast.auxstate.Uc .= -0

    return nothing
end

@inline function pass_tendency_from_slow_to_fast!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qfast,
    dQslow, # probably not needed here, need Qslow instead
)
    # not sure if this is needed at all
    return nothing
end

@inline function cummulate_fast_solution!(
    fast::FastODE,
    dgFast,
    Qfast,
    fast_time,
    fast_dt,
    total_fast_step,
)
    dgFast.auxstate.Uc .+= Qfast.U

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::SlowODE,
    fast::FastODE,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    total_fast_step,
)

#@info @sprintf """ total_fast_step = %3i""" total_fast_step
    dgFast.auxstate.Uc .= dgFast.auxstate.Uc / total_fast_step
    dgSlow.auxstate.delta_u .= dgFast.auxstate.Uc - Qslow.u

    return nothing
end
