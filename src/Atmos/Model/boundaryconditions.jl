using CLIMA.PlanetParameters
export BoundaryCondition, InitStateBC


using CLIMA.PlanetParameters
export InitStateBC, DYCOMS_BC, RayleighBenardBC

export AtmosBC,
  Impenetrable, FreeSlip, NoSlip, DragLaw,
  Insulating, PrescribedTemperature, ConstEnergyFlux,
  Impermeable, ConstMoistureFlux

"""
    AtmosBC(momentum = Impenetrable(FreeSlip())
            energy   = Insulating()
            moisture = Impermeable())

The standard boundary condition for [`AtmosModel`](@ref). The default implies a "no flux" boundary condition.
"""
Base.@kwdef struct AtmosBC{M,E,Q}
  momentum::M = Impenetrable(FreeSlip())
  energy::E = Insulating()
  moisture::Q = Impermeable()
end


function boundary_state!(nf, atmos::AtmosModel, args...)
  return atmos_boundary_state!(nf, atmos.boundarycondition, atmos, args...)
end

function atmos_boundary_state!(nf, tup::Tuple, atmos, state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  # TODO figure out a better way to unroll tuple loop
  if bctype == 1
    return atmos_boundary_state!(nf, tup[1], atmos, state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  else if bctype == 2
    return atmos_boundary_state!(nf, tup[2], atmos, state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  end
end

function atmos_boundary_state!(nf, bc::AtmosBC, atmos, args...)
  atmos_boundary_state!(nf, bc.momentum, atmos, args...)
  atmos_boundary_state!(nf, bc.energy,   atmos, args...)
  atmos_boundary_state!(nf, bc.moisture, atmos, args...)

  return nothing
end


function normal_boundary_flux_diffusive!(nf, atmos::AtmosModel, fluxᵀn::Vars{S},
                                         n⁻, state⁻, diff⁻, aux⁻,
                                         state⁺, diff⁺, aux⁺,
                                         bctype::Integer, t, args...) where {S}
  return atmos_normal_boundary_flux_diffusive!(nf, atmos.boundarycondition, atmos, fluxᵀn,
                                               n⁻, state⁻, diff⁻, aux⁻,
                                               state⁺, diff⁺, aux⁺,
                                               bctype, t, args...)
end

function atmos_normal_boundary_flux_diffusive!(nf, tup::Tuple, atmos::AtmosModel,
                                               fluxᵀn, n⁻, state⁻, diff⁻, aux⁻,
                                               state⁺, diff⁺, aux⁺,
                                               bctype, t, args...)
  if bctype == 1
    return atmos_normal_boundary_flux_diffusive!(nf, tup[1], atmos,
                                                 fluxᵀn, n⁻, state⁻, diff⁻, aux⁻,
                                                 state⁺, diff⁺, aux⁺,
                                                 bctype, t, args...)
  else if bctype == 2
    return atmos_normal_boundary_flux_diffusive!(nf, tup[2], atmos,
                                                 fluxᵀn, n⁻, state⁻, diff⁻, aux⁻,
                                                 state⁺, diff⁺, aux⁺,
                                                 bctype, t, args...)
  end
end

function atmos_normal_boundary_flux_diffusive!(nf, bc::AtmosBC, atmos::AtmosModel, args...)
  atmos_normal_boundary_flux_diffusive!(nf, bc.momentum, atmos, args...)
  atmos_normal_boundary_flux_diffusive!(nf, bc.energy,   atmos, args...)
  atmos_normal_boundary_flux_diffusive!(nf, bc.moisture, atmos, args...)

  return nothing
end

abstract type MomentumBC end

"""
    Impenetrable(drag::MomentumDragBC) :: MomentumBC

Defines an impenetrable wall model for momentum.
"""
struct Impenetrable{D} <: MomentumBC
  drag::D
end

function atmos_boundary_state!(nf::NumericalFluxNonDiffusive, bc_momentum::Impenetrable, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  state⁺.ρu -= 2*dot(state⁻.ρu, n) .* SVector(n)

  return atmos_boundary_state!(nf, bc_momentum.drag, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
end

function atmos_boundary_state!(nf::NumericalFluxGradient, bc_momentum::Impenetrable, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  state⁺.ρu -= dot(state⁻.ρu, n) .* SVector(n)

  return atmos_boundary_state!(nf, bc_momentum.drag, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
end

function atmos_normal_boundary_flux_diffusive!(nf, bc_momentum::Impenetrable, atmos, args...)
  return atmos_normal_boundary_flux_diffusive!(nf, bc_momentum.drag, atmos, args...)
end

abstract type MomentumDragBC end

function atmos_boundary_state!(nf, bc_momentum_drag::MomentumDragBC, atmos, args...) end
function atmos_normal_boundary_flux_diffusive!(nf, bc_momentum_drag::MomentumDragBC, atmos, args...) end

"""
    FreeSlip() :: MomentumDragBC

"no drag" model.
"""
struct FreeSlip <: MomentumDragBC end

"""
    NoSlip() :: MomentumDragBC

"""
struct NoSlip <: MomentumDragBC end

function atmos_boundary_state!(nf::NumericalFluxNonDiffusive, bc_momentum::Impenetrable{NoSlip}, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  state⁺.ρu = -state⁻.ρu

  return nothing
end

function atmos_boundary_state!(nf::NumericalFluxGradient, bc_momentum::Impenetrable{NoSlip}, atmos,
                               state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  state⁺.ρu = zero(state⁺.ρu)

  return nothing
end

"""
    DragLaw(C) :: MomentumDragBC

"""
struct DragLaw{FT} <: MomentumDragBC
  C::FT
end

function atmos_normal_boundary_flux_diffusive!(nf, bc_momentum_drag::DragLaw, atmos,
                                               fluxᵀn, n, state⁻, diff⁻, aux⁻,
                                               state⁺, diff⁺, aux⁺,
                                               bctype, t, state1⁻, diff1⁻, aux1⁻)
  u1⁻ = state1⁻.ρu / state1⁻.ρ
  Pu1⁻ = u1⁻ .- dot(u1⁻, n) .* n

  # NOTE: difference from design docs since normal points outwards
  τn = bc_momentum_drag.C * norm(Pu1⁻) * Pu1⁻

  fluxᵀn.ρu += state⁻.ρ   * τn
  fluxᵀn.ρe += state⁻.ρu' * τn

  return nothing
end


abstract type EnergyBC end

function atmos_boundary_state!(nf, bc_energy::EnergyBC, atmos, args...) end
function atmos_normal_boundary_flux_diffusive!(nf, bc_energy::EnergyBC, atmos, args...) end

"""
    Insulating() :: EnergyBC

No energy flux.
"""
struct Insulating <: EnergyBC end

"""
    PrescribedTemperature(T) :: EnergyBC

Fixed boundary temperature `T` (K).
"""
struct PrescribedTemperature{FT} <: EnergyBC
  T::FT
end

function atmos_boundary_state!(nf, bc_energy::PrescribedTemperature, atmos, state⁺, aux⁺, n, state⁻, aux⁻, bctype, t, args...)
  E_int⁺ = state⁺.ρ * cv_d * (bc_energy.T - T_0)
  state⁺.ρe = E_int⁺ + state⁺.ρ * gravitational_potential(atmos.orientation, aux⁻)

  return nothing
end

"""
    ConstEnergyFlux() :: EnergyBC

"""
struct ConstEnergyFlux{FT} <: EnergyBC
  nd_h_tot::FT
end
function atmos_normal_boundary_flux_diffusive!(nf, bc_energy::ConstEnergyFlux, atmos,
                                               fluxᵀn, n⁻, state⁻, diff⁻, aux⁻, state⁺, diff⁺, aux⁺, bctype, t, args...)
  fluxᵀn.ρe += bc_energy.nd_h_tot * state⁻.ρ

  return nothing
end




abstract type MoistureBC end

function atmos_boundary_state!(nf, bc_moisture::MoistureBC, atmos, args...) end
function atmos_normal_boundary_flux_diffusive!(nf, bc_moisture::MoistureBC, atmos, args...) end
"""
    Impermeable() :: MoistureBC

No moisture flux.
"""
struct Impermeable <: MoistureBC end

struct ConstMoistureFlux{FT} <: MoistureBC
  nd_q_tot::FT
end

function atmos_normal_boundary_flux_diffusive!(nf, bc_moisture::ConstMoistureFlux, atmos,
                                               fluxᵀn, n⁻, state⁻, diff⁻, aux⁻, state⁺, diff⁺, aux⁺, bctype, t, args...)
  nρd_q_tot = bc_moisture.nd_q_tot * state⁻.ρ
  fluxᵀn.ρ += nρd_q_tot
  fluxᵀn.ρu += bc_moisture.nd_q_tot .* state⁻.ρu
  # assumes EquilMoist
  fluxᵀn.moisture.ρq_tot += nρd_q_tot

  return nothing
end



"""
    InitStateBC

Set the value at the boundary to match the `init_state!` function. This is
mainly useful for cases where the problem has an explicit solution.

# TODO: This should be fixed later once BCs are figured out (likely want
# different things here?)
"""
struct InitStateBC end

function atmos_boundary_state!(::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                               bc::InitStateBC, m::AtmosModel, state⁺::Vars,
                               aux⁺::Vars, n⁻, state⁻::Vars, aux⁻::Vars, bctype,
                               t, _...)
  return init_state!(m, state⁺, aux⁺, aux⁺.coord, t)
end

function atmos_normal_boundary_flux_diffusive!(nf, bc::InitStateBC, atmos,
                                               fluxᵀn, n⁻, state⁻, diff⁻, aux⁻,
                                               state⁺, diff⁺, aux⁺,
                                               bctype, t, args...)
  return normal_boundary_flux_diffusive!(nf, atmos,
                                         fluxᵀn, n⁻, state⁻, diff⁻, aux⁻,
                                         state⁺, diff⁺, aux⁺,
                                         bc, t, args...)
end

function boundary_state!(::NumericalFluxDiffusive,
                         m::AtmosModel, state⁺::Vars, diff⁺::Vars,
                         aux⁺::Vars, n⁻, state⁻::Vars, diff⁻::Vars,
                         aux⁻::Vars, bc::InitStateBC, t, args...)
  return init_state!(m, state⁺, aux⁺, aux⁺.coord, t)
end
