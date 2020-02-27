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
