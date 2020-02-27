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
