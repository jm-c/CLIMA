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
