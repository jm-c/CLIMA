using CLIMA.PlanetParameters

export AtmosBC,
       Impenetrable, FreeSlip, NoSlip, DragLaw,
       Insulating, PrescribedTemperature, ConstEnergyFlux,
       Impermeable, ConstMoistureFlux,
       InitStateBC

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

include("momentumBC.jl")
include("energyBC.jl")
include("moistureBC.jl")
include("initstateBC.jl")
