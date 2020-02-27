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
