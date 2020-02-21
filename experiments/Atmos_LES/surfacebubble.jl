using Distributions
using Random
using StaticArrays
using Test
using DocStringExtensions
using LinearAlgebra

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: EnergyBC, atmos_normal_boundary_flux_diffusive!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

# -------------------- Surface Driven Bubble ----------------- #
# Rising thermals driven by a prescribed surface heat flux.
# 1) Boundary Conditions:
#       Laterally periodic with no flow penetration through top
#       and bottom wall boundaries.
#       Momentum: Impenetrable(FreeSlip())
#       Energy:   Spatially varying non-zero heat flux up to time t₁
# 2) Domain: 1250m × 1250m × 1000m
# Configuration defaults are in `src/Driver/Configurations.jl`

"""
  SurfaceDrivenBubbleBC <: BoundaryCondition
Y ≡ state vars
Σ ≡ diffusive vars
A ≡ auxiliary vars
X⁺ and X⁻ refer to exterior, interior faces
X₁ refers to the first interior node

# Fields
$(DocStringExtensions.FIELDS)
"""
struct SurfaceDrivenBubbleBC{FT} <: EnergyBC
  "Prescribed MSEF Magnitude `[W/m^2]`"
  F₀::FT
  "Time Cutoff `[s]`"
  t₁::FT
  "Plume wavelength scaling"
  x₀::FT
end

function atmos_normal_boundary_flux_diffusive!(nf, bc_energy::SurfaceDrivenBubbleBC, atmos,
    fluxᵀn, n⁻, state⁻, diff⁻, aux⁻, state⁺, diff⁺, aux⁺, bctype, t, args...)

  if t < bc_energy.t₁
    x = aux⁻.coord[1]
    y = aux⁻.coord[2]
    MSEF = bc_energy.F₀ * (cospi(2*x/bc_energy.x₀))^2 * (cospi(2*y/bc_energy.x₀))^2
    fluxᵀn.ρe += MSEF * state⁻.ρ
  end
end


"""
  Surface Driven Thermal Bubble
"""
function init_surfacebubble!(bl, state, aux, (x,y,z), t)
  FT            = eltype(state)
  R_gas::FT     = R_d
  c_p::FT       = cp_d
  c_v::FT       = cv_d
  γ::FT         = c_p / c_v
  p0::FT        = MSLP

  xc::FT        = 1250
  yc::FT        = 1250
  zc::FT        = 1250
  θ_ref::FT     = 300
  Δθ::FT        = 0

  #Perturbed state:
  θ            = θ_ref + Δθ # potential temperature
  π_exner      = FT(1) - grav / (c_p * θ) * z # exner pressure
  ρ            = p0 / (R_gas * θ) * (π_exner)^ (c_v / R_gas) # density
  P            = p0 * (R_gas * (ρ * θ) / p0) ^(c_p/c_v) # pressure (absolute)
  T            = P / (ρ * R_gas) # temperature
  ρu           = SVector(FT(0),FT(0),FT(0))
  # energy definitions
  e_kin        = FT(0)
  e_pot        = grav * z
  ρe_tot       = ρ * total_energy(e_kin, e_pot, T)
  state.ρ      = ρ
  state.ρu     = ρu
  state.ρe     = ρe_tot
  state.moisture.ρq_tot = FT(0)
end

function config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)

  # Boundary conditions
  # Heat Flux Peak Magnitude
  F₀ = FT(100)
  # Time [s] at which `heater` turns off
  t₁ = FT(500)
  # Plume wavelength scaling
  x₀ = xmax

  C_smag = FT(0.23)

  imex_solver = CLIMA.DefaultSolverType()
  explicit_solver = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)

  model = AtmosModel{FT}(AtmosLESConfiguration;
                         turbulence=SmagorinskyLilly{FT}(C_smag),
                         source=(Gravity(),),
                         boundarycondition=(AtmosBC(energy=SurfaceDrivenBubbleBC{FT}(F₀, t₁, x₀)),
                                            AtmosBC()),
                         moisture=EquilMoist(),
                         init_state=init_surfacebubble!)
  config = CLIMA.Atmos_LES_Configuration("SurfaceDrivenBubble",
                                   N, resolution, xmax, ymax, zmax,
                                   init_surfacebubble!,
                                   solver_type=explicit_solver,
                                   model=model)
  return config
end

function main()
  CLIMA.init()
  FT = Float64
  # DG polynomial order
  N = 4
  # Domain resolution and size
  Δh = FT(50)
  Δv = FT(50)
  resolution = (Δh, Δh, Δv)
  xmax = 2000
  ymax = 2000
  zmax = 2000
  t0 = FT(0)
  timeend = FT(2000)

  CFL_max = FT(0.4)

  driver_config = config_surfacebubble(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, Courant_number=CFL_max, driver_config, forcecpu=true)

  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
      Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
      nothing
  end

  result = CLIMA.invoke!(solver_config;
                        user_callbacks=(cbtmarfilter,),
                        check_euclidean_distance=true)

  @test isapprox(result,FT(1); atol=1.5e-3)
end

main()
