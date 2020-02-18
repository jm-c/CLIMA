using Distributions
using Random
using StaticArrays
using Test
using Printf
using NCDatasets
using Dierckx
using LinearAlgebra
using DocStringExtensions

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.Mesh.Filters
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates
import CLIMA.DGmethods: vars_state, vars_aux, vars_integrals,
                        integrate_aux!

import CLIMA.DGmethods: boundary_state!
import CLIMA.Atmos: atmos_boundary_state!, atmos_boundary_flux_diffusive!, flux_diffusive!, atmos_source!
import CLIMA.DGmethods.NumericalFluxes: boundary_flux_diffusive!

"""
CMIP6 Test Dataset - cfsites
@Article{gmd-10-359-2017,
AUTHOR = {Webb, M. J. and Andrews, T. and Bodas-Salcedo, A. and Bony, S. and Bretherton, C. S. and Chadwick, R. and Chepfer, H. and Douville, H. and Good, P. and Kay, J. E. and Klein, S. A. and Marchand, R. and Medeiros, B. and Siebesma, A. P. and Skinner, C. B. and Stevens, B. and Tselioudis, G. and Tsushima, Y. and Watanabe, M.},
TITLE = {The Cloud Feedback Model Intercomparison Project (CFMIP) contribution to CMIP6},
JOURNAL = {Geoscientific Model Development},
VOLUME = {10},
YEAR = {2017},
NUMBER = {1},
PAGES = {359--384},
URL = {https://www.geosci-model-dev.net/10/359/2017/},
DOI = {10.5194/gmd-10-359-2017}
}
"""

const seed = MersenneTwister(0)

# ---------------------------- Begin Boundary Conditions ----------------- #
"""
  CFSites_BC <: BoundaryCondition
  Prescribes boundary conditions for Dynamics of Marine Stratocumulus Case
#Fields
$(DocStringExtensions.FIELDS)
"""
struct CFSites_BC{FT} <: BoundaryCondition
  "Drag coefficient"
  C_drag::FT
  "Latent Heat Flux"
  LHF::FT
  "Sensible Heat Flux"
  SHF::FT
end

"""
    atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                          bc::CFSites_BC, args...)

For the non-diffussive and gradient terms we just use the `NoFluxBC`
"""
atmos_boundary_state!(nf::Union{NumericalFluxNonDiffusive, NumericalFluxGradient},
                      bc::CFSites_BC, 
                      args...) = atmos_boundary_state!(nf, NoFluxBC(), args...)

"""
    atmos_boundary_flux_diffusive!(nf::NumericalFluxDiffusive,
                                   bc::CFSites_BC, atmos::AtmosModel,
                                   F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)

When `bctype == 1` the `NoFluxBC` otherwise the specialized CFSites BC is used
"""
function atmos_boundary_flux_diffusive!(nf::CentralNumericalFluxDiffusive,
                                        bc::CFSites_BC, 
                                        atmos::AtmosModel, F,
                                        state⁺, diff⁺, aux⁺, 
                                        n⁻,
                                        state⁻, diff⁻, aux⁻,
                                        bctype, t,
                                        state1⁻, diff1⁻, aux1⁻)
  if bctype != 1
    atmos_boundary_flux_diffusive!(nf, NoFluxBC(), atmos, F,
                                   state⁺, diff⁺, aux⁺, n⁻,
                                   state⁻, diff⁻, aux⁻,
                                   bctype, t,
                                   state1⁻, diff1⁻, aux1⁻)
  else
    # Start with the noflux BC and then build custom flux from there
    atmos_boundary_state!(nf, NoFluxBC(), atmos,
                          state⁺, diff⁺, aux⁺, n⁻,
                          state⁻, diff⁻, aux⁻,
                          bctype, t)

    # ------------------------------------------------------------------------
    # (<var>_FN) First node values (First interior node from bottom wall)
    # ------------------------------------------------------------------------
    u_FN = state1⁻.ρu / state1⁻.ρ
    windspeed_FN = norm(u_FN)

    # ----------------------------------------------------------
    # Extract components of diffusive momentum flux (minus-side)
    # ----------------------------------------------------------
    _, τ⁻ = turbulence_tensors(atmos.turbulence, state⁻, diff⁻, aux⁻, t)

    # ----------------------------------------------------------
    # Boundary momentum fluxes
    # ----------------------------------------------------------
    # Case specific for flat bottom topography, normal vector is n⃗ = k⃗ = [0, 0, 1]ᵀ
    # A more general implementation requires (n⃗ ⋅ ∇A) to be defined where A is
    # replaced by the appropriate flux terms
    C_drag = bc.C_drag
    @inbounds begin
      τ13⁺ = - C_drag * windspeed_FN * u_FN[1]
      τ23⁺ = - C_drag * windspeed_FN * u_FN[2]
      τ21⁺ = τ⁻[2,1]
    end

    # Assign diffusive momentum and moisture fluxes
    # (i.e. ρ𝛕 terms)
    FT = eltype(state⁺)
    τ⁺ = SHermitianCompact{3, FT, 6}(SVector(0   ,
                                             τ21⁺, τ13⁺,
                                             0   , τ23⁺, 0))

    # ----------------------------------------------------------
    # Boundary moisture fluxes
    # ----------------------------------------------------------
    # really ∇q_tot is being used to store d_q_tot
    d_q_tot⁺  = SVector(0, 0, bc.LHF/(LH_v0))

    # ----------------------------------------------------------
    # Boundary energy fluxes
    # ----------------------------------------------------------
    # Assign diffusive enthalpy flux (i.e. ρ(J+D) terms)
    d_h_tot⁺ = SVector(0, 0, bc.LHF + bc.SHF)

    # Set the flux using the now defined plus-side data
    flux_diffusive!(atmos, F, state⁺, τ⁺, d_h_tot⁺)
    flux_diffusive!(atmos.moisture, F, state⁺, d_q_tot⁺)
  end
end

struct GCMRelaxation{FT} <: Source
  "Relaxation timescale"
  τ_relax::FT
end
function atmos_source!(s::GCMRelaxation, atmos::AtmosModel, source::Vars, state::Vars, aux::Vars, t::Real)
  source.ρe                   -= (state.ρe - aux.ref_state.ρe) / s.τ_relax
  source.moisture.ρq_tot      -= (state.moisture.ρq_tot - aux.ref_state.ρq_tot) /s.τ_relax
  source.ρu                   -= (state.ρu - aux.ref_state.ρu) / s.τ_relax
end
# ------------------------ End Boundary Condition --------------------- # 
#
# Get initial condition from NCData 
#
function get_ncdata()
  data = Dataset("/home/asridhar/CLIMA/datasets/cfsites_forcing.2010071518.nc","r");
  # Load specific site group via numeric ID in NetCDF file (requires generalisation)
  siteid = data.group["site22"];
  # Allow strings to be read as varnames
  function str2var(str::String, var::Any)
    str = Symbol(str);
    @eval(($str)=($var));
  end
  # Load all variables
  for (varname,var) in siteid
    str2var(varname,var[:,1]);
  end
  initdata = [height pfull temp ucomp vcomp sphum]
  return initdata
end

function init_cfsites!(state::Vars, aux::Vars, (x,y,z), t, splines)
  
  FT = eltype(state)
  (spl_temp, spl_pfull, spl_ucomp, spl_vcomp, spl_sphum) = splines

  T     = FT(spl_temp(z))
  q_tot = FT(spl_sphum(z))
  u     = FT(spl_ucomp(z))
  v     = FT(spl_vcomp(z))
  P     = FT(spl_pfull(z))

  ρ     = air_density(T,P,PhasePartition(q_tot))
  e_int = internal_energy(T,PhasePartition(q_tot))
  e_kin = (u^2 + v^2)/2  
  e_pot = grav * z 
  # Assignment of state variables
  state.ρ = ρ
  state.ρu = ρ * SVector(u,v,0)
  state.ρe = ρ * (e_kin + e_pot + e_int)
  if z <= FT(600)
    state.ρe += rand(seed)*FT(1/100)*(state.ρe)
  end
  state.moisture.ρq_tot = ρ * q_tot
  #=
  FT = eltype(state)
  # Assignment of state variables
  state.ρ =  aux.ref_state.ρ
  state.ρu = aux.ref_state.ρu
  state.ρe = aux.ref_state.ρe
  state.moisture.ρq_tot = aux.ref_state.ρq_tot
  if z <= FT(500)
    state.ρe += rand(seed)*FT(2/100)*(state.ρe)
  end
  =# 
end

function config_cfsites(FT, N, resolution, xmax, ymax, zmax)
  
  # Boundary Conditions

  model = AtmosModel{FT}(AtmosLESConfiguration;
                         ref_state=GCMForcedState(),
                         turbulence=SmagorinskyLilly{FT}(0.23),
                         moisture=EquilMoist(),
                         source=(Gravity(),),
                         boundarycondition=NoFluxBC(),
                         init_state=init_cfsites!)
  
  imex_solver = CLIMA.DefaultSolverType()
  exp_solver  = CLIMA.ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch)
  
  config = CLIMA.Atmos_LES_Configuration("CFSites Experiments",
                                         N, resolution, xmax, ymax, zmax,
                                         init_cfsites!,
                                         solver_type=exp_solver,
                                         model=model)
  return config
end

function main()
  CLIMA.init()

  # Working precision
  FT = Float64
  # DG polynomial order
  N = 4
  # Domain resolution and size
  Δh = FT(50)
  Δv = FT(50)
  resolution = (Δh, Δh, Δv)
  # Domain extents
  xmax = 2000
  ymax = 2000
  zmax = 2000
  # Simulation time
  t0 = FT(0)
  timeend = FT(3600*6)
  # Courant number
  CFL = FT(0.25)

  initdata = get_ncdata()

  z = initdata[:,1];
  pfull = initdata[:,2];
  temp  = initdata[:,3];
  ucomp = initdata[:,4];
  vcomp = initdata[:,5];
  sphum = initdata[:,6];

  splines = (spl_temp  = Spline1D(z,temp), 
             spl_pfull = Spline1D(z,pfull), 
             spl_ucomp = Spline1D(z,ucomp),
             spl_vcomp = Spline1D(z,vcomp),
             spl_sphum = Spline1D(z,sphum))

  driver_config = config_cfsites(FT, N, resolution, xmax, ymax, zmax)
  solver_config = CLIMA.setup_solver(t0, timeend, driver_config, splines; 
                                     forcecpu=true, Courant_number=CFL, aux_args=splines)

  # User defined filter (TMAR positivity preserving filter)
  cbtmarfilter = GenericCallbacks.EveryXSimulationSteps(1) do (init=false)
      Filters.apply!(solver_config.Q, 6, solver_config.dg.grid, TMARFilter())
      nothing
  end

  # Invoke solver (calls solve! function for time-integrator)
  result = CLIMA.invoke!(solver_config;
                        user_callbacks=(cbtmarfilter,),
                        check_euclidean_distance=true)

end
main()
