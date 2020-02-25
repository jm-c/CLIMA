using Distributions: Uniform
using LinearAlgebra
using StaticArrays
using Random: rand
using Test

using CLIMA
using CLIMA.Atmos
using CLIMA.GenericCallbacks
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.AdditiveRungeKuttaMethod
using CLIMA.ColumnwiseLUSolver: ManyColumnLU
using CLIMA.Mesh.Filters
using CLIMA.Mesh.Grids
using CLIMA.MoistThermodynamics
using CLIMA.PlanetParameters
using CLIMA.VariableTemplates


const pressure_ground = MSLP
const T_init = 300.0 # unit: Kelvin 
const domain_height = 30e3 # unit: meters


function init_heldsuarez!(bl, state, aux, coords, t)
    FT            = eltype(state)
    pressure_sfc  = FT(MSLP)
    temp_init     = FT(300.0)
    radius        = FT(planet_radius)

    # Initialize the state variables of the model
    height = norm(coords, 2) - radius #TODO: altitude(bl.orientation, aux)
    scale_height = R_d * temp_init / grav
    pressure = pressure_sfc * exp(-height / scale_height)

    rnd      = FT(1.0 + rand(Uniform([-1e-3, 1e-3])))
    state.ρ  = rnd * air_density(temp_init, pressure)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (internal_energy(temp_init) + aux.orientation.Φ)

    nothing
end


function config_heldsuarez(FT, poly_order, resolution)
    name          = "HeldSuarez"
    domain_height = FT(30e3)
    T_ref         = FT(300.0)
    Rh_ref        = FT(0.0)
    turb_visc     = FT(0.0)

    # Configure the model setup
    model = AtmosModel{FT}(
      AtmosGCMConfiguration;
      
      ref_state  = HydrostaticState(
                     IsothermalProfile(T_ref), 
                     Rh_ref
                   ),
      turbulence = ConstantViscosityWithDivergence(turb_visc),
      moisture   = DryModel(),
      source     = (Gravity(), Coriolis(), held_suarez_forcing!),
      init_state = init_heldsuarez!
    )

    config = CLIMA.Atmos_GCM_Configuration(
      name, 
      poly_order, 
      resolution,
      domain_height,
      init_heldsuarez!;
      
      model = model
    )

    return config
end


function held_suarez_forcing!(bl, source, state, diffusive, aux, t::Real)
    global T_init

    FT = eltype(state)

    ρ     = state.ρ
    ρu    = state.ρu
    ρe    = state.ρe
    coord = aux.coord
    Φ     = aux.orientation.Φ
    e     = ρe / ρ
    u     = ρu / ρ
    e_int = e - u' * u / 2 - Φ
    T     = air_temperature(e_int)

    # Held-Suarez constants
    k_a       = FT(1 / (40 * day))
    k_f       = FT(1 / day)
    k_s       = FT(1 / (4 * day)) # TODO: day is actually seconds per day; should it be named better?
    ΔT_y      = FT(60)
    Δθ_z      = FT(10)
    T_equator = FT(315)
    T_min     = FT(200)

    σ_b          = FT(7 / 10)
    r            = norm(coord, 2)
    @inbounds λ  = atan(coord[2], coord[1])
    @inbounds φ  = asin(coord[3] / r)
    h            = r - FT(planet_radius)
    scale_height = R_d * FT(T_init) / grav
    σ            = exp(-h / scale_height)

    # TODO: use
    #  p = air_pressure(T, ρ)
    #  σ = p/p0
    exner_p       = σ ^ (R_d / cp_d)
    Δσ            = (σ - σ_b) / (1 - σ_b)
    height_factor = max(0, Δσ)
    T_equil       = (T_equator - ΔT_y * sin(φ) ^ 2 - Δθ_z * log(σ) * cos(φ) ^ 2 ) * exner_p
    T_equil       = max(T_min, T_equil)
    k_T           = k_a + (k_s - k_a) * height_factor * cos(φ) ^ 4
    k_v           = k_f * height_factor

    # TODO: bottom drag should only be applied in tangential direction
    source.ρu += -k_v * ρu
    source.ρe += -k_T * ρ * cv_d * (T - T_equil)
end

function main()
    CLIMA.init()

    # Driver configuration parameters
    FT            = Float32           # floating type precision
    poly_order    = 5                 # discontinuous Galerkin polynomial order
    n_horz        = 5                 # horizontal element number  
    n_vert        = 5                 # vertical element number
    days          = 1                 # experiment day number
    timestart     = FT(0)             # start time (seconds)
    timeend       = FT(days*24*60*60) # end time (seconds)
    
    # Set up driver configuration
    driver_config = config_heldsuarez(FT, poly_order, (n_horz, n_vert))
    
    # Set up ODE solver configuration
    #ode_solver_type = CLIMA.ExplicitSolverType(
    #  solver_method=LSRK144NiegemannDiehlBusch
    #)
    ode_solver_type = CLIMA.IMEXSolverType(
      linear_solver = ManyColumnLU,
      solver_method = ARK2GiraldoKellyConstantinescu
    )

    # Set up experiment
    solver_config = CLIMA.setup_solver(
      timestart, 
      timeend, 
      driver_config,
      ode_solver_type=ode_solver_type,
      Courant_number=0.4,
      forcecpu=true
    )

    # Set up user-defined callbacks
    # TODO: This callback needs to live somewhere else 
    filterorder = 14
    filter = ExponentialFilter(solver_config.dg.grid, 0, filterorder)
    cbfilter = GenericCallbacks.EveryXSimulationSteps(1) do
        Filters.apply!(
          solver_config.Q, 
          1:size(solver_config.Q, 2),
          solver_config.dg.grid, 
          filter
        )
        nothing
    end

    # Run the model
    result = CLIMA.invoke!(
      solver_config;
      user_callbacks = (cbfilter,),
      check_euclidean_distance = true
    )
end

main()
