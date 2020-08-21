#!/usr/bin/env julia --project

include("simple_box.jl")
ClimateMachine.init(
    vtk = "8640",
    checkpoint = "8640",
    checkpoint_keep_one = false,
    checkpoint_at_end = true,
)

# Float type
const FT = Float64

# simulation time
const timestart = FT(0)      # s
const timestep = FT(10)     # s
const timeend = FT(60 * 86400) # s
timespan = (timestart, timeend)

# DG polynomial order
const N = Int(4)

# Domain resolution
const Nˣ = Int(48)
const Nʸ = Int(48)
const Nᶻ = Int(15)
resolution = (N, Nˣ, Nʸ, Nᶻ)

# Domain size
const Lˣ = 1e6    # m
const Lʸ = 1e6    # m
const H = 3000   # m
dimensions = (Lˣ, Lʸ, H)

BC = (
    OceanBC(Impenetrable(FreeSlip()), Insulating()),
    OceanBC(Impenetrable(NoSlip()), Insulating()),
    OceanBC(Penetrable(KinematicStress()), TemperatureFlux()),
)

run_simple_box(
    "ocean_gyre",
    resolution,
    dimensions,
    timespan,
    OceanGyre,
    imex = false,
    Δt = timestep,
    BC = BC,
)
