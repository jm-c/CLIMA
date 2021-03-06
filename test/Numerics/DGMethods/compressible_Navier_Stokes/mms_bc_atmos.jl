using Test
using Dates
using LinearAlgebra
using MPI
using Printf
using StaticArrays
using UnPack

using ClimateMachine
using ClimateMachine.ConfigTypes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.GenericCallbacks
using ClimateMachine.Atmos
using ClimateMachine.BalanceLaws
using ClimateMachine.Orientations
using ClimateMachine.VariableTemplates
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.VTK

import ClimateMachine.Atmos: filter_source, atmos_source!
import ClimateMachine.BalanceLaws: source

using CLIMAParameters
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import CLIMAParameters
# Assume zero reference temperature
CLIMAParameters.Planet.T_0(::EarthParameterSet) = 0

if !@isdefined integration_testing
    const integration_testing = parse(
        Bool,
        lowercase(get(ENV, "JULIA_CLIMA_INTEGRATION_TESTING", "false")),
    )
end

include("mms_solution_generated.jl")

import ClimateMachine.Thermodynamics: total_specific_enthalpy
using ClimateMachine.Atmos

total_specific_enthalpy(ts::PhaseDry{FT}, e_tot::FT) where {FT <: Real} =
    zero(FT)

function mms2_init_state!(problem, bl, state::Vars, aux::Vars, localgeo, t)
    (x1, x2, x3) = localgeo.coord
    state.ρ = ρ_g(t, x1, x2, x3, Val(2))
    state.ρu = SVector(
        U_g(t, x1, x2, x3, Val(2)),
        V_g(t, x1, x2, x3, Val(2)),
        W_g(t, x1, x2, x3, Val(2)),
    )
    state.ρe = E_g(t, x1, x2, x3, Val(2))
end

struct MMSSource{PV <: Union{Mass, Momentum, Energy}, N} <:
       TendencyDef{Source, PV} end

filter_source(pv::PV, m::AtmosModel, s::MMSSource{PV}) where {PV} = s
atmos_source!(::MMSSource, args...) = nothing

MMSSource(N::Int) =
    (MMSSource{Mass, N}(), MMSSource{Momentum, N}(), MMSSource{Energy, N}())


function source(s::MMSSource{Mass, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return Sρ_g(t, x1, x2, x3, Val(N))
end
function source(s::MMSSource{Momentum, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return SVector(
        SU_g(t, x1, x2, x3, Val(N)),
        SV_g(t, x1, x2, x3, Val(N)),
        SW_g(t, x1, x2, x3, Val(N)),
    )
end
function source(s::MMSSource{Energy, N}, m, args) where {N}
    @unpack aux, t = args
    x1, x2, x3 = aux.coord
    return SE_g(t, x1, x2, x3, Val(N))
end

function mms3_init_state!(problem, bl, state::Vars, aux::Vars, localgeo, t)
    (x1, x2, x3) = localgeo.coord
    state.ρ = ρ_g(t, x1, x2, x3, Val(3))
    state.ρu = SVector(
        U_g(t, x1, x2, x3, Val(3)),
        V_g(t, x1, x2, x3, Val(3)),
        W_g(t, x1, x2, x3, Val(3)),
    )
    state.ρe = E_g(t, x1, x2, x3, Val(3))
end

# initial condition

function test_run(mpicomm, ArrayType, dim, topl, warpfun, N, timeend, FT, dt)

    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
        meshwarp = warpfun,
    )

    if dim == 2
        problem = AtmosProblem(
            boundaryconditions = (InitStateBC(),),
            init_state_prognostic = mms2_init_state!,
        )
        model = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            problem = problem,
            orientation = NoOrientation(),
            ref_state = NoReferenceState(),
            turbulence = ConstantDynamicViscosity(
                FT(μ_exact),
                WithDivergence(),
            ),
            moisture = DryModel(),
            source = (MMSSource(2)...,),
        )
    else
        problem = AtmosProblem(
            boundaryconditions = (InitStateBC(),),
            init_state_prognostic = mms3_init_state!,
        )
        model = AtmosModel{FT}(
            AtmosLESConfigType,
            param_set;
            problem = problem,
            orientation = NoOrientation(),
            ref_state = NoReferenceState(),
            turbulence = ConstantDynamicViscosity(
                FT(μ_exact),
                WithDivergence(),
            ),
            moisture = DryModel(),
            source = (MMSSource(3)...,),
        )
    end
    show_tendencies(model)

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))
    Qcpu = init_ode_state(dg, FT(0); init_on_cpu = true)
    @test euclidean_distance(Q, Qcpu) < sqrt(eps(FT))

    lsrk = LSRK54CarpenterKennedy(dg, Q; dt = dt, t0 = 0)

    eng0 = norm(Q)
    @info @sprintf """Starting
    norm(Q₀) = %.16e""" eng0

    # Set up the information callback
    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q)
            @info @sprintf(
                """Update
                simtime = %.16e
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(lsrk),
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    solve!(Q, lsrk; timeend = timeend, callbacks = (cbinfo,))
    # solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))


    # Print some end of the simulation information
    engf = norm(Q)
    Qe = init_ode_state(dg, FT(timeend))

    engfe = norm(Qe)
    errf = euclidean_distance(Q, Qe)
    @info @sprintf """Finished
    norm(Q)                 = %.16e
    norm(Q) / norm(Q₀)      = %.16e
    norm(Q) - norm(Q₀)      = %.16e
    norm(Q - Qe)            = %.16e
    norm(Q - Qe) / norm(Qe) = %.16e
    """ engf engf / eng0 engf - eng0 errf errf / engfe
    errf
end

let
    ClimateMachine.init()
    ArrayType = ClimateMachine.array_type()

    mpicomm = MPI.COMM_WORLD

    polynomialorder = 4
    base_num_elem = 4

    expected_result = [
        1.6931876910307017e-01 5.4603193051929394e-03 2.3307776694542282e-04
        3.3983777728925593e-02 1.7808380837573065e-03 9.176181458773599e-5
    ]
    lvls = integration_testing ? size(expected_result, 2) : 1

    @testset "mms_bc_atmos" begin
        for FT in (Float64,) #Float32)
            result = zeros(FT, lvls)
            for dim in 2:3
                for l in 1:lvls
                    if dim == 2
                        Ne = (
                            2^(l - 1) * base_num_elem,
                            2^(l - 1) * base_num_elem,
                        )
                        brickrange = (
                            range(FT(0); length = Ne[1] + 1, stop = 1),
                            range(FT(0); length = Ne[2] + 1, stop = 1),
                        )
                        topl = BrickTopology(
                            mpicomm,
                            brickrange,
                            periodicity = (false, false),
                        )
                        dt = 1e-2 / Ne[1]
                        warpfun =
                            (x1, x2, _) -> begin
                                (x1 + sin(x1 * x2), x2 + sin(2 * x1 * x2), 0)
                            end

                    elseif dim == 3
                        Ne = (
                            2^(l - 1) * base_num_elem,
                            2^(l - 1) * base_num_elem,
                        )
                        brickrange = (
                            range(FT(0); length = Ne[1] + 1, stop = 1),
                            range(FT(0); length = Ne[2] + 1, stop = 1),
                            range(FT(0); length = Ne[2] + 1, stop = 1),
                        )
                        topl = BrickTopology(
                            mpicomm,
                            brickrange,
                            periodicity = (false, false, false),
                        )
                        dt = 5e-3 / Ne[1]
                        warpfun =
                            (x1, x2, x3) -> begin
                                (
                                    x1 +
                                    (x1 - 1 / 2) * cos(2 * π * x2 * x3) / 4,
                                    x2 + exp(sin(2π * (x1 * x2 + x3))) / 20,
                                    x3 + x1 / 4 + x2^2 / 2 + sin(x1 * x2 * x3),
                                )
                            end
                    end
                    timeend = 1
                    nsteps = ceil(Int64, timeend / dt)
                    dt = timeend / nsteps

                    @info (ArrayType, FT, dim, nsteps, dt)
                    result[l] = test_run(
                        mpicomm,
                        ArrayType,
                        dim,
                        topl,
                        warpfun,
                        polynomialorder,
                        timeend,
                        FT,
                        dt,
                    )
                    @test result[l] ≈ FT(expected_result[dim - 1, l])
                end
                if integration_testing
                    @info begin
                        msg = ""
                        for l in 1:(lvls - 1)
                            rate = log2(result[l]) - log2(result[l + 1])
                            msg *= @sprintf(
                                "\n  rate for level %d = %e\n",
                                l,
                                rate
                            )
                        end
                        msg
                    end
                end
            end
        end
    end
end
