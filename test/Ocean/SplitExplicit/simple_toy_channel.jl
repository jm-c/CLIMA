#!/usr/bin/env julia --project
using ClimateMachine
ClimateMachine.init(parse_clargs = true)

using ClimateMachine.BalanceLaws: vars_state, Prognostic, Auxiliary
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Filters
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates: flattenednames
using ClimateMachine.Ocean.SplitExplicit01
using ClimateMachine.GenericCallbacks
using ClimateMachine.VTK
using ClimateMachine.Checkpoint

using MPI
using LinearAlgebra
using StaticArrays
using Logging, Printf, Dates

using CLIMAParameters
using CLIMAParameters.Planet: grav
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

import ClimateMachine.Ocean.SplitExplicit01:
    ocean_init_aux!,
    ocean_init_state!,
    ocean_boundary_state!,
    CoastlineFreeSlip,
    CoastlineNoSlip,
    OceanFloorFreeSlip,
    OceanFloorNoSlip,
    OceanFloorLinearDrag,
    OceanSurfaceNoStressNoForcing,
    OceanSurfaceStressNoForcing,
    OceanSurfaceNoStressForcing,
    OceanSurfaceStressForcing,
    velocity_flux,
    temperature_flux,
    sponge_relaxation
import ClimateMachine.DGMethods:
    update_auxiliary_state!, update_auxiliary_state_gradient!, VerticalDirection
# using GPUifyLoops

const ArrayType = ClimateMachine.array_type()

struct ReentrantChannel{T} <: AbstractOceanProblem
    Lˣ::T
    Lʸ::T
    H::T
    τₒ::T # zonal wind-stress magnitude
    Cd_lin::T # linear bottom drag coefficient
    λʳ::T # surface temperature relaxtion speed
    θᴱ::T # temperature range in forcing & initial state
    hSp::T  # temperature e-folding depth
    ySp::T # sponge relaxation exponential decay length
    σʳ::T # sponge relaxation inverse time scale
end

init_temp(θᴱ, Lʸ, h, H, y, z) =
    0.5 * θᴱ * (1 - cos(π * y / Lʸ)) * (exp(z / h) - exp(-H / h)) /
    (1 - exp(-H / h))

@inline function velocity_flux(p::ReentrantChannel, y, ρ)
    # Width2 = (p.Lʸ^2 / 32)
    # term1 = exp(-(y - p.Lʸ / 2)^2 / Width2)
    # term2 = exp(-(p.Lʸ / 2)^2 / Width2)
    # return -(p.τₒ / ρ) * (term1 - term2)
    #- wind-stress from Ryan Abernethey etal, JPO, 2011 paper
    return (p.τₒ / ρ) * sin(π * y / p.Lʸ)
end

@inline function temperature_flux(p::ReentrantChannel, y, θ)
    θʳ = p.θᴱ * y / p.Lʸ
    return p.λʳ * (θʳ - θ)
end

@inline function sponge_relaxation(m::OceanModel, p::ReentrantChannel, S, Q, A)
    θʳ = init_temp(p.θᴱ, p.Lʸ, p.hSp, p.H, A.y, A.z)
    S.θ = p.σʳ * (θʳ - Q.θ) * exp((A.y - p.Lʸ) / p.ySp)

    return nothing
end

@inline function ocean_boundary_state!(
    m::OceanModel,
    p::ReentrantChannel,
    bctype,
    x...,
)
    if bctype == 1
        ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    elseif bctype == 2
        ocean_boundary_state!(m, OceanFloorLinearDrag(), x...)
    elseif bctype == 3
        ocean_boundary_state!(m, OceanSurfaceStressForcing(), x...)
    end
end

@inline function ocean_boundary_state!(
    m::Continuity3dModel,
    p::ReentrantChannel,
    bctype,
    x...,
)
    #if bctype == 1
    ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
    #end
end

@inline function ocean_boundary_state!(
    m::BarotropicModel,
    p::ReentrantChannel,
    bctype,
    x...,
)
    return ocean_boundary_state!(m, CoastlineFreeSlip(), x...)
end

function ocean_init_state!(p::ReentrantChannel, Q, A, coords, t)
    #   @inbounds y = coords[2]
    #   @inbounds z = coords[3]
    #   @inbounds H = p.H

    Q.u = @SVector [-0, -0]
    Q.η = -0
    #   Q.θ = init_temp(p.θᴱ, p.Lʸ, p.hSp, H, y, z)
    Q.θ = init_temp(p.θᴱ, p.Lʸ, p.hSp, p.H, A.y, A.z)

    return nothing
end

function ocean_init_aux!(m::OceanModel, p::ReentrantChannel, A, geom)
    FT = eltype(A)
    @inbounds A.y = geom.coord[2]
    @inbounds A.z = geom.coord[3]

    # not sure if this is needed but getting weird intialization stuff
    A.w = -0
    A.pkin = -0
    A.wz0 = -0
    A.u_d = @SVector [-0, -0]
    A.ΔGu = @SVector [-0, -0]

    return nothing
end

# A is Filled afer the state
function ocean_init_aux!(m::BarotropicModel, P::ReentrantChannel, A, geom)
    @inbounds A.y = geom.coord[2]

    A.Gᵁ = @SVector [-0, -0]
    A.U_c = @SVector [-0, -0]
    A.η_c = -0
    A.U_s = @SVector [-0, -0]
    A.η_s = -0
    A.Δu = @SVector [-0, -0]
    A.η_diag = -0
    A.Δη = -0

    return nothing
end

function main(; restart = 0)
    mpicomm = MPI.COMM_WORLD

    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    if restart == 0 && MPI.Comm_rank(mpicomm) == 0 && isdir(vtkpath)
        @info @sprintf("""Remove old dir: %s and make new one""", vtkpath)
        rm(vtkpath, recursive = true)
    end

    brickrange_2D = (xrange, yrange)
    topl_2D = BrickTopology(
        mpicomm,
        brickrange_2D;
        periodicity = (true, false),
        boundary = ((0, 0), (1, 1)),
    )
    grid_2D = DiscontinuousSpectralElementGrid(
        topl_2D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    brickrange_3D = (xrange, yrange, zrange)
    topl_3D = StackedBrickTopology(
        mpicomm,
        brickrange_3D;
        periodicity = (true, false, false),
        boundary = ((0, 0), (1, 1), (2, 3)),
    )
    grid_3D = DiscontinuousSpectralElementGrid(
        topl_3D,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = N,
    )

    prob = ReentrantChannel{FT}(Lˣ, Lʸ, H, τₒ, Cd_lin, λʳ, θᴱ, hSp, ySp, σʳ)
    gravity::FT = grav(param_set)

    #- set model time-step:
    dt_fast = 96
    dt_slow = 3456
    if t_chkp > 0
        n_chkp = ceil(Int64, t_chkp / dt_slow)
        dt_slow = t_chkp / n_chkp
    else
        n_chkp = ceil(Int64, runTime / dt_slow)
        dt_slow = runTime / n_chkp
        n_chkp = 0
    end
    n_outp = t_outp > 0 ? floor(Int64, t_outp / dt_slow) :
        ceil(Int64, runTime / dt_slow)
    ivdc_dt = numImplSteps > 0 ? dt_slow / FT(numImplSteps) : dt_slow

    model = OceanModel{FT}(
        prob,
        grav = gravity,
        cʰ = cʰ,
        add_fast_substeps = add_fast_substeps,
        numImplSteps = numImplSteps,
        ivdc_dt = ivdc_dt,
        νʰ = FT(15e3),
        νᶻ = FT(5e-3),
        κʰ = FT(1e3),
        κᶻ = FT(3e-4),
        κᶜ = FT(1.0),
        fₒ = FT(-1e-4),
        β = FT(1.e-11),
    )
    # model = OceanModel{FT}(prob, cʰ = cʰ, νʰ = FT(0), νᶻ = FT(1e-3), fₒ = FT(0), β = FT(0) )

    barotropicmodel = BarotropicModel(model)

    minΔx = min_node_distance(grid_3D, HorizontalDirection())
    minΔz = min_node_distance(grid_3D, VerticalDirection())
    #- 2 horiz directions
    gravity_max_dT = 1 / (2 * sqrt(gravity * H) / minΔx)
    # dt_fast = minimum([gravity_max_dT])

    #- 2 horiz directions + harmonic visc or diffusion: 2^2 factor in CFL:
    viscous_max_dT = 1 / (2 * model.νʰ / minΔx^2 + model.νᶻ / minΔz^2) / 4
    diffusive_max_dT = 1 / (2 * model.κʰ / minΔx^2 + model.κᶻ / minΔz^2) / 4
    # dt_slow = minimum([diffusive_max_dT, viscous_max_dT])

    @info @sprintf(
        """Update
           Gravity Max-dT = %.1f
           Timestep       = %.1f""",
        gravity_max_dT,
        dt_fast
    )

    @info @sprintf(
        """Update
       Viscous   Max-dT = %.1f
       Diffusive Max-dT = %.1f
       Timestep      = %.1f""",
        viscous_max_dT,
        diffusive_max_dT,
        dt_slow
    )

    if restart > 0
        direction = EveryDirection()
        Q_3D, _, t0 =
            read_checkpoint(vtkpath, "baroclinic", ArrayType, mpicomm, restart)
        Q_2D, _, _ =
            read_checkpoint(vtkpath, "barotropic", ArrayType, mpicomm, restart)

        dg = OceanDGModel(
            model,
            grid_3D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )
        barotropic_dg = DGModel(
            barotropicmodel,
            grid_2D,
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )

        Q_3D = restart_ode_state(dg, Q_3D; init_on_cpu = true)
        Q_2D = restart_ode_state(barotropic_dg, Q_2D; init_on_cpu = true)

    else
        t0 = 0
        dg = OceanDGModel(
            model,
            grid_3D,
            # CentralNumericalFluxFirstOrder(),
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )
        barotropic_dg = DGModel(
            barotropicmodel,
            grid_2D,
            # CentralNumericalFluxFirstOrder(),
            RusanovNumericalFlux(),
            CentralNumericalFluxSecondOrder(),
            CentralNumericalFluxGradient(),
        )

        Q_3D = init_ode_state(dg, FT(0); init_on_cpu = true)
        Q_2D = init_ode_state(barotropic_dg, FT(0); init_on_cpu = true)

    end
    timeend = runTime + t0

    lsrk_ocean = LS3NRK33Heuns(dg, Q_3D, dt = dt_slow, t0 = t0)
    lsrk_barotropic = LS3NRK33Heuns(barotropic_dg, Q_2D, dt = dt_fast, t0 = t0)

    odesolver = SplitExplicitLSRK3nSolver(lsrk_ocean, lsrk_barotropic)

    #-- Set up State Check call back for config state arrays, called every ntFrq_SC time steps
    cbcs_dg = ClimateMachine.StateCheck.sccreate(
        [
            (Q_3D, "oce Q_3D"),
            (dg.state_auxiliary, "oce aux"),
            # (dg.diffstate,"oce diff",),
            # (lsrk_ocean.dQ,"oce_dQ",),
            # (dg.modeldata.tendency_dg.state_auxiliary,"tend Int aux",),
            # (dg.modeldata.conti3d_Q,"conti3d_Q",),
            (Q_2D, "baro Q_2D"),
            (barotropic_dg.state_auxiliary, "baro aux"),
        ],
        ntFrq_SC;
        prec = 12,
    )
    # (barotropic_dg.diffstate,"baro diff",),
    # (lsrk_barotropic.dQ,"baro_dQ",)
    #--

    cb_ntFrq = [n_outp, n_chkp]
    outp_nb = round(Int64, restart * n_chkp / n_outp)
    step = [outp_nb, outp_nb, restart + 1]
    cbvector = make_callbacks(
        vtkpath,
        step,
        cb_ntFrq,
        timeend,
        mpicomm,
        odesolver,
        dg,
        model,
        Q_3D,
        barotropic_dg,
        barotropicmodel,
        Q_2D,
    )

    eng0 = norm(Q_3D)
    @info @sprintf """Starting
    norm(Q₀) = %.16e
    ArrayType = %s""" eng0 ArrayType

    # slow fast state tuple
    Qvec = (slow = Q_3D, fast = Q_2D)
    # solve!(Qvec, odesolver; timeend = timeend, callbacks = cbvector)
    cbv = (cbvector..., cbcs_dg)
    solve!(Qvec, odesolver; timeend = timeend, callbacks = cbv)

    ## Enable the code block below to print table for use in reference value code
    ## reference value code sits in a file named $(@__FILE__)_refvals.jl. It is hand
    ## edited using code generated by block below when reference values are updated.
    regenRefVals = false
    if regenRefVals
        ## Print state statistics in format for use as reference values
        println(
            "# SC ========== Test number ",
            1,
            " reference values and precision match template. =======",
        )
        println("# SC ========== $(@__FILE__) test reference values ======================================")
        ClimateMachine.StateCheck.scprintref(cbcs_dg)
        println("# SC ====================================================================================")
    end

    ## Check results against reference if present
    checkRefVals = true
    if checkRefVals
        include("../refvals/simple_toy_channel_refvals.jl")
        refDat = (refVals[1], refPrecs[1])
        checkPass = ClimateMachine.StateCheck.scdocheck(cbcs_dg, refDat)
        checkPass ? checkRep = "Pass" : checkRep = "Fail"
        @info @sprintf("""Compare vs RefVals: %s""", checkRep)
    end

    return nothing
end

function make_callbacks(
    vtkpath,
    step,
    ntFrq,
    timeend,
    mpicomm,
    odesolver,
    dg_slow,
    model_slow,
    Q_slow,
    dg_fast,
    model_fast,
    Q_fast,
)
    n_outp = ntFrq[1]
    n_chkp = ntFrq[2]
    mkpath(vtkpath)
    mkpath(vtkpath * "/slow")
    mkpath(vtkpath * "/fast")

    function do_output(span, step, model, dg, Q)
        outprefix = @sprintf(
            "%s/%s/mpirank%04d_step%04d",
            vtkpath,
            span,
            MPI.Comm_rank(mpicomm),
            step
        )
        @info "doing VTK output" outprefix
        statenames = flattenednames(vars_state(model, Prognostic(), eltype(Q)))
        auxnames = flattenednames(vars_state(model, Auxiliary(), eltype(Q)))
        writevtk(outprefix, Q, dg, statenames, dg.state_auxiliary, auxnames)

        mycomm = Q.mpicomm
        ## Generate the pvtu file for these vtk files
        if MPI.Comm_rank(mpicomm) == 0 && MPI.Comm_size(mpicomm) > 1
            ## name of the pvtu file
            pvtuprefix = @sprintf("%s/%s/step%04d", vtkpath, span, step)
            ## name of each of the ranks vtk files
            prefixes = ntuple(MPI.Comm_size(mpicomm)) do i
                @sprintf("mpirank%04d_step%04d", i - 1, step)
            end
            writepvtu(
                pvtuprefix,
                prefixes,
                (statenames..., auxnames...),
                eltype(Q),
            )
            @info "Done writing VTK: $pvtuprefix"
        end

    end

    do_output("slow", step[1], model_slow, dg_slow, Q_slow)
    step[1] += 1
    cbvtk_slow =
        GenericCallbacks.EveryXSimulationSteps(n_outp) do (init = false)
            do_output("slow", step[1], model_slow, dg_slow, Q_slow)
            step[1] += 1
            nothing
        end

    do_output("fast", step[2], model_fast, dg_fast, Q_fast)
    step[2] += 1
    cbvtk_fast =
        GenericCallbacks.EveryXSimulationSteps(n_outp) do (init = false)
            do_output("fast", step[2], model_fast, dg_fast, Q_fast)
            step[2] += 1
            nothing
        end

    starttime = Ref(now())
    cbinfo = GenericCallbacks.EveryXWallTimeSeconds(60, mpicomm) do (s = false)
        if s
            starttime[] = now()
        else
            energy = norm(Q_slow)
            @info @sprintf(
                """Update
                simtime = %8.2f / %8.2f
                runtime = %s
                norm(Q) = %.16e""",
                ODESolvers.gettime(odesolver),
                timeend,
                Dates.format(
                    convert(Dates.DateTime, Dates.now() - starttime[]),
                    Dates.dateformat"HH:MM:SS",
                ),
                energy
            )
        end
    end

    if n_chkp > 0
        # Note: write zeros instead of Aux vars (not needed to restart); would be
        # better just to write state vars (once write_checkpoint() can handle it)
        cb_checkpoint = GenericCallbacks.EveryXSimulationSteps(n_chkp) do
            write_checkpoint(
                Q_slow,
                zero(Q_slow),
                odesolver,
                vtkpath,
                "baroclinic",
                mpicomm,
                step[3],
            )

            write_checkpoint(
                Q_fast,
                zero(Q_fast),
                odesolver,
                vtkpath,
                "barotropic",
                mpicomm,
                step[3],
            )

            step[3] += 1
            nothing
        end
        return (cbvtk_slow, cbvtk_fast, cbinfo, cb_checkpoint)
    else
        return (cbvtk_slow, cbvtk_fast, cbinfo)
    end

end

#################
# RUN THE TESTS #
#################
FT = Float64
vtkpath = "vtk_split"

const runTime = 5 * 86400 # s
const t_outp = 1 * 86400  # s
const t_chkp = runTime  # s
#const ntFrq_SC = 5 # frequency (in time-step) for State-Check output
#const runTime = 5 * 3456 # s
#const t_outp = 5 * 3456  # s
#const t_chkp = 0
const ntFrq_SC = 1 # frequency (in time-step) for State-Check output

const N = 4
const Nˣ = 5
const Nʸ = 10
const Nᶻ = 15
const Lˣ = 1e6  # m
const Lʸ = 2e6  # m
const H = 3000  # m

xrange = range(FT(0); length = Nˣ + 1, stop = Lˣ)
yrange = range(FT(0); length = Nʸ + 1, stop = Lʸ)
zrange = range(FT(-H); length = Nᶻ + 1, stop = 0)
# dz = [576, 540, 433, 339, 266, 208, 162, 128, 99, 75, 58, 43, 31, 22, 20]
# zrange = zeros(FT, Nᶻ + 1)
# zrange[2:(Nᶻ + 1)] .= cumsum(dz)
# zrange .-= H

#const cʰ = sqrt(gravity * H)
const cʰ = 1  # typical of ocean internal-wave speed
const cᶻ = 0

#- inverse ratio of additional fast time steps (for weighted average)
#  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
# e.g., = 1 --> 100% more ; = 2 --> 50% more ; = 3 --> 33% more ...
add_fast_substeps = 3

#- number of Implicit vertical-diffusion sub-time-steps within one model full time-step
# default = 0 : disable implicit vertical diffusion
numImplSteps = 5

#- values for τₒ , Cd_lin , θᴱ , hSp & σʳ are from Ryan Abernethey etal, JPO, 2011 paper: 
const τₒ = 2e-1        # zonal wind-stress magnitude (Pa = N/m^2)
const Cd_lin = 1.1e-3  # linear bottom drag coefficient (m/s)
const λʳ = 10 // 86400 # surface temperature relaxtion speed (m/s)
const θᴱ = 10          # temperature range in forcing & initial state (deg.C)
const hSp = 1000       # temperature e-folding depth (m)
const ySp = 20e3       # sponge relaxation exponential decay length (m)
const σʳ = 1 // (7 * 86400) # sponge relaxation inverse time scale (s^-1)
#- Note: in Ryan's channel, 
#  a) seems θᴱ = 8 
#  b) sponge done with pkg/rbcs, zero relaxation @ y = Ly - 100.km, upto 1/(7.d) @ y = Ly
#  c) no Temp surface relaxation but instead Qnet(y)

main(restart = 0)
