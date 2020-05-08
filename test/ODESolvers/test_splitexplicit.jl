using MPI
using Test
using Logging
using StaticArrays
using LinearAlgebra: norm

using CLIMA
using CLIMA.Mesh.Topologies
using CLIMA.Mesh.Grids
using CLIMA.DGmethods
using CLIMA.DGmethods.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.ODESolvers

using CLIMA.GenericCallbacks
using Logging, Printf, Dates
 #=
#using CLIMA.VTK
#-- Add State statistics package
using Pkg
Pkg.add(
 PackageSpec(url="https://github.com/christophernhill/temp-clima-statetools",rev="0.1.2")
)
using CLIMAStateCheck
#--
 =#

#const mrrk_methods =
#    ((LSRK54CarpenterKennedy, 4), (LSRK144NiegemannDiehlBusch, 4))
mrrk_methods = LSRK54CarpenterKennedy
#const mrrk_methods = (LSRK144NiegemannDiehlBusch, 4)

include("two_state_model.jl")

@testset "Split-Explicit RK solvers" begin
    CLIMA.init()
    ArrayType = CLIMA.array_type()

    mpicomm = MPI.COMM_WORLD
    ll = uppercase(get(ENV, "JULIA_LOG_LEVEL", "INFO"))
    loglevel = ll == "DEBUG" ? Logging.Debug :
        ll == "WARN" ? Logging.Warn :
        ll == "ERROR" ? Logging.Error : Logging.Info
    logger_stream = MPI.Comm_rank(mpicomm) == 0 ? stderr : devnull
    global_logger(ConsoleLogger(logger_stream, loglevel))

    FT = Float64

    brickrange = (
        range(FT(0); length = 2, stop = FT(1)),
        range(FT(0); length = 2, stop = FT(1)),
    )
    topl = BrickTopology(mpicomm, brickrange, periodicity = (false, false))
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = ArrayType,
        polynomialorder = 1,
    )

    ω = 60
    dts = FT(60)
    finaltime = FT(2 * 3600)

    model_fast = FastODE{FT}(ω)
    model_slow = SlowODE{FT}(ω)
    model_sAlt = Alt_ODE(model_slow)

    dg_fast = DGModel(
        model_fast,
        grid,
        CentralNumericalFluxNonDiffusive(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )
    dg_slow = DGModel(
        model_slow,
        grid,
        CentralNumericalFluxNonDiffusive(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )
    dg_sAlt = DGModel(
        model_sAlt,
        grid,
        CentralNumericalFluxNonDiffusive(),
        CentralNumericalFluxDiffusive(),
        CentralNumericalFluxGradient(),
    )

    slow_method = mrrk_methods
    sAlt_method = mrrk_methods
    fast_method = mrrk_methods
    fast_dt = dts
    slow_dt = ω * fast_dt

    Qfast = init_ode_state(dg_fast, FT(0))
    Qslow = init_ode_state(dg_slow, FT(0))
    solver = MultistateMultirateRungeKutta(
        slow_method(dg_slow, Qslow; dt = slow_dt),
        sAlt_method(dg_sAlt, Qslow; dt = slow_dt),
        fast_method(dg_fast, Qfast; dt = fast_dt),
    )

   #=
    ntFreq=1
    cbcs_dg=CLIMAStateCheck.StateCheck.sccreate(
            [(Qslow,"dg Qslow"),
             (dg_slow.auxstate,"Slow aux"),
             (Qfast,"dg Qfast"),
             (dg_fast.auxstate ,"Fast auxstate"),
            ],
            ntFreq);
   =#

    Qvec = (slow = Qslow, fast = Qfast)
    solve!(Qvec, solver; timeend = finaltime)
  # cbv=(cbvector...,cbcs_dg)
  # solve!(Qvec, solver; timeend = finaltime, callbacks = cbv )
  # solve!(Qvec, solver; timeend = finaltime, callbacks = cbcs_dg )

   eng0 = norm(Qslow)
   @info @sprintf """Final
    norm(Qslow) = %.16e
    ArrayType = %s""" eng0 ArrayType

   eng1 = norm(Qfast)
   @info @sprintf """Final
    norm(Qfast) = %.16e
    ArrayType = %s""" eng1 ArrayType

   eng2 = norm( dg_fast.auxstate )
   @info @sprintf """Final
    norm(U.cum) = %.16e
    ArrayType = %s""" eng2 ArrayType

   eng3 = norm( dg_slow.auxstate )
   @info @sprintf """Final
    norm(Del.u) = %.16e
    ArrayType = %s""" eng3 ArrayType

end
