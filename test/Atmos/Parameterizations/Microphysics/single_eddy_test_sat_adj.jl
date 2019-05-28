# The test is based on a modelling set-up designed for the
# 8th International Cloud Modelling Workshop
# (ICMW, Muhlbauer et al., 2013, case 1, doi:10.1175/BAMS-D-12-00188.1)
#
# See chapter 2 in Arabas et al 2015 for setup details:
#@Article{gmd-8-1677-2015,
#AUTHOR = {Arabas, S. and Jaruga, A. and Pawlowska, H. and Grabowski, W. W.},
#TITLE = {libcloudph++ 1.0: a single-moment bulk, double-moment bulk, and particle-based warm-rain microphysics library in C++},
#JOURNAL = {Geoscientific Model Development},
#VOLUME = {8},
#YEAR = {2015},
#NUMBER = {6},
#PAGES = {1677--1707},
#URL = {https://www.geosci-model-dev.net/8/1677/2015/},
#DOI = {10.5194/gmd-8-1677-2015}
#}

using MPI
using CLIMA.Topologies
using CLIMA.Grids
using CLIMA.DGBalanceLawDiscretizations
using CLIMA.DGBalanceLawDiscretizations.NumericalFluxes
using CLIMA.MPIStateArrays
using CLIMA.LowStorageRungeKuttaMethod
using CLIMA.ODESolvers
using CLIMA.GenericCallbacks
using CLIMA.Vtk

using LinearAlgebra
using StaticArrays
using Printf

using CLIMA.PlanetParameters
using CLIMA.MoistThermodynamics
using CLIMA.Microphysics

@static if Base.find_package("CuArrays") !== nothing
  using CUDAdrv
  using CUDAnative
  using CuArrays
  const ArrayTypes = VERSION >= v"1.2-pre.25" ? (Array, CuArray) : (Array,)
else
  const ArrayTypes = (Array, )
end

const _nstate = 5
const _ρ, _ρu, _ρw, _ρet, _ρqt = 1:_nstate
const stateid = (ρid = _ρ, ρuid = _ρu, ρwid = _ρw, ρetid = _ρet, ρqtid = _ρqt)
const statenames = ("ρ", "ρu", "ρw", "ρet", "ρqt")

const _nauxcstate = 3
const _c_z, _c_x, _c_p = 1:_nauxcstate


# preflux computation for wavespeed function
@inline function preflux(Q, _...)
  @inbounds begin
    # unpack all the state variables
    ρ, ρu, ρw, ρet, ρqt = Q[_ρ], Q[_ρu], Q[_ρw], Q[_ρet], Q[_ρqt]
    u, w, et, qt = ρu / ρ, ρw / ρ, ρet / ρ, ρqt / ρ

    (u, w, ρ, qt, et)
  end
end


# boundary condition
@inline function bcstate!(QP, VFP, auxP, nM, QM, VFM, auxM, bctype, t,
                          u, w, ρ, qt, et)
  @inbounds begin
    ρu_M, ρw_M, ρet_M, ρqt_M = QM[_ρu], QM[_ρw], QM[_ρet], QM[_ρqt]

    ρu_nM = nM[1] * ρu_M + nM[2] * ρw_M

    QP[_ρu] = ρu_M - 2 * nM[1] * ρu_nM
    QP[_ρw] = ρw_M - 2 * nM[2] * ρu_nM

    QP[_ρet], QP[_ρqt] = ρet_M, ρqt_M

    auxM .= auxP

    # Required return from this function is either nothing
    # or preflux with plus state as arguments
    preflux(QP, auxP, t)
  end
end


# max eigenvalue
@inline function wavespeed(n, Q, aux, t, u, w, ρ, qt, et)
  @inbounds abs(n[1] * u + n[2] * w)
end


@inline function constant_auxiliary_init!(aux, x, z, _...)
  @inbounds begin
    aux[_c_z] = z  # for gravity
    aux[_c_x] = x  # tmp for printing

    DFloat = eltype(aux)

    # initial condition
    θ_0::DFloat    = 289         # K
    p_0::DFloat    = 101500      # Pa
    p_1000::DFloat = 100000      # Pa
    qt_0::DFloat   = 7.5 * 1e-3  # kg/kg
    z_0::DFloat    = 0           # m

    R_m, cp_m, cv_m, γ = moist_gas_constants(PhasePartition(qt_0))

    # Pressure profile assuming hydrostatic and constant θ and qt profiles.
    # It is done this way to be consistent with Arabas paper.
    # It's not neccesarily the best way to initialize with our model variables.
    p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
                R_d / cp_d * grav / θ_0 / R_m * (z - z_0)
               )^(cp_d / R_d)

    aux[_c_p] = p  # for prescribed pressure gradient (kinematic setup)
  end
end


# physical flux function
@inline function eulerflux!(F, Q, QV, aux, t, u, w, ρ, qt, et)
  @inbounds begin
    p = aux[_c_p]

    F .= 0
    # advect the moisture and energy
    F[1, _ρqt], F[2, _ρqt] = u *  ρ * qt,      w *  ρ * qt
    F[1, _ρet], F[2, _ρet] = u * (ρ * et + p), w * (ρ * et + p)
    # don't advect momentum (kinematic setup)
  end
end


# initial condition
const w_max = .6    # m/s
const Z_max = 1500. # m
const X_max = 1500. # m

@inline function single_eddy!(Q, t, x, z, _...)
  DFloat = eltype(Q)

  # initial condition
  θ_0::DFloat    = 289         # K
  p_0::DFloat    = 101500      # Pa
  p_1000::DFloat = 100000      # Pa
  qt_0::DFloat   = 7.5 * 1e-3  # kg/kg
  z_0::DFloat    = 0           # m

  R_m, cp_m, cv_m, γ = moist_gas_constants(PhasePartition(qt_0))

  @inbounds begin
    # Pressure profile assuming hydrostatic and constant θ and qt profiles.
    # It is done this way to be consistent with Arabas paper.
    # It's not neccesarily the best way to initialize with our model variables.
    p = p_1000 * ((p_0 / p_1000)^(R_d / cp_d) -
                R_d / cp_d * grav / θ_0 / R_m * (z - z_0)
               )^(cp_d / R_d)
    T::DFloat = θ_0 * exner(p, PhasePartition(qt_0))
    ρ::DFloat = p / R_m / T

    # TODO should this be more "grid aware"?
    # the velocity is calculated as derivative of streamfunction
    ρu::DFloat = w_max * X_max/Z_max * cos(π * z/Z_max) * cos(2*π * x/X_max)
    ρw::DFloat = 2*w_max * sin(π * z/Z_max) * sin(2*π * x/X_max)
    u = ρu / ρ
    w = ρw / ρ

    ρqt::DFloat = ρ * qt_0

    ei  = internal_energy(T, PhasePartition(qt_0))
    ρet = ρ * (grav * z + (1//2)*(u^2 + w^2) + ei)

    Q[_ρ], Q[_ρu], Q[_ρw], Q[_ρet], Q[_ρqt] = ρ, ρu, ρw, ρet, ρqt
  end
end


function main(mpicomm, DFloat, topl::AbstractTopology{dim}, N, timeend,
              ArrayType, dt) where {dim}

  grid = DiscontinuousSpectralElementGrid(topl,
                                          FloatType = DFloat,
                                          DeviceArray = ArrayType,
                                          polynomialorder = N,
                                         )
  numflux!(x...) = NumericalFluxes.rusanov!(x...,
                                            eulerflux!,
                                            wavespeed,
                                            preflux
                                           )
  numbcflux!(x...) = NumericalFluxes.rusanov_boundary_flux!(x...,
                                                            eulerflux!,
                                                            bcstate!,
                                                            wavespeed,
                                                            preflux
                                                           )



  # spacedisc = data needed for evaluating the right-hand side function
  spacedisc = DGBalanceLaw(grid = grid,
                           length_state_vector = _nstate,
                           flux! = eulerflux!,
                           numerical_flux! = numflux!,
                           numerical_boundary_flux! = numbcflux!,
                           auxiliary_state_length = _nauxcstate,
                           auxiliary_state_initialization! =
                             constant_auxiliary_init!)

  # This is a actual state/function that lives on the grid
  initialcondition(Q, x...) = single_eddy!(Q, DFloat(0), x...)
  Q = MPIStateArray(spacedisc, initialcondition)

  npoststates = 10
  v_ql, v_qi, v_qv, v_qt, v_p, v_T, v_et, v_ei, v_ek, v_ep = 1:npoststates
  postnames = ("ql", "qi", "qv","qt", "p", "T",
               "e_tot", "e_int", "e_kin", "e_pot")
  postprocessarray = MPIStateArray(spacedisc; nstate=npoststates)

  writevtk("initial_condition", Q, spacedisc, statenames)

  lsrk = LowStorageRungeKutta(spacedisc, Q; dt = dt, t0 = 0)

  io = MPI.Comm_rank(mpicomm) == 0 ? stdout : devnull
  eng0 = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ (initial) =  %.16e\n", eng0)

  # Set up the information callback
  timer = [time_ns()]
  cbinfo = GenericCallbacks.EveryXWallTimeSeconds(10, mpicomm) do (s=false)
    if s
      timer[1] = time_ns()
    else
      run_time = (time_ns() - timer[1]) * 1e-9
      (min, sec) = fldmod(run_time, 60)
      (hrs, min) = fldmod(min, 60)
      @printf(io, "----\n")
      @printf(io, "simtime =  %.16e\n", ODESolvers.gettime(lsrk))
      @printf(io, "runtime =  %03d:%02d:%05.2f (hour:min:sec)\n", hrs, min, sec)
      @printf(io, "||Q||₂  =  %.16e\n", norm(Q))
    end
    nothing
  end

  step = [0]
  mkpath("vtk")

  cbvtk = GenericCallbacks.EveryXSimulationSteps(60) do (init=false)
    DGBalanceLawDiscretizations.dof_iteration!(postprocessarray, spacedisc,
                                               Q) do R, Q, QV, aux
      @inbounds begin
        # TODO - how to get preflux(Q) here?
        ρ, ρet, ρu, ρw, ρqt = Q[_ρ], Q[_ρet], Q[_ρu], Q[_ρw], Q[_ρqt]
        u, w, qt, et = ρu / ρ, ρw / ρ, ρqt / ρ, ρet / ρ
        z = aux[_c_z]
        p = aux[_c_p]

        ei = et - 1//2 * (u^2 + w^2) - grav * z
        ts = PhaseEquil(ei, qt, ρ)  # saturation adjustment happens here
        pp = PhasePartition(ts)
        R[v_T] = ts.T #air_temperature(e_int, pp)
        R[v_p] = p

        R[v_qt] = qt
        R[v_qv] = qt - pp.liq - pp.ice
        R[v_ql] = pp.liq
        R[v_qi] = pp.ice

        R[v_et] = et
        R[v_ei] = ei
        R[v_ek] = 1//2 * (u^2 + w^2)
        R[v_ep] = grav * z

      end
    end

    outprefix = @sprintf("vtk/eddy_sat_adj_%dD_mpirank%04d_step%04d",
                         dim, MPI.Comm_rank(mpicomm), step[1])
    @printf(io, "----\n")
    @printf(io, "doing VTK output =  %s\n", outprefix)
    writevtk(outprefix, Q, spacedisc, statenames, postprocessarray, postnames)
    step[1] += 1
    nothing
  end

  solve!(Q, lsrk; timeend=timeend, callbacks=(cbinfo, cbvtk))

  Qe = MPIStateArray(spacedisc,
                    (Q, x...) -> single_eddy!(Q, DFloat(timeend), x...))

  # Print some end of the simulation information
  engf = norm(Q)
  @printf(io, "----\n")
  @printf(io, "||Q||₂ ( final ) = %.16e\n", engf)
end

function run(dim, Ne, N, timeend, DFloat)

  #ArrayType = CuArray
  ArrayType = Array

  MPI.Initialized() || MPI.Init()
  Sys.iswindows() || (isinteractive() && MPI.finalize_atexit())

  mpicomm = MPI.COMM_WORLD

  brickrange = ntuple(j->range(DFloat(0); length=Ne[j]+1, stop=Z_max), 2)

  topl = BrickTopology(mpicomm, brickrange, periodicity=(true, false))
  dt = 1

  main(mpicomm, DFloat, topl, N, timeend, ArrayType, dt)

end

using Test
let
  timeend = 15 * 60 # TODO 30 * 60
  numelem = (75, 75)
  lvls = 3
  dim = 2
  DFloat = Float64

  polynomialorder = 4

  run(dim, ntuple(j->numelem[j], dim), polynomialorder, timeend, DFloat)
end

isinteractive() || MPI.Finalize()

nothing
