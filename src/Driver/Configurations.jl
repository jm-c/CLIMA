"""
    CLIMA driver configurations

Use to run CLIMA using the CLIMA driver's `CLIMA.invoke()`. User-customized
configurations can use these as templates.
"""

using MPI

using ..Atmos
using ..HydrostaticBoussinesq
using ..DGmethods
using ..DGmethods.NumericalFluxes
using ..Mesh.Grids
using ..Mesh.Interpolation
using ..Mesh.Topologies
using ..ODESolvers
using ..PlanetParameters

abstract type AbstractSolverType end
struct ExplicitSolverType <: AbstractSolverType
    solver_method::Function
    ExplicitSolverType(;solver_method=LSRK54CarpenterKennedy) = new(solver_method)
end
struct IMEXSolverType <: AbstractSolverType
    linear_model::Type
    linear_solver::Type
    solver_method::Function
    function IMEXSolverType(;linear_model=AtmosAcousticGravityLinearModel,
                            linear_solver=ManyColumnLU,
                            solver_method=ARK2GiraldoKellyConstantinescu)
        return new(linear_model, linear_solver, solver_method)
    end
end
DefaultSolverType = IMEXSolverType

struct DriverConfiguration{FT}
    name::String
    N::Int
    array_type
    solver_type::AbstractSolverType
    #
    # AtmosModel details
    bl::BalanceLaw
    #
    # execution details
    mpicomm::MPI.Comm
    #
    # mesh details
    grid::DiscontinuousSpectralElementGrid
    interp::Union{Nothing,InterpolationEntity}
    #
    # DGModel details
    numfluxnondiff::NumericalFluxNonDiffusive
    numfluxdiff::NumericalFluxDiffusive
    gradnumflux::NumericalFluxGradient

    function DriverConfiguration(name::String, N::Int, FT, array_type,
                                 solver_type::AbstractSolverType,
                                 bl::BalanceLaw,
                                 mpicomm::MPI.Comm,
                                 grid::DiscontinuousSpectralElementGrid,
                                 interp::InterpolationEntity,
                                 numfluxnondiff::NumericalFluxNonDiffusive,
                                 numfluxdiff::NumericalFluxDiffusive,
                                 gradnumflux::NumericalFluxGradient)
        return new{FT}(name, N, array_type, solver_type, bl, mpicomm, grid,
                       interp, numfluxnondiff, numfluxdiff, gradnumflux)
    end
end

function Atmos_LES_Configuration(
        name::String,
        N::Int,
        (Δx, Δy, Δz)::NTuple{3,FT},
        xmax::Int, ymax::Int, zmax::Int,
        init_LES!;
        xmin           = 0,
        ymin           = 0,
        zmin           = 0,
        array_type     = CLIMA.array_type(),
        solver_type    = IMEXSolverType(linear_solver=SingleColumnLU),
        model          = AtmosModel{FT}(AtmosLESConfiguration;
                                        init_state=init_LES!),
        mpicomm        = MPI.COMM_WORLD,
        boundary       = ((0,0), (0,0), (1,2)),
        periodicity    = (true, true, false),
        meshwarp       = (x...)->identity(x),
        idom_bnd       = nothing,
        idom_res       = nothing,
        numfluxnondiff = Rusanov(),
        numfluxdiff    = CentralNumericalFluxDiffusive(),
        gradnumflux    = CentralNumericalFluxGradient()
    ) where {FT<:AbstractFloat}

    @info @sprintf("""Establishing Atmos LES configuration for %s
                   precision        = %s
                   polynomial order = %d
                   grid             = %dx%dx%d
                   resolution       = %dx%dx%d
                   MPI ranks        = %d""",
                   name, FT, N,
                   xmax, ymax, zmax,
                   Δx, Δy, Δz,
                   MPI.Comm_size(mpicomm))

    brickrange = (grid1d(xmin, xmax, elemsize=Δx*N),
                  grid1d(ymin, ymax, elemsize=Δy*N),
                  grid1d(zmin, zmax, elemsize=Δz*N))
    topology = StackedBrickTopology(mpicomm, brickrange,
                                    periodicity=periodicity,
                                    boundary=boundary)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType=FT,
                                            DeviceArray=array_type,
                                            polynomialorder=N,
                                            meshwarp=meshwarp)

    if idom_bnd !== nothing && idom_res !== nothing
        interp = InterpolationBrick(grid, idom_bnd, idom_res)
    else
        interp = nothing
    end
    return DriverConfiguration(name, N, FT, array_type, solver_type, model,
                               mpicomm, grid, interp,
                               numfluxnondiff, numfluxdiff, gradnumflux)
end

function Atmos_GCM_Configuration(
        name::String,
        N::Int,
        (nelem_horz, nelem_vert)::NTuple{2,Int},
        domain_height::FT,
        init_GCM!;
        array_type         = CLIMA.array_type(),
        solver_type        = DefaultSolverType(),
        model              = AtmosModel{FT}(AtmosGCMConfiguration;
                                             init_state=init_GCM!),
        mpicomm            = MPI.COMM_WORLD,
        meshwarp::Function = cubedshellwarp,
        lat_res            = FT(10.0 * π / 180.0),
        long_res           = FT( 10.0 * π / 180.0),
        nel_vert_grd       = -1,
        numfluxnondiff     = Rusanov(),
        numfluxdiff        = CentralNumericalFluxDiffusive(),
        gradnumflux        = CentralNumericalFluxGradient()
    ) where {FT<:AbstractFloat}
    @info @sprintf("""Establishing Atmos GCM configuration for %s
                   precision        = %s
                   polynomial order = %d
                   #horiz elems     = %d
                   #vert_elems      = %d
                   domain height    = %.2e
                   MPI ranks        = %d""",
                   name, FT, N,
                   nelem_horz, nelem_vert, domain_height,
                   MPI.Comm_size(mpicomm))

    vert_range = grid1d(FT(planet_radius), FT(planet_radius+domain_height), nelem=nelem_vert)

    topology = StackedCubedSphereTopology(mpicomm, nelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType=FT,
                                            DeviceArray=array_type,
                                            polynomialorder=N,
                                            meshwarp=meshwarp)

    if nel_vert_grd > 0
        nhor = trunc(Int64, √( length(topology.elems) / topology.stacksize / 6))
        rad_res = FT((vert_range[end] - vert_range[1]) / FT(nel_vert_grd))
        interp = InterpolationCubedSphere(grid, vert_range, nhor, lat_res, long_res, rad_res)
    else
        interp = nothing
    end
    return DriverConfiguration(name, N, FT, array_type, solver_type, model,
                               mpicomm, grid, interp,
                               numfluxnondiff, numfluxdiff, gradnumflux)
end

function Ocean_BoxGCM_Configuration(
    name::String,
    N::Int,
    (Nˣ, Nʸ, Nᶻ)::NTuple{3,Int},
    model::HydrostaticBoussinesqModel;
    FT             = Float64,
    array_type     = CLIMA.array_type(),
    solver_type    = ExplicitSolverType(solver_method=LSRK144NiegemannDiehlBusch),
    mpicomm        = MPI.COMM_WORLD,
    numfluxnondiff = Rusanov(),
    numfluxdiff    = CentralNumericalFluxDiffusive(),
    gradnumflux    = CentralNumericalFluxGradient(),
    periodicity    = (false, false, false),
    boundary       = ((1, 1), (1, 1), (2, 3))
    )

    brickrange = (range(FT(0);  length=Nˣ+1, stop=model.problem.Lˣ),
                  range(FT(0);  length=Nʸ+1, stop=model.problem.Lʸ),
                  range(FT(-model.problem.H); length=Nᶻ+1, stop=0))

    topology = StackedBrickTopology(mpicomm, brickrange;
                                    periodicity = periodicity,
                                    boundary = boundary)

    grid = DiscontinuousSpectralElementGrid(topology,
                                            FloatType = FT,
                                            DeviceArray = array_type,
                                            polynomialorder = N)

    return DriverConfiguration(name, N, FT, array_type, solver_type, model,
                               mpicomm, grid, numfluxnondiff, numfluxdiff,
                               gradnumflux)
end
