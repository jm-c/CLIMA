using Dates
using LinearAlgebra
using Logging
using MPI
using Printf
using StaticArrays
using Statistics
using Test
import GaussQuadrature
using KernelAbstractions

using ClimateMachine
ClimateMachine.init()
using ClimateMachine.ConfigTypes
using ClimateMachine.Atmos
using ClimateMachine.Atmos: vars_state
using ClimateMachine.Orientations
using ClimateMachine.DGMethods
using ClimateMachine.DGMethods.NumericalFluxes
using ClimateMachine.Mesh.Topologies
using ClimateMachine.Mesh.Grids
using ClimateMachine.Mesh.Geometry
using ClimateMachine.Mesh.Interpolation
using ClimateMachine.Thermodynamics
using ClimateMachine.TurbulenceClosures
using ClimateMachine.MPIStateArrays
using ClimateMachine.ODESolvers
using ClimateMachine.VariableTemplates
using ClimateMachine.Writers

using CLIMAParameters
using CLIMAParameters.Planet: R_d, planet_radius, grav, MSLP
struct EarthParameterSet <: AbstractEarthParameterSet end
const param_set = EarthParameterSet()

#-------------------------------------
fcn(x, y, z) = sin(x) * cos(y) * cos(z) # sample function
#-------------------------------------
function Initialize_Brick_Interpolation_Test!(
    problem,
    bl,
    state::Vars,
    aux::Vars,
    localgeo,
    t,
)
    FT = eltype(state)
    # Dummy variables for initial condition function
    state.ρ = FT(0)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = FT(0)
    state.moisture.ρq_tot = FT(0)
end
#------------------------------------------------
struct TestSphereSetup{DT}
    p_ground::DT
    T_initial::DT
    domain_height::DT

    function TestSphereSetup(
        p_ground::DT,
        T_initial::DT,
        domain_height::DT,
    ) where {DT <: AbstractFloat}
        return new{DT}(p_ground, T_initial, domain_height)
    end
end
#----------------------------------------------------------------------------
function (setup::TestSphereSetup)(problem, bl, state, aux, coords, t)
    # callable to set initial conditions
    FT = eltype(state)
    _grav::FT = grav(param_set)
    _R_d::FT = R_d(param_set)

    z = altitude(bl, aux)

    scale_height::FT = _R_d * setup.T_initial / _grav
    p::FT = setup.p_ground * exp(-z / scale_height)
    e_int = internal_energy(bl.param_set, setup.T_initial)
    e_pot = gravitational_potential(bl.orientation, aux)

    # TODO: Fix type instability: typeof(setup.T_initial) == typeof(p) fails
    state.ρ = air_density(bl.param_set, FT(setup.T_initial), p)
    state.ρu = SVector{3, FT}(0, 0, 0)
    state.ρe = state.ρ * (e_int + e_pot)
    return nothing
end
#----------------------------------------------------------------------------
function run_brick_interpolation_test(
    ::Type{DA},
    ::Type{FT},
    polynomialorders,
    toler::FT,
) where {DA, FT <: AbstractFloat}
    mpicomm = MPI.COMM_WORLD
    root = 0
    pid = MPI.Comm_rank(mpicomm)
    npr = MPI.Comm_size(mpicomm)

    xmin, ymin, zmin = FT(0), FT(0), FT(0)         # defining domain extent
    xmax, ymax, zmax = FT(2000), FT(400), FT(2000)
    xres = [FT(10), FT(10), FT(10)] # resolution of interpolation grid

    Ne = (20, 4, 20)
    #-------------------------
    _x, _y, _z = ClimateMachine.Mesh.Grids.vgeoid.x1id,
    ClimateMachine.Mesh.Grids.vgeoid.x2id,
    ClimateMachine.Mesh.Grids.vgeoid.x3id
    #-------------------------
    brickrange = (
        range(FT(xmin); length = Ne[1] + 1, stop = xmax),
        range(FT(ymin); length = Ne[2] + 1, stop = ymax),
        range(FT(zmin); length = Ne[3] + 1, stop = zmax),
    )
    topl = StackedBrickTopology(
        mpicomm,
        brickrange,
        periodicity = (true, true, false),
    )
    grid = DiscontinuousSpectralElementGrid(
        topl,
        FloatType = FT,
        DeviceArray = DA,
        polynomialorder = polynomialorders,
    )
    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = Initialize_Brick_Interpolation_Test!,
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        source = (Gravity(),),
    )

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )


    Q = init_ode_state(dg, FT(0))

    #------------------------------
    x1 = @view grid.vgeo[:, _x:_x, :]
    x2 = @view grid.vgeo[:, _y:_y, :]
    x3 = @view grid.vgeo[:, _z:_z, :]
    #----calling interpolation function on state variable # st_idx--------------------------
    nvars = size(Q.data, 2)
    Q.data .= sin.(x1 ./ xmax) .* cos.(x2 ./ ymax) .* cos.(x3 ./ zmax)

    xbnd = Array{FT}(undef, 2, 3)

    xbnd[1, 1] = FT(xmin)
    xbnd[2, 1] = FT(xmax)
    xbnd[1, 2] = FT(ymin)
    xbnd[2, 2] = FT(ymax)
    xbnd[1, 3] = FT(zmin)
    xbnd[2, 3] = FT(zmax)
    #----------------------------------------------------------
    x1g = collect(range(xbnd[1, 1], xbnd[2, 1], step = xres[1]))
    nx1 = length(x1g)
    x2g = collect(range(xbnd[1, 2], xbnd[2, 2], step = xres[2]))
    nx2 = length(x2g)
    x3g = collect(range(xbnd[1, 3], xbnd[2, 3], step = xres[3]))
    nx3 = length(x3g)

    filename = "test.nc"
    varnames = ("ρ", "ρu", "ρv", "ρw", "e", "other")

    intrp_brck = InterpolationBrick(grid, xbnd, x1g, x2g, x3g)        # sets up the interpolation structure
    iv = DA(Array{FT}(undef, intrp_brck.Npl, nvars))                  # allocating space for the interpolation variable
    if pid == 0
        fiv = DA(Array{FT}(undef, nx1, nx2, nx3, nvars))    # allocating space for the full interpolation variables accumulated on proc# 0
    else
        fiv = DA(Array{FT}(undef, 0, 0, 0, 0))
    end
    interpolate_local!(intrp_brck, Q.data, iv)                    # interpolation

    accumulate_interpolated_data!(intrp_brck, iv, fiv)      # write interpolation data to file
    #------------------------------
    err_inf_dom = zeros(FT, nvars)

    x1g = intrp_brck.x1g
    x2g = intrp_brck.x2g
    x3g = intrp_brck.x3g

    if pid == 0
        nx1 = length(x1g)
        nx2 = length(x2g)
        nx3 = length(x3g)
        x1 = Array{FT}(undef, nx1, nx2, nx3)
        x2 = similar(x1)
        x3 = similar(x1)

        fiv_cpu = Array(fiv)

        for k in 1:nx3, j in 1:nx2, i in 1:nx1
            x1[i, j, k] = x1g[i]
            x2[i, j, k] = x2g[j]
            x3[i, j, k] = x3g[k]
        end

        fex = sin.(x1 ./ xmax) .* cos.(x2 ./ ymax) .* cos.(x3 ./ zmax)

        for vari in 1:nvars
            err_inf_dom[vari] =
                maximum(abs.(fiv_cpu[:, :, :, vari] .- fex[:, :, :]))
        end
    end

    MPI.Bcast!(err_inf_dom, root, mpicomm)

    if maximum(err_inf_dom) > toler
        if pid == 0
            println("err_inf_domain = $(maximum(err_inf_dom)) is larger than prescribed tolerance of $toler")
        end
        MPI.Barrier(mpicomm)
    end
    @test maximum(err_inf_dom) < toler
    return nothing
    #----------------
end #function run_brick_interpolation_test
#----------------------------------------------------------------------------

#----------------------------------------------------------------------------
# Cubed sphere, lat/long interpolation test
#----------------------------------------------------------------------------
function run_cubed_sphere_interpolation_test(
    ::Type{DA},
    ::Type{FT},
    polynomialorders,
    toler::FT,
) where {DA, FT <: AbstractFloat}
    mpicomm = MPI.COMM_WORLD
    root = 0
    pid = MPI.Comm_rank(mpicomm)
    npr = MPI.Comm_size(mpicomm)

    domain_height = FT(30e3)
    numelem_horz = 6
    numelem_vert = 4
    #-------------------------
    _x, _y, _z = ClimateMachine.Mesh.Grids.vgeoid.x1id,
    ClimateMachine.Mesh.Grids.vgeoid.x2id,
    ClimateMachine.Mesh.Grids.vgeoid.x3id
    _ρ, _ρu, _ρv, _ρw = 1, 2, 3, 4
    #-------------------------
    _planet_radius::FT = planet_radius(param_set)

    vert_range = grid1d(
        _planet_radius,
        FT(_planet_radius + domain_height),
        nelem = numelem_vert,
    )

    lat_res = FT(1) # 1 degree resolution
    long_res = FT(1) # 1 degree resolution
    nel_vert_grd = 20 #100 #50 #10#50
    rad_res = FT((vert_range[end] - vert_range[1]) / FT(nel_vert_grd)) # 1000 m vertical resolution
    #----------------------------------------------------------
    _MSLP::FT = MSLP(param_set)
    setup = TestSphereSetup(_MSLP, FT(255), FT(30e3))

    topology = StackedCubedSphereTopology(mpicomm, numelem_horz, vert_range)

    grid = DiscontinuousSpectralElementGrid(
        topology,
        FloatType = FT,
        DeviceArray = DA,
        polynomialorder = polynomialorders,
        meshwarp = ClimateMachine.Mesh.Topologies.cubedshellwarp,
    )

    model = AtmosModel{FT}(
        AtmosLESConfigType,
        param_set;
        init_state_prognostic = setup,
        orientation = SphericalOrientation(),
        ref_state = NoReferenceState(),
        turbulence = ConstantDynamicViscosity(FT(0)),
        moisture = DryModel(),
        source = (),
    )

    dg = DGModel(
        model,
        grid,
        RusanovNumericalFlux(),
        CentralNumericalFluxSecondOrder(),
        CentralNumericalFluxGradient(),
    )

    Q = init_ode_state(dg, FT(0))

    #------------------------------
    x1 = @view grid.vgeo[:, _x:_x, :]
    x2 = @view grid.vgeo[:, _y:_y, :]
    x3 = @view grid.vgeo[:, _z:_z, :]

    xmax = _planet_radius
    ymax = _planet_radius
    zmax = _planet_radius

    nvars = size(Q.data, 2)

    Q.data .= sin.(x1 ./ xmax) .* cos.(x2 ./ ymax) .* cos.(x3 ./ zmax)
    #for ivar in 1:nvars
    #    Q.data[:, ivar, :] .=
    #        sin.(x1[:, 1, :] ./ xmax) .* cos.(x2[:, 1, :] ./ ymax) .*
    #        cos.(x3[:, 1, :] ./ zmax)
    #end
    #------------------------------
    lat_min, lat_max = FT(-90.0), FT(90.0)            # inclination/zeinth angle range
    long_min, long_max = FT(-180.0), FT(180.0)     # azimuthal angle range
    rad_min, rad_max = vert_range[1], vert_range[end] # radius range


    lat_grd = collect(range(lat_min, lat_max, step = lat_res))
    n_lat = length(lat_grd)
    long_grd = collect(range(long_min, long_max, step = long_res))
    n_long = length(long_grd)
    rad_grd = collect(range(rad_min, rad_max, step = rad_res))
    n_rad = length(rad_grd)

    _ρu, _ρv, _ρw = 2, 3, 4

    filename = "test.nc"
    varnames = ("ρ", "ρu", "ρv", "ρw", "e")
    projectv = true

    intrp_cs = InterpolationCubedSphere(
        grid,
        collect(vert_range),
        numelem_horz,
        lat_grd,
        long_grd,
        rad_grd,
    ) # sets up the interpolation structure
    iv = DA(Array{FT}(undef, intrp_cs.Npl, nvars))             # allocating space for the interpolation variable
    if pid == 0
        fiv = DA(Array{FT}(undef, n_long, n_lat, n_rad, nvars))    # allocating space for the full interpolation variables accumulated on proc# 0
    else
        fiv = DA(Array{FT}(undef, 0, 0, 0, 0))
    end

    interpolate_local!(intrp_cs, Q.data, iv)                   # interpolation
    project_cubed_sphere!(intrp_cs, iv, (_ρu, _ρv, _ρw))         # project velocity onto unit vectors along rad, lat & long
    accumulate_interpolated_data!(intrp_cs, iv, fiv)           # accumulate interpolated data on to proc# 0
    #----------------------------------------------------------
    # Testing
    err_inf_dom = zeros(FT, nvars)
    rad = Array(intrp_cs.rad_grd)
    lat = Array(intrp_cs.lat_grd)
    long = Array(intrp_cs.long_grd)
    fiv_cpu = Array(fiv)
    if pid == 0
        nrad = length(rad)
        nlat = length(lat)
        nlong = length(long)
        x1g = Array{FT}(undef, nrad, nlat, nlong)
        x2g = similar(x1g)
        x3g = similar(x1g)

        fex = zeros(FT, nlong, nlat, nrad, nvars)

        for vari in 1:nvars
            for i in 1:nlong, j in 1:nlat, k in 1:nrad
                x1g_ijk = rad[k] * cosd(lat[j]) * cosd(long[i]) # inclination -> latitude; azimuthal -> longitude.
                x2g_ijk = rad[k] * cosd(lat[j]) * sind(long[i]) # inclination -> latitude; azimuthal -> longitude.
                x3g_ijk = rad[k] * sind(lat[j])

                fex[i, j, k, vari] =
                    fcn(x1g_ijk / xmax, x2g_ijk / ymax, x3g_ijk / zmax)
            end
        end

        if projectv
            for i in 1:nlong, j in 1:nlat, k in 1:nrad
                fex[i, j, k, _ρu] =
                    -fex[i, j, k, _ρ] * sind(long[i]) +
                    fex[i, j, k, _ρ] * cosd(long[i])

                fex[i, j, k, _ρv] =
                    -fex[i, j, k, _ρ] * sind(lat[j]) * cosd(long[i]) -
                    fex[i, j, k, _ρ] * sind(lat[j]) * sind(long[i]) +
                    fex[i, j, k, _ρ] * cosd(lat[j])

                fex[i, j, k, _ρw] =
                    fex[i, j, k, _ρ] * cosd(lat[j]) * cosd(long[i]) +
                    fex[i, j, k, _ρ] * cosd(lat[j]) * sind(long[i]) +
                    fex[i, j, k, _ρ] * sind(lat[j])
            end
        end

        for vari in 1:nvars
            err_inf_dom[vari] =
                maximum(abs.(fiv_cpu[:, :, :, vari] .- fex[:, :, :, vari]))
        end
    end

    MPI.Bcast!(err_inf_dom, root, mpicomm)

    if maximum(err_inf_dom) > toler
        if pid == 0
            println("err_inf_domain = $(maximum(err_inf_dom)) is larger than prescribed tolerance of $toler")
        end
        MPI.Barrier(mpicomm)
    end
    @test maximum(err_inf_dom) < toler
    return nothing
end
#----------------------------------------------------------------------------
@testset "Interpolation tests" begin
    DA = ClimateMachine.array_type()

    run_brick_interpolation_test(DA, Float32, (0), Float32(1E-1))
    run_brick_interpolation_test(DA, Float64, (0), Float64(1E-1))

    run_brick_interpolation_test(DA, Float32, (5), Float32(1E-6))
    run_brick_interpolation_test(DA, Float64, (5), Float64(1E-9))

    run_brick_interpolation_test(DA, Float32, (5, 6), Float32(1E-6))
    run_brick_interpolation_test(DA, Float64, (5, 6), Float64(1E-9))

    run_cubed_sphere_interpolation_test(DA, Float32, (0), Float32(2e-1))
    run_cubed_sphere_interpolation_test(DA, Float64, (0), Float64(2e-1))

    run_cubed_sphere_interpolation_test(DA, Float32, (5), Float32(2e-6))
    run_cubed_sphere_interpolation_test(DA, Float64, (5), Float64(2e-7))

    run_cubed_sphere_interpolation_test(DA, Float32, (5, 6), Float32(2e-6))
    run_cubed_sphere_interpolation_test(DA, Float64, (5, 6), Float64(2e-7))
end
#------------------------------------------------
