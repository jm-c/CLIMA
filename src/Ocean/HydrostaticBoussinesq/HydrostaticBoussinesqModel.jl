module HydrostaticBoussinesq

export HydrostaticBoussinesqModel, HydrostaticBoussinesqProblem, OceanDGModel,
       LinearHBModel, calculate_dt

using StaticArrays
using LinearAlgebra: I, dot, Diagonal
using ..VariableTemplates
using ..MPIStateArrays
using ..DGmethods: init_ode_state
using ..PlanetParameters: grav
using ..Mesh.Filters: CutoffFilter, apply!, ExponentialFilter
using ..Mesh.Grids: polynomialorder, VerticalDirection, HorizontalDirection, min_node_distance

using ..DGmethods.NumericalFluxes: Rusanov, CentralNumericalFluxGradient,
                                   CentralNumericalFluxDiffusive,
                                   CentralNumericalFluxNonDiffusive

import ..DGmethods.NumericalFluxes: update_penalty!, numerical_flux_diffusive!,
                                    NumericalFluxNonDiffusive

import ..DGmethods: BalanceLaw, vars_aux, vars_state, vars_gradient,
                    vars_diffusive, vars_integrals, flux_nondiffusive!,
                    flux_diffusive!, source!, wavespeed,
                    boundary_state!, update_aux!, update_aux_diffusive!,
                    gradvariables!, init_aux!, init_state!,
                    LocalGeometry, indefinite_stack_integral!,
                    reverse_indefinite_stack_integral!, integrate_aux!,
                    DGModel, nodal_update_aux!, diffusive!,
                    copy_stack_field_down!, create_state, calculate_dt

×(a::SVector, b::SVector) = StaticArrays.cross(a, b)
∘(a::SVector, b::SVector) = StaticArrays.dot(a, b)

abstract type OceanBoundaryCondition end
struct CoastlineFreeSlip             <: OceanBoundaryCondition end
struct CoastlineNoSlip               <: OceanBoundaryCondition end
struct OceanFloorFreeSlip            <: OceanBoundaryCondition end
struct OceanFloorNoSlip              <: OceanBoundaryCondition end
struct OceanSurfaceNoStressNoForcing <: OceanBoundaryCondition end
struct OceanSurfaceStressNoForcing   <: OceanBoundaryCondition end
struct OceanSurfaceNoStressForcing   <: OceanBoundaryCondition end
struct OceanSurfaceStressForcing     <: OceanBoundaryCondition end

abstract type AbstractHydrostaticBoussinesqProblem end
struct HydrostaticBoussinesqProblem <: AbstractHydrostaticBoussinesqProblem end

struct HydrostaticBoussinesqModel{P,T} <: BalanceLaw
  problem::P
  cʰ::T
  cᶻ::T
  αᵀ::T
  νʰ::T
  νᶻ::T
  κʰ::T
  κᶻ::T
  function HydrostaticBoussinesqModel{FT}(problem;
                                      cʰ = FT(0),     # m/s
                                      cᶻ = FT(0),     # m/s
                                      αᵀ = FT(2e-4),  # (m/s)^2 / K
                                      νʰ = FT(5e3),   # m^2 / s
                                      νᶻ = FT(5e-3),  # m^2 / s
                                      κʰ = FT(1e3),   # m^2 / s
                                      κᶻ = FT(1e-4),  # m^2 / s
                                      ) where {FT <: AbstractFloat}
    return new{typeof(problem),FT}(problem, cʰ, cᶻ, αᵀ, νʰ, νᶻ, κʰ, κᶻ)
  end
end

function calculate_dt(grid, model::HydrostaticBoussinesqModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())
    minΔz = min_node_distance(grid, VerticalDirection())

    CFL_gravity = minΔx / model.cʰ
    CFL_diffusive = minΔz^2 / (1000 * model.κᶻ)
    CFL_viscous = minΔz^2 / model.νᶻ

    dt = 1//2 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

struct LinearHBModel{M} <: BalanceLaw
  ocean::M
  function LinearHBModel(ocean::M) where {M}
    return new{M}(ocean)
  end
end

function calculate_dt(grid, model::LinearHBModel, Courant_number)
    minΔx = min_node_distance(grid, HorizontalDirection())

    CFL_gravity = minΔx / model.ocean.cʰ
    CFL_diffusive = minΔx^2 / model.ocean.κʰ
    CFL_viscous = minΔx^2 / model.ocean.νʰ

    dt = 1//10 * minimum([CFL_gravity, CFL_diffusive, CFL_viscous])

    return dt
end

HBModel   = HydrostaticBoussinesqModel
HBProblem = HydrostaticBoussinesqProblem

function OceanDGModel(bl::HBModel, grid, numfluxnondiff, numfluxdiff,
                      gradnumflux; kwargs...)
  vert_filter = CutoffFilter(grid, polynomialorder(grid)-1)
  exp_filter  = ExponentialFilter(grid, 1, 8)

  modeldata = (vert_filter = vert_filter, exp_filter=exp_filter)

  return DGModel(bl, grid, numfluxnondiff, numfluxdiff, gradnumflux;
                 kwargs..., modeldata=modeldata)
end

# If this order is changed check the filter usage!
function vars_state(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    η::T # real a 2-D variable TODO: should be 2D
    θ::T
  end
end

# If this order is changed check update_aux!
function vars_aux(m::HBModel, T)
  @vars begin
    w::T
    pkin_reverse::T # ∫(-αᵀ θ) # TODO: remove me after better integral interface
    w_reverse::T               # TODO: remove me after better integral interface
    pkin::T         # ∫(-αᵀ θ)
    wz0::T          # w at z=0
    θʳ::T           # SST given    # TODO: Should be 2D
    f::T            # coriolis
    τ::T            # wind stress  # TODO: Should be 2D
    ν::SVector{3, T}
    κ::SVector{3, T}
  end
end

function vars_gradient(m::HBModel, T)
  @vars begin
    u::SVector{2, T}
    θ::T
  end
end

function vars_diffusive(m::HBModel, T)
  @vars begin
    ∇u::SMatrix{3, 2, T, 6}
    ∇θ::SVector{3, T}
  end
end

function vars_integrals(m::HBModel, T)
  @vars begin
    ∇hu::T
    αᵀθ::T
  end
end

@inline function flux_nondiffusive!(m::HBModel, F::Grad, Q::Vars,
                                    A::Vars, t::Real)
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    η = Q.η
    θ = Q.θ
    w = A.w   # vertical velocity
    pkin = A.pkin
    v = @SVector [u[1], u[2], w]
    Ih = @SMatrix [ 1 -0;
                   -0  1;
                   -0 -0]

    # ∇ • (u θ)
    F.θ += v * θ

    # ∇h • (g η)
    F.u += grav * η * Ih

    # ∇h • (- ∫(αᵀ θ))
    F.u += grav * pkin * Ih

    # ∇h • (v ⊗ u)
    # F.u += v * u'

  end

  return nothing
end

@inline wavespeed(m::HBModel, n⁻, _...) = abs(SVector(m.cʰ, m.cʰ, m.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(::Rusanov, ::HBModel, n⁻, λ, ΔQ::Vars,
                         Q⁻, A⁻, Q⁺, A⁺, t)
  ΔQ.η = -0

  #=
  θ⁻ = Q⁻.θ
  u⁻ = Q⁻.u
  w⁻ = A⁻.w
  @inbounds v⁻ = @SVector [u⁻[1], u⁻[2], w⁻]
  n̂_v⁻ = n⁻∘v⁻

  θ⁺ = Q⁺.θ
  u⁺ = Q⁺.u
  w⁺ = A⁺.w
  @inbounds v⁺ = @SVector [u⁺[1], u⁺[2], w⁺]
  n̂_v⁺ = n⁻∘v⁺

  # max velocity
  # n̂∘v = (abs(n̂∘v⁺) > abs(n̂∘v⁻) ? n̂∘v⁺ : n̂∘v⁻

  # average velocity
  n̂_v = (n̂_v⁻ + n̂_v⁺) / 2

  ΔQ.θ = ((n̂_v > 0) ? 1 : -1) * (n̂_v⁻ * θ⁻ - n̂_v⁺ * θ⁺)
  # ΔQ.θ = abs(n̂_v⁻) * θ⁻ - abs(n̂_v⁺) * θ⁺
  =#

  return nothing
end

@inline function flux_diffusive!(m::HBModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

@inline function gradvariables!(m::HBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(m::HBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

@inline function source!(m::HBModel{P}, source::Vars, Q::Vars, A::Vars,
                         t::Real) where P
  @inbounds begin
    u = Q.u # Horizontal components of velocity
    f = A.f
    wz0 = A.wz0

    # f × u
    source.u -= @SVector [-f * u[2], f * u[1]]

    source.η += wz0
  end

  return nothing
end

@inline function integrate_aux!(m::HBModel, integrand::Vars, Q::Vars, A::Vars)
  αᵀ = m.αᵀ
  integrand.αᵀθ = -αᵀ * Q.θ
  integrand.∇hu = A.w # borrow the w value from A...

  return nothing
end

function update_aux!(dg::DGModel, m::HBModel, Q::MPIStateArray, t::Real)
  MD = dg.modeldata

  # required to ensure that after integration velocity field is divergence free
  vert_filter = MD.vert_filter
  # Q[1] = u[1] = u, Q[2] = u[2] = v
  apply!(Q, (1, 2), dg.grid, vert_filter, VerticalDirection())

  exp_filter = MD.exp_filter
  # Q[4] = θ
  apply!(Q, (4,), dg.grid, exp_filter, VerticalDirection())

  return true
end

function update_aux_diffusive!(dg::DGModel, m::HBModel, Q::MPIStateArray, t::Real)
  A  = dg.auxstate

  # store ∇ʰu as integrand for w
  # update vertical diffusivity for convective adjustment
  function f!(::HBModel, Q, A, D, t)
    @inbounds begin
      A.w = -(D.∇u[1,1] + D.∇u[2,2])

      D.∇θ[3] < 0 ? A.κ = (m.κʰ, m.κʰ, 1000 * m.κᶻ) : A.κ = (m.κʰ, m.κʰ, m.κᶻ)
    end

    return nothing
  end
  nodal_update_aux!(f!, dg, m, Q, t; diffusive=true)

  # compute integrals for w and pkin
  indefinite_stack_integral!(dg, m, Q, A, t) # bottom -> top
  reverse_indefinite_stack_integral!(dg, m, A, t) # top -> bottom

  # project w(z=0) down the stack
  # Need to be consistent with vars_aux
  # A[1] = w, A[5] = wz0
  copy_stack_field_down!(dg, m, A, 5, 1)

  return true
end

function ocean_init_aux! end
function init_aux!(m::HBModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

function ocean_init_state! end
function init_state!(m::HBModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

@inline function boundary_state!(nf, m::HBModel, Q⁺::Vars, A⁺::Vars, n⁻,
                                 Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(m, m.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

@inline function boundary_state!(nf, m::HBModel,
                                 Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻,
                                 Q⁻::Vars, D⁻::Vars, A⁻::Vars,
                                 bctype, t, _...)
  return ocean_boundary_state!(m, m.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end


@inline function ocean_boundary_state!(::HBModel, ::CoastlineFreeSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  Q⁺.u = SVector(-zero(FT), -zero(FT))

  return nothing
end

@inline function ocean_boundary_state!(::HBModel, ::CoastlineNoSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  Q⁺.u = -Q⁻.u

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  A⁺.w = -A⁻.w

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  A⁺.w = -zero(FT)

  return nothing
end



@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorFreeSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  A⁺.w = -A⁻.w
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::Rusanov,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  return nothing
end
@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::CentralNumericalFluxGradient,
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  FT = eltype(Q⁺)
  Q⁺.u = SVector(-zero(FT), -zero(FT))
  A⁺.w = -zero(FT)

  return nothing
end


@inline function ocean_boundary_state!(m::HBModel, ::OceanFloorNoSlip,
                                       ::CentralNumericalFluxDiffusive, Q⁺,
                                       D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)

  Q⁺.u = -Q⁻.u
  A⁺.w = -A⁻.w

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel, ::Union{
                                       OceanSurfaceNoStressNoForcing,
                                       OceanSurfaceStressNoForcing,
                                       OceanSurfaceNoStressForcing,
                                       OceanSurfaceStressForcing},
                                       ::Union{Rusanov,
                                               CentralNumericalFluxGradient},
                                       Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressNoForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = @SMatrix [ -0 -0; -0 -0; A⁺.τ / 1000 -0]
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u + 2 * τ)

  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceNoStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u)

  θ  = Q⁻.θ
  θʳ = A⁺.θʳ
  λʳ = m.problem.λʳ

  σ = @SVector [-0, -0, λʳ * (θʳ - θ)]
  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ + 2 * σ)

  return nothing
end

@inline function ocean_boundary_state!(m::HBModel,
                                       ::OceanSurfaceStressForcing,
                                       ::CentralNumericalFluxDiffusive,
                                       Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
  τ = @SMatrix [ -0 -0; -0 -0; A⁺.τ / 1000 -0]
  D⁺.∇u = Diagonal(A⁺.ν) \ (Diagonal(A⁻.ν) * -D⁻.∇u + 2 * τ)

  θ  = Q⁻.θ
  θʳ = A⁺.θʳ
  λʳ = m.problem.λʳ

  σ = @SVector [-0, -0, λʳ * (θʳ - θ)]
  D⁺.∇θ = Diagonal(A⁺.κ) \ (Diagonal(A⁻.κ) * -D⁻.∇θ + 2 * σ)

  return nothing
end

# Linear model for 1D IMEX
vars_state(lm::LinearHBModel, FT) = vars_state(lm.ocean,FT)
vars_gradient(lm::LinearHBModel, FT) = vars_gradient(lm.ocean,FT)
vars_diffusive(lm::LinearHBModel, FT) = vars_diffusive(lm.ocean,FT)
vars_aux(lm::LinearHBModel, FT) = vars_aux(lm.ocean,FT)
vars_integrals(lm::LinearHBModel, FT) = @vars()

@inline integrate_aux!(::LinearHBModel, _...) = nothing
@inline flux_nondiffusive!(::LinearHBModel, _...) = nothing
@inline source!(::LinearHBModel, _...) = nothing

function wavespeed(lm::LinearHBModel, n⁻, _...)
  C = abs(SVector(lm.ocean.cʰ, lm.ocean.cʰ, lm.ocean.cᶻ)' * n⁻)
  return C
end

@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, A⁺, n⁻, Q⁻, A⁻, t)
end

@inline function boundary_state!(nf, lm::LinearHBModel, Q⁺::Vars, D⁺::Vars, A⁺::Vars,
                                 n⁻, Q⁻::Vars, D⁻::Vars, A⁻::Vars, bctype, t, _...)
  return ocean_boundary_state!(lm.ocean, lm.ocean.problem, bctype, nf, Q⁺, D⁺, A⁺, n⁻, Q⁻, D⁻, A⁻, t)
end

init_aux!(lm::LinearHBModel, A::Vars, geom::LocalGeometry) = nothing
init_state!(lm::LinearHBModel, Q::Vars, A::Vars, coords, t) = nothing

@inline function flux_diffusive!(lm::LinearHBModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
  F.u -= Diagonal(A.ν) * D.∇u
  F.θ -= Diagonal(A.κ) * D.∇θ

  return nothing
end

@inline function gradvariables!(m::LinearHBModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u
  G.θ = Q.θ

  return nothing
end

@inline function diffusive!(lm::LinearHBModel, D::Vars, G::Grad, Q::Vars,
                            A::Vars, t)
  D.∇u = G.u
  D.∇θ = G.θ

  return nothing
end

end
