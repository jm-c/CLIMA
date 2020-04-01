import CLIMA.DGmethods:
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

using CLIMA.DGmethods: basic_grid_info

struct BarotropicModel{M} <: AbstractOceanModel
    baroclinic::M
    function BarotropicModel(baroclinic::M) where {M}
        return new{M}(baroclinic)
    end
end

function vars_state(m::BarotropicModel, T)
    @vars begin
        η::T
        U::SVector{2, T}
    end
end

function init_state!(m::BarotropicModel, Q::Vars, A::Vars, coords, t)
    Q.η = 0
    Q.U = @SVector [-0, -0]
    return nothing
end

function vars_aux(m::BarotropicModel, T)
    @vars begin
        Gᵁ::SVector{2, T} # integral of baroclinic tendency
        η̄::T              # running averge of η
        Ū::SVector{2, T}  # running averge of U
    end
end

function init_aux!(m::BarotropicModel, A::Vars, geom::LocalGeometry)
     A.η̄ = -0
     A.Ū = @SVector [-0, -0]
    return ocean_init_aux!(m, m.baroclinic.problem, A, geom)
end

vars_gradient(m::BarotropicModel, T) = @vars()
vars_diffusive(m::BarotropicModel, T) = @vars()
vars_integrals(m::BarotropicModel, T) = @vars()

@inline flux_diffusive!(::BarotropicModel, args...) = nothing

@inline function flux_nondiffusive!(
    m::BarotropicModel,
    F::Grad,
    Q::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        U = @SVector [Q.U[1], Q.U[2], 0]
        η = Q.η
        H = m.baroclinic.problem.H
        Iʰ = @SMatrix [
            1 -0
            -0 1
            -0 -0
        ]

        F.η += U
        F.U += grav * H * η * Iʰ
    end
end

@inline function source!(
    m::BarotropicModel,
    S::Vars,
    Q::Vars,
    D::Vars,
    A::Vars,
    t::Real,
)
    @inbounds begin
        S.U += A.Gᵁ
    end
end

@inline wavespeed(m::BarotropicModel, n⁻, _...) =
    abs(SVector(m.baroclinic.cʰ, m.baroclinic.cʰ, m.baroclinic.cᶻ)' * n⁻)

# We want not have jump penalties on η (since not a flux variable)
function update_penalty!(
    ::Rusanov,
    ::BarotropicModel,
    n⁻,
    λ,
    ΔQ::Vars,
    Q⁻,
    A⁻,
    Q⁺,
    A⁺,
    t,
)
    ΔQ.η = -0

    return nothing
end

"""
    boundary_state!(nf, ::BarotropicModel, Q⁺, A⁺, Q⁻, A⁻, bctype)

applies boundary conditions for the hyperbolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(
    nf,
    m::BarotropicModel,
    Q⁺::Vars,
    A⁺::Vars,
    n⁻,
    Q⁻::Vars,
    A⁻::Vars,
    bctype,
    t,
    _...,
)
    return ocean_boundary_state!(
        m,
        m.baroclinic.problem,
        bctype,
        nf,
        Q⁺,
        A⁺,
        n⁻,
        Q⁻,
        A⁻,
        t,
    )
end

"""
    boundary_state!(nf, ::BarotropicModel, Q⁺, D⁺, A⁺, Q⁻, D⁻, A⁻, bctype)

applies boundary conditions for the parabolic fluxes
dispatches to a function in OceanBoundaryConditions.jl based on bytype defined by a problem such as SimpleBoxProblem.jl
"""
@inline function boundary_state!(
    nf,
    m::BarotropicModel,
    Q⁺::Vars,
    D⁺::Vars,
    A⁺::Vars,
    n⁻,
    Q⁻::Vars,
    D⁻::Vars,
    A⁻::Vars,
    bctype,
    t,
    _...,
)
    return ocean_boundary_state!(
        m,
        m.baroclinic.problem,
        bctype,
        nf,
        Q⁺,
        D⁺,
        A⁺,
        n⁻,
        Q⁻,
        D⁻,
        A⁻,
        t,
    )
end

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    Qslow,
    Qfast,
    dgSlow,
    dgFast,
)
    dgFast.auxstate.η̄ .= -0
    dgFast.auxstate.Ū .= (@SVector [-0, -0])'

    # copy η and U from 3D equation
    # to calculate U we need to do an integral of u from the 3D
    indefinite_stack_integral!(dgSlow, slow, Qslow, dgSlow.auxstate, 0)

    Nq, Nqk, _, _, nelemv, _, nelemh, _ = basic_grid_info(dgSlow)

    ### copy results of integral to 2D equation
    boxy_∫u = reshape(dgSlow.auxstate.∫u, Nq^2, Nqk, 2, nelemv, nelemh)
    flat_∫u = @view boxy_∫u[:, end, :, end, :]
    Qfast.U .= reshape(flat_∫u, Nq^2, 2, nelemh)

    boxy_η = reshape(Qslow.η, Nq^2, Nqk, nelemv, nelemh)
    flat_η = @view boxy_η[:, end, end, :]
    Qfast.η .= reshape(flat_η, Nq^2, 1, nelemh)

    return nothing
end

@inline function pass_tendency_from_slow_to_fast!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qfast,
    dQslow,
)
    # integrate the tendency
    tendency_dg = dgSlow.modeldata.tendency_dg
    update_aux!(tendency_dg, tendency_dg.balancelaw, dQslow, 0)

    Nq, Nqk, _, _, nelemv, _, nelemh, _ = basic_grid_info(dgSlow)

    ### copying ∫du from newdg into Gᵁ of dgFast
    boxy_∫du = reshape(tendency_dg.auxstate.∫du, Nq^2, Nq, 2, nelemv, nelemh)
    flat_∫du = @view boxy_∫du[:, end, :, end, :]
    dgFast.auxstate.Gᵁ .= reshape(flat_∫du, Nq^2, 2, nelemh)

    return nothing
end

@inline function cummulate_fast_solution!(
    fast::BarotropicModel,
    dgFast,
    Qfast,
    fast_time,
    fast_dt,
    total_fast_step,
)
    #- might want to use some of the weighting factors: weights_η & weights_U
    #- should account for case where fast_dt < fast.param.dt
    total_fast_step += 1

    # cumulate Fast solution:
    # dgFast.auxstate.η̄ .+= Qfast.η
    # dgFast.auxstate.Ū .+= Qfast.U
    # for now, with our simple weight, we just take the most recent value for the average
    dgFast.auxstate.η̄ .= Qfast.η
    dgFast.auxstate.Ū .= Qfast.U

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    dQslow,
    Qslow,
    Qfast,
    total_fast_step,
)
    # need to calculate int_u using integral kernels
    # u_slow := u_slow + (1/H) * (u_fast - \int_{-H}^{0} u_slow)

    # Compute: \int_{-H}^{0} u_slow)
    ### need to make sure this is stored into aux.∫u
    indefinite_stack_integral!(dgSlow, slow, Qslow, dgSlow.auxstate, 0)

    Nq, Nqk, _, _, nelemv, _, nelemh, _ = basic_grid_info(dgSlow)

    ### substract ∫u from U and divide by H
    boxy_∫u = reshape(dgSlow.auxstate.∫u, Nq^2, Nq, 2, nelemv, nelemh)
    flat_∫u = @view boxy_∫u[:, end, :, end, :]

    ### apples is a place holder for 1/H * (Ū - ∫u)
    apples = dgFast.auxstate.Ū
    apples .= 1 / slow.problem.H * (Qfast.U - flat_∫u)
  # apples .=
  #     1 / slow.problem.H * (dgFast.auxstate.Ū / total_fast_step - flat_∫u)

    ### need to reshape these things for the broadcast
    boxy_du = reshape(dQslow.u, Nq^2, Nqk, 2, nelemv, nelemh)
    boxy_u = reshape(Qslow.u, Nq^2, Nqk, 2, nelemv, nelemh)
    boxy_apples = reshape(apples, Nq^2, 1, 2, 1, nelemh)

    ## this works, we tested it
    ## copy the 2D contribution down the 3D solution
  # boxy_u .+= boxy_apples
  # boxy_du .+= boxy_apples

    ### copy 2D eta over to 3D model
    boxy_η_3D = reshape(Qslow.η, Nq^2, Nq, nelemv, nelemh)
    boxy_η̄_2D = reshape(dgFast.auxstate.η̄, Nq^2, 1, 1, nelemh)
  # boxy_η_3D .= boxy_η̄_2D

    return nothing
end
