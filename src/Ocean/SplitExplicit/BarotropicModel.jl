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
        η̄::T              # running averge of η        
        Ū::SVector{2, T}  # running averge of U
    end
end

function init_state!(m::BarotropicModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

function vars_aux(m::BarotropicModel, T)
    @vars begin
        Gᵁ::SVector{2, T}
        weights_η::T
        weights_U::T
        ∫u::SVector{2, T}
    end
end

function init_aux!(m::BarotropicModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

vars_gradient(m::BarotropicModel, T) = @vars()
vars_diffusive(m::BarotropicModel, T) = @vars()
vars_integrals(m::BarotropicModel, T) = @vars()

@inline flux_diffusive!(::BarotropicModel, ...) = nothing

@inline function flux_nondiffusive!(m::BarotropicModel, F::Grad,
                                    Q::Vars, A::Vars, t::Real)
    @inbounds begin
        U = Q.U
        η = Q.η
        H = m.baroclinic.problem.H
        I = LinearAlgebra.I
        
        F.η += U
        F.U += grav * H * η * I
    end
end

@inline function source!(m::BarotropicModel, S::Vars, Q::Vars, A::Vars, t::Real)
    @inbounds begin
        S.U += A.Gᵁ
    end
end

@inline function initialize_tendency(Qfast, Qslow, fast::BarotropicModel,
                                     slow::HydrostaticBoussinesqModel)
    Qfast.η̄ = -0
    Qfast.Ū = @SVector [-0, -0]

    # copy η and U from 3D equation
    # to calculate U we need to do an integral of u from the 3D
    
    return nothing
end

@inline function pass_tendency_from_slow_to_fast!(Qfast, dQslow, fast::BarotropicModel, slow::HydrostaticBoussinesqModel)
    ### need to copy w at z=0 into wz0

    dgFast.A.Gᵁ
    return nothing
end

@inline function reconcile_from_fast_to_slow!(slow::HydrostaticBoussinesqModel,
                                              fast::BarotropicModel,
                                              dQslow, Qslow, Qfast)
    ### need to copy η to aux for 3D

    # project w(z=0) down the stack
    # Need to be consistent with vars_aux
    # A[1] = w, A[5] = wz0
    copy_stack_lowdim!(dgSlow, slow, dgSlow.A, Qfast, -1, 2)

    # need to calculate int_u using integral kernels
    # u_slow := u_slow + (1/H) * (u_fast - \int_{-H}^{0} u_slow)
    # Compute: \int_{-H}^{0} u_slow)
    # TODO: write definite_stack_integral!
    definite_stack_integral!(dgSlow, slow, fast, Qslow, dgFast.auxstate, t)

    # TODO: Write single kernel the takes the 2D state and reconciles in a
    # generic fashion all the points down the stack allowing the user to modify
    # dQ, Q, and aux from the higher dim model

    return nothing
end
