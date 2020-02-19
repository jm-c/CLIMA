struct BarotropicModel{M} <: BalanceLaw
    baroclinic::M
    function LinearHBModel(baroclinic::M) where {M}
        return new{M}(baroclinic)
    end
end

function vars_state(m::BarotropicModel, T)
    @vars begin
        η::T
        η̄::T              # running averge of η
        U::SVector{2, T}
        Ū::SVector{2, T}  # running averge of U
    end
end

function vars_aux(m::BarotropicModel, T)
    @vars begin
        wz0::T
        Gᵁ::SVector{2, T} 
    end
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
        # S.η += A.wz0
        S.U += A.Gᵁ 
    end
end

@inline function initialize_tendency(Qfast, Qslow, fast::BarotropicModel, slow::HydrostaticBoussinesqModel)
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

@inline function reconcile_from_fast_to_slow!(Qslow, Qfast, slow::HydrostaticBoussinesqModel, fast::BarotropicModel, scaling)
    ### need to copy η to aux for 3D

    # project w(z=0) down the stack
    # Need to be consistent with vars_aux
    # A[1] = w, A[5] = wz0
    copy_stack_lowdim!(dgSlow, slow, dgSlow.A, Qfast, -1, 2)

    # need to calculate int_u using integral kernels
    indefinite_stack_integral(dgSlow, slow, Qslow, dgSlow.A, t)
    reverse_indefinite_stack_integral(dgSlow, slow, Qslow, dgSlow.A, t)
    # then need to take the very top value
    copy_stack_field_down!(dgSlow, slow, dgSlow.A, ints)

    Qslow.u += 1//H * (scaling * Qfast.Ū - dgSlow.A.∫u)
    
    return nothing
end
