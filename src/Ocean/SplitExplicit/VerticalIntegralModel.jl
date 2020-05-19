struct TendencyIntegralModel{M} <: AbstractOceanModel
    ocean::M
    function TendencyIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(tm::TendencyIntegralModel, FT) = vars_state(tm.ocean, FT)
vars_diffusive(tm::TendencyIntegralModel, FT) = @vars()

function vars_aux(m::TendencyIntegralModel, T)
    @vars begin
        ∫du::SVector{2, T}
    end
end

function vars_integrals(m::TendencyIntegralModel, T)
    @vars begin
        ∫du::SVector{2, T}
    end
end

@inline function integral_load_aux!(
    m::TendencyIntegralModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∫du = A.∫du

    return nothing
end

@inline function integral_set_aux!(m::TendencyIntegralModel, A::Vars, I::Vars)
    A.∫du = I.∫du

    return nothing
end

init_aux!(tm::TendencyIntegralModel, A::Vars, geom::LocalGeometry) = nothing

function update_aux!(
    dg::DGModel,
    tm::TendencyIntegralModel,
    dQ::MPIStateArray,
    t::Real,
)
    A = dg.auxstate

    # copy tendency vector to aux state for integration
    function f!(::TendencyIntegralModel, dQ, A, t)
        @inbounds begin
            A.∫du = @SVector [dQ.u[1], dQ.u[2]]
        end

        return nothing
    end
    nodal_update_aux!(f!, dg, tm, dQ, t)

    # compute integral for Gᵁ
    indefinite_stack_integral!(dg, tm, dQ, A, t) # bottom -> top

    return true
end

#-------------------------------------------------------------------------------
struct FlowIntegralModel{M} <: AbstractOceanModel
    ocean::M
    function FlowIntegralModel(ocean::M) where {M}
        return new{M}(ocean)
    end
end
vars_state(fm::FlowIntegralModel, FT) = vars_state(fm.ocean, FT)
vars_diffusive(fm::FlowIntegralModel, FT) = @vars()

function vars_aux(m::FlowIntegralModel, T)
    @vars begin
        ∫u::SVector{2, T}
    end
end

function vars_integrals(m::FlowIntegralModel, T)
    @vars begin
        ∫u::SVector{2, T}
    end
end

@inline function integral_load_aux!(
    m::FlowIntegralModel,
    I::Vars,
    Q::Vars,
    A::Vars,
)
    I.∫u = Q.u

    return nothing
end

@inline function integral_set_aux!(m::FlowIntegralModel, A::Vars, I::Vars)
    A.∫u = I.∫u

    return nothing
end

init_aux!(fm::FlowIntegralModel, A::Vars, geom::LocalGeometry) = nothing

function update_aux!(
    dg::DGModel,
    fm::FlowIntegralModel,
    Q::MPIStateArray,
    t::Real,
)
    A = dg.auxstate

    # compute vertical integral of u
    indefinite_stack_integral!(dg, fm, Q, A, t) # bottom -> top

    return true
end

