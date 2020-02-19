struct HorizontalModel{M} <: AbstractOceanModel
    ocean::M
    function HorizontalModel{ocean::M} where {M}
        return new{M}(ocean)
    end
end

function vars_state(m::HorizontalModel, T)
    @vars begin
        u::SVector{2, T}
    end
end

function init_state!(m::HorizontalModel, Q::Vars, A::Vars, coords, t)
  return ocean_init_state!(m.problem, Q, A, coords, t)
end

function vars_aux(m::HorizontalModel, T)
    @vars begin
        η::T
        ν::SVector{2, T}
    end
end

function init_aux!(m::HorizontalModel, A::Vars, geom::LocalGeometry)
  return ocean_init_aux!(m, m.problem, A, geom)
end

function vars_gradient(m::HorizontalModel, T)
  @vars begin
    u::SVector{2, T}
  end
end

@inline function gradvariables!(m::HorizontalModel, G::Vars, Q::Vars, A, t)
  G.u = Q.u

  return nothing
end

function vars_diffusive(m::HorizontalModel, T)
  @vars begin
    ∇u::SMatrix{3, 2, T, 6}
  end
end

@inline function diffusive!(m::HorizontalModel, D::Vars, G::Grad,
                            Q::Vars, A::Vars, t)
  D.∇u = G.u

  return nothing
end

@inline function flux_nondiffusive!(m::HorizontalModel, F::Grad, Q::Vars,
                                    A::Vars, t::Real)
    @inbounds begin
        η = Q.η
        Ih = @SMatrix [ 1 -0;
                       -0  1;
                       -0 -0]
        
        # ∇h • (g η)
        F.u += grav * η * Ih
    end
    
    return nothing
end

@inline function flux_diffusive!(m::HorizontalModel, F::Grad, Q::Vars, D::Vars,
                                 A::Vars, t::Real)
    F.u -= Diagonal([A.ν[1], A.ν[2], -0]) * D.∇u

    return nothing
end

function wavespeed(hm::HorizontalModel, n⁻, _...)
  C = abs(SVector(hm.ocean.cʰ, hm.ocean.cʰ, hm.ocean.cᶻ)' * n⁻)
  return C
end
