import CLIMA.DGmethods:
    initialize_fast_state!,
    pass_tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

using CLIMA.DGmethods: basic_grid_info

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
)
    dgFast.auxstate.η̄ .= -0
    dgFast.auxstate.Ū .= (@SVector [-0, -0])'

  #=
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
  =#

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
    # total_fast_step += 1  # now done outside since Integer are passed by value !!!

    # cumulate Fast solution:
    dgFast.auxstate.Ū .+= Qfast.U
    # dgFast.auxstate.η̄ .+= Qfast.η
    # for now, with our simple weight, we just take the most recent value for η
    dgFast.auxstate.η̄ .= Qfast.η

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
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

    ### Δu is a place holder for 1/H * (Ū - ∫u)
    Δu = dgFast.auxstate.Δu
    Δu .= 1 / slow.problem.H * (Qfast.U - flat_∫u)
 #  Δu .= 1 / slow.problem.H * (dgFast.auxstate.Ū / total_fast_step - flat_∫u)

    ### copy the 2D contribution down the 3D solution
    ### need to reshape these things for the broadcast
    boxy_u = reshape(Qslow.u, Nq^2, Nqk, 2, nelemv, nelemh)
    boxy_Δu = reshape(Δu, Nq^2, 1, 2, 1, nelemh)
    ### this works, we tested it
 #  boxy_u .+= boxy_Δu

    ### save eta from 3D model into η_diag (aux var of 2D model)
    ### and store difference between η from Barotropic Model and η_diag
    η_3D = Qslow.η
    boxy_η_3D = reshape(η_3D, Nq^2, Nq, nelemv, nelemh)
    flat_η = @view boxy_η_3D[:, end, end, :]
    dgFast.auxstate.η_diag .= reshape(flat_η, Nq^2, 1, nelemh)
    dgFast.auxstate.Δη .= dgFast.auxstate.η̄  - dgFast.auxstate.η_diag

    ### copy 2D eta over to 3D model
    boxy_η̄_2D = reshape(dgFast.auxstate.η̄, Nq^2, 1, 1, nelemh)
 #  boxy_η_3D .= boxy_η̄_2D

    return nothing
end