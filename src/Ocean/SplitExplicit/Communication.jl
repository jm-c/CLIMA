import CLIMA.DGmethods:
    initialize_fast_state!,
    initialize_adjustment!,
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    reconcile_from_fast_to_slow!

using CLIMA.DGmethods: basic_grid_info
#using Printf

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    slow_dt, fast_time_rec, fast_steps
)

    #- inverse ratio of additional fast time steps (for weighted average)
    #  --> do 1/add more time-steps and average from: 1 - 1/add up to: 1 + 1/add
    add = slow.add_fast_substeps

    #- set time-step and number of sub-steps we need
    #  will time-average fast over: fast_steps[1] , fast_steps[3]
    #  centered on fast_steps[2] which corresponds to advance in time of slow
    fast_dt = fast_time_rec[1]
    if add == 0
        steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt ) : 1
        fast_steps[1:3] = [1 1 1] * steps
    else
        steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt / add ) : 1
        fast_steps[2] = add * steps
        fast_steps[1] = ( add - 1 ) * steps
        fast_steps[3] = ( add + 1 ) * steps
    end
    fast_time_rec[1] = slow_dt / fast_steps[2]
    fast_time_rec[2] = 0.
   # @printf("Update: frac_dt = %.1f , dt_fast = %.1f , nsubsteps= %i\n",
   #          slow_dt,fast_time_rec[1],fast_steps[3])
   # println(" fast_steps = ",fast_steps)

    dgFast.auxstate.η_c .= -0
    dgFast.auxstate.U_c .= (@SVector [-0, -0])'

    # preliminary test: no average
    Qfast.η .= dgFast.auxstate.η_s
    Qfast.U .= dgFast.auxstate.U_s

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

@inline function initialize_adjustment!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
)
    ## reset tendency adjustment before calling Baroclinic Model
    dgSlow.auxstate.ΔGu .= 0

    return nothing
end

@inline function tendency_from_slow_to_fast!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    dQslow2fast,
)
    # integrate the tendency
    tendency_dg = dgSlow.modeldata.tendency_dg
    update_aux!(tendency_dg, tendency_dg.balancelaw, dQslow2fast, 0)

    Nq, Nqk, _, _, nelemv, _, nelemh, _ = basic_grid_info(dgSlow)

    ## get top value (=integral over full depth) of ∫du
    boxy_∫du = reshape(tendency_dg.auxstate.∫du, Nq^2, Nqk, 2, nelemv, nelemh)
    flat_∫du = @view boxy_∫du[:, end, :, end, :]

    ## copy into Gᵁ of dgFast
    dgFast.auxstate.Gᵁ .= reshape(flat_∫du, Nq^2, 2, nelemh)

    ## scale by -1/H and copy back to ΔGu
    # note: since tendency_dg.auxstate.∫du is not used after this, could be
    #   re-used to store a 3-D copy of "-Gu"
    boxy_∫gu = reshape(dgSlow.auxstate.ΔGu, Nq^2, Nqk, 2, nelemv, nelemh)
    boxy_∫gu .= -reshape(flat_∫du, Nq^2, 1, 2, 1, nelemh) / slow.problem.H

    return nothing
end

@inline function cummulate_fast_solution!(
    fast::BarotropicModel,
    dgFast,
    Qfast,
    fast_time,
    fast_dt,
    substep, fast_steps, fast_time_rec,
)
    #- might want to use some of the weighting factors: weights_η & weights_U
    #- should account for case where fast_dt < fast.param.dt

    # cumulate Fast solution:
    if substep >= fast_steps[1]
      dgFast.auxstate.U_c .+= Qfast.U
      dgFast.auxstate.η_c .+= Qfast.η
      fast_time_rec[2] += 1.
    end

    # save mid-point solution to start from the next time-step
    if substep == fast_steps[2]
      dgFast.auxstate.U_s .= Qfast.U
      dgFast.auxstate.η_s .= Qfast.η
    end

    return nothing
end

@inline function reconcile_from_fast_to_slow!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    fast_time_rec,
)
    Nq, Nqk, _, _, nelemv, _, nelemh, _ = basic_grid_info(dgSlow)

    # need to calculate int_u using integral kernels
    # u_slow := u_slow + (1/H) * (u_fast - \int_{-H}^{0} u_slow)

    # Compute: \int_{-H}^{0} u_slow)
    ### need to make sure this is stored into aux.∫u

    # integrate vertically horizontal velocity
    flowintegral_dg = dgSlow.modeldata.flowintegral_dg
    update_aux!(flowintegral_dg, flowintegral_dg.balancelaw, Qslow, 0)

    ## get top value (=integral over full depth)
    boxy_∫u = reshape(flowintegral_dg.auxstate.∫u, Nq^2, Nqk, 2, nelemv, nelemh)
    flat_∫u = @view boxy_∫u[:, end, :, end, :]

    ## get time weighted averaged out of cumulative arrays
    dgFast.auxstate.U_c .*= 1 / fast_time_rec[2]
    dgFast.auxstate.η_c .*= 1 / fast_time_rec[2]

    ### substract ∫u from U and divide by H

    ### Δu is a place holder for 1/H * (Ū - ∫u)
    Δu = dgFast.auxstate.Δu
    Δu .= 1 / slow.problem.H * (dgFast.auxstate.U_c - flat_∫u)

    ### copy the 2D contribution down the 3D solution
    ### need to reshape these things for the broadcast
    boxy_u = reshape(Qslow.u, Nq^2, Nqk, 2, nelemv, nelemh)
    boxy_Δu = reshape(Δu, Nq^2, 1, 2, 1, nelemh)
    ### this works, we tested it
    boxy_u .+= boxy_Δu

    ### save eta from 3D model into η_diag (aux var of 2D model)
    ### and store difference between η from Barotropic Model and η_diag
    η_3D = Qslow.η
    boxy_η_3D = reshape(η_3D, Nq^2, Nqk, nelemv, nelemh)
    flat_η = @view boxy_η_3D[:, end, end, :]
    dgFast.auxstate.η_diag .= reshape(flat_η, Nq^2, 1, nelemh)
    dgFast.auxstate.Δη .= dgFast.auxstate.η_c - dgFast.auxstate.η_diag

    ### copy 2D eta over to 3D model
    boxy_η_2D = reshape(dgFast.auxstate.η_c, Nq^2, 1, 1, nelemh)
    boxy_η_3D .= boxy_η_2D

    return nothing
end
