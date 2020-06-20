import CLIMA.DGmethods:
    initialize_fast_state!,
    initialize_adjustment!,
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    cummulate_last_solution!,
    reconcile_from_fast_to_slow!

using CLIMA.DGmethods: basic_grid_info
#using Printf

"""
Use vector 'fast_time_rec' to store fast-model time record:
      fast_time_rec[1] = adjusted fast_dt
      fast_time_rec[2] = local time advance of fast-model in this "dostep"
                         (i.e., relative to "time", the starting time of full time-step):
                         fast_time = time + fast_time_rec[2]
      fast_time_rec[3] = time to start averaging
      fast_time_rec[4] = time to  end  averaging
      fast_time_rec[5] = cumulated time-weights
      fast_time_rec[6] = partial weight for next fast solution
      fast_time_rec[7] = to debug
Use vector 'fast_steps' to store fast-model time-step number:
      fast_steps[1] = which time-step to save to start next "dostep" time-step
      fast_steps[2] = how many fast time-step to perform
"""

@inline function initialize_fast_state!(
    slow::OceanModel,
    fast::BarotropicModel,
    dgSlow,
    dgFast,
    Qslow,
    Qfast,
    slow_dt, fast_time_rec, fast_steps
)

    #- ratio of additional fast time steps (for weighted average)
    #  --> add more time-step and average from: (1-add)*slow_dt up to: (1+add)*slow_dt
    add = slow.add_fast_substeps

    #- set starting and ending time for fast solution time-averaging:
    fast_time_rec[3] = slow_dt
    fast_time_rec[4] = slow_dt
    if ( add > 0 ) & ( add <= 1 )
       fast_time_rec[3] -= add * slow_dt
       fast_time_rec[4] += add * slow_dt
    end

    #@printf(" Time-averaging interval: fast_time_rec(3,4)= %8.3f , %8.3f\n",
    #         fast_time_rec[3], fast_time_rec[4])

    dgFast.auxstate.η_c .= -0
    dgFast.auxstate.U_c .= (@SVector [-0, -0])'

    # reset fast solution from where we left it
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
    last_step, slow_dt, fast_time_rec, fast_steps
)
    ## reset tendency adjustment before calling Baroclinic Model
    dgSlow.auxstate.ΔGu .= 0

    fast_dt = fast_time_rec[1]
    steps = fast_dt > 0 ? ceil(Int, slow_dt / fast_dt ) : 1
    fast_steps[1] = steps
    fast_time_rec[1] = slow_dt / steps
    if last_step
      fast_steps[2] = ceil( Int64, ( fast_time_rec[4] - fast_time_rec[2] )/ fast_time_rec[1] )
    else
      fast_steps[2] = steps
    end

    #@printf("Update: t=%9.3f, frac_dt =%9.3f , dt_fast =%8.3f , fast_steps = %i, %i\n",
    #         fast_time_rec[2], slow_dt, fast_time_rec[1], fast_steps[1], fast_steps[2])

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
    fast_dt,
    substep, fast_steps, fast_time_rec,
)
    #- might want to use some of the weighting factors: weights_η & weights_U
    local_time = fast_time_rec[2]
    future_time = local_time + fast_dt

    # cumulate Fast solution:
    if ( fast_time_rec[3] <  fast_time_rec[4] ) & ( future_time >=  fast_time_rec[3] )
      if fast_time_rec[5] == 0.
        p_weight = 0.5 * ( future_time - fast_time_rec[3] ) / fast_dt
        n_weight = p_weight * ( fast_dt + fast_time_rec[3] - local_time )
        p_weight *= ( future_time - fast_time_rec[3] )
      else
        if ( future_time >  fast_time_rec[4] )
          n_weight = 0.5 * ( fast_time_rec[4] - local_time ) / fast_dt
          p_weight = n_weight * ( fast_dt + future_time - fast_time_rec[4] )
          n_weight *= ( fast_time_rec[4] - local_time )
        else
          p_weight = 0.5 * fast_dt
          n_weight = 0.5 * fast_dt
        end
      end
      #@printf(" ns= %3i Cumul: t=%9.3f , p_w=%8.3f, n_w=%8.3f,",
      #         substep, local_time, p_weight, n_weight)
      dt_weight = fast_time_rec[6] + p_weight
      fast_time_rec[6] = n_weight
    else
      #@printf(" ns= %3i Cumul: t=%9.3f ,                            ", substep, local_time )
      dt_weight = 0.
    end

    if dt_weight > 0.
      dgFast.auxstate.U_c .+= Qfast.U * dt_weight
      dgFast.auxstate.η_c .+= Qfast.η * dt_weight
      fast_time_rec[5] += dt_weight
     #fast_time_rec[7] += dt_weight * local_time
    end
    #@printf(" W=%8.3f , Sum=%10.3f ,%13.3f\n",
    #         dt_weight, fast_time_rec[5], fast_time_rec[7])

    # save mid-point solution to start from the next time-step
    if substep == fast_steps[1]
      dgFast.auxstate.U_s .= Qfast.U
      dgFast.auxstate.η_s .= Qfast.η
    end

    return nothing
end

@inline function cummulate_last_solution!(
    fast::BarotropicModel,
    dgFast,
    Qfast,
    fast_dt,
    substep, fast_steps, fast_time_rec,
)
    #- might want to use some of the weighting factors: weights_η & weights_U
    local_time = fast_time_rec[2]

    # cumulate Fast solution:
    if ( fast_time_rec[3] <  fast_time_rec[4] )
      dt_weight = fast_time_rec[6]
      fast_time_rec[6] = 0.
    else
      dt_weight = 1.
    end

    if dt_weight > 0.
      dgFast.auxstate.U_c .+= Qfast.U * dt_weight
      dgFast.auxstate.η_c .+= Qfast.η * dt_weight
      fast_time_rec[5] += dt_weight
     #fast_time_rec[7] += dt_weight * local_time
    end

    # save mid-point solution to start from the next time-step
    if substep == fast_steps[1]
      dgFast.auxstate.U_s .= Qfast.U
      dgFast.auxstate.η_s .= Qfast.η
    end

    #@printf(
    # " ns= %3i Final: t=%9.3f ,                             W=%8.3f , Sum=%10.3f ,%13.3f\n",
    #        substep, fast_time_rec[2], dt_weight, fast_time_rec[5], fast_time_rec[7] )

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
    dgFast.auxstate.U_c .*= 1 / fast_time_rec[5]
    dgFast.auxstate.η_c .*= 1 / fast_time_rec[5]

    #fast_time_rec[7] *= 1 / fast_time_rec[5]
    #@printf("Final averaged: fast_time_rec(7) =%12.6f\n", fast_time_rec[7])

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
