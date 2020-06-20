using CLIMA.DGmethods:
    initialize_fast_state!,
    initialize_adjustment!,
    tendency_from_slow_to_fast!,
    cummulate_fast_solution!,
    cummulate_last_solution!,
    reconcile_from_fast_to_slow!

include("MultirateRungeKuttaMethod_kernels.jl")

export MultistateMultirateRungeKutta

ODEs = ODESolvers
LSRK2N = LowStorageRungeKutta2N

"""
    MultistateMultirateRungeKutta(slow_solver, fast_solver; dt, t0 = 0)

This is a time stepping object for explicitly time stepping the differential
equation given by the right-hand-side function `f` with the state `Q`, i.e.,

```math
  \\dot{Q_fast} = f_fast(Q_fast, Q_slow, t)
  \\dot{Q_slow} = f_slow(Q_slow, Q_fast, t)
```

with the required time step size `dt` and optional initial time `t0`.  This
time stepping object is intended to be passed to the `solve!` command.

The constructor builds a multistate multirate Runge-Kutta scheme using two different RK
solvers and two different MPIStateArrays. This is based on

Currently only the low storage RK methods can be used as slow solvers

  - [`LowStorageRungeKuttaMethod`](@ref)

### References
"""
mutable struct MultistateMultirateRungeKutta{SS, SA, FS, RT} <:
               ODEs.AbstractODESolver
    "slow solver"
    slow_solver::SS
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultistateMultirateRungeKutta(
        slow_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = ODEs.getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        return new{SS, SA, FS, RT}(
            slow_solver,
            fast_solver,
            RT(dt),
            RT(t0),
        )
    end
end
MSMRRK = MultistateMultirateRungeKutta

function ODEs.dostep!(
    Qvec,
    msmrrk::MSMRRK,
    param,
    timeend::Real,
    adjustfinalstep::Bool,
)
    time, dt = msmrrk.t, msmrrk.dt
    @assert dt > 0
    if adjustfinalstep && time + dt > timeend
        dt = timeend - time
        @assert dt > 0
    end

    ODEs.dostep!(Qvec, msmrrk, param, time, dt)

    if dt == msmrrk.dt
        msmrrk.t += dt
    else
        msmrrk.t = timeend
    end
    return msmrrk.t
end

function ODEs.dostep!(
    Qvec,
    msmrrk::MSMRRK{SS},
    param,
    time::Real,
    slow_dt::AbstractFloat,
) where {SS <: LSRK2N}
    slow = msmrrk.slow_solver
    fast = msmrrk.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast

    dQslow = slow.dQ
    dQ2fast = similar(dQslow)

    slow_bl = slow.rhs!.balancelaw
    fast_bl = fast.rhs!.balancelaw

    FT = typeof(slow_dt)

    #- get the RK weight that apply to each RK tendency in the final solution
    rkW = zeros(FT, length(slow.RKA))
    rkW .= slow.RKB
    for s in length(slow.RKA):-1:2
        rkW[s-1] += slow.RKA[s] * rkW[s]
    end

    # Initialize fast model and set time-step and number of substeps we need
    fast_dt_in = ODEs.getdt(fast)
    fast_time_rec = zeros(FT, 7)
    fast_steps = [0 0]
    initialize_fast_state!(slow_bl, fast_bl, slow.rhs!, fast.rhs!, Qslow, Qfast,
                           slow_dt, fast_time_rec, fast_steps )

    groupsize = 256

    for slow_s in 1:length(slow.RKA)
        # Current slow state time
        slow_stage_time = time + slow.RKC[slow_s] * slow_dt

        # Fractional time for slow stage
        last_step = slow_s == length(slow.RKA) ? true : false
        fract_dt = rkW[slow_s] * slow_dt

        # Initialize tentency adjustment before evalution of slow mode
        # and set time-step and number of substeps we need
        fast_time_rec[1] = fast_dt_in
        initialize_adjustment!(slow_bl, fast_bl, slow.rhs!, fast.rhs!, Qslow, Qfast,
                               last_step, fract_dt, fast_time_rec, fast_steps )

        # Evaluate the slow mode
        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        # vertically integrate slow tendency to advance fast equation
        # and use vertical mean for slow model (negative source)
        # ---> work with dQ2fast as input
        tendency_from_slow_to_fast!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            dQ2fast,
        )

        # Compute (and RK update) slow tendency
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

        # Update (RK-stage) slow state
        event = Event(device(Qslow))
        event = update!(device(Qslow), groupsize)(
            realview(dQslow),
            realview(Qslow),
            slow.RKA[slow_s % length(slow.RKA) + 1],
            slow.RKB[slow_s],
            slow_dt,
            nothing,
            nothing,
            nothing;
            ndrange = length(realview(Qslow)),
            dependencies = (event,),
        )
        wait(device(Qslow), event)

        # Determine number of substeps we need
        fast_dt = fast_time_rec[1]
        nsubsteps = fast_steps[2]

        for substep in 0:nsubsteps-1
          # fast_time = slow_stage_time + substep * fast_dt
            fast_time = time + fast_time_rec[2]
            # cumulate fast solution
            cummulate_fast_solution!(
                fast_bl,
                fast.rhs!,
                Qfast,
                fast_dt,
                substep, fast_steps, fast_time_rec,
            )

            #-- step forward fast model and update time-record
            ODEs.dostep!(Qfast, fast, param, fast_time, fast_dt)
            fast_time_rec[2] += fast_dt
        end

    end

    # add last fast solution to cumulate
    fast_dt = fast_time_rec[1]
    nsubsteps = fast_steps[2]
    cummulate_last_solution!(
            fast_bl,
            fast.rhs!,
            Qfast,
            fast_dt,
            nsubsteps, fast_steps, fast_time_rec,
    )

    # reconcile slow equation using fast equation
    reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            fast_time_rec,
    )

    return nothing
end
