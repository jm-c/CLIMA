using CLIMA.DGmethods:
    initialize_adjustment!,
    initialize_fast_from_slow!,
    cummulate_fast_solution!,
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
#   "sAlt solver"
#   sAlt_solver::SA
    "fast solver"
    fast_solver::FS
    "time step"
    dt::RT
    "time"
    t::RT

    function MultistateMultirateRungeKutta(
        slow_solver::LSRK2N,
#       sAlt_solver::LSRK2N,
        fast_solver,
        Q = nothing;
        dt = ODEs.getdt(slow_solver),
        t0 = slow_solver.t,
    ) where {AT <: AbstractArray}
        SS = typeof(slow_solver)
#       SA = typeof(sAlt_solver)
        FS = typeof(fast_solver)
        RT = real(eltype(slow_solver.dQ))
        return new{SS, SA, FS, RT}(
            slow_solver,
#           sAlt_solver,
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
#   sAlt = msmrrk.sAlt_solver
    fast = msmrrk.fast_solver

    Qslow = Qvec.slow
    Qfast = Qvec.fast

    dQslow = slow.dQ
    dQ2fast = similar(dQslow)
#   dQ2fast = slow.dQ
#   dQslow = sAlt.dQ

    slow_bl = slow.rhs!.balancelaw
    fast_bl = fast.rhs!.balancelaw

    groupsize = 256

    for slow_s in 1:length(slow.RKA)
        # Currnent slow state time
        slow_stage_time = time + slow.RKC[slow_s] * slow_dt

        # Initialize tentency adjustment before evalution of slow mode
        initialize_adjustment!(slow_bl, fast_bl, slow.rhs!, fast.rhs!, Qslow, Qfast)

        # Evaluate the slow mode
        # --> save tendency for the fast
        slow.rhs!(dQ2fast, Qslow, param, slow_stage_time, increment = false)

        # initialize fast model and get slow tendency contribution to advance
        # fast equation  ---> work with dQ2fast as input
        total_fast_step = 0
        initialize_fast_from_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            dQ2fast,
        )

        # TODO: replace slow.rhs! call with use of dQ2fast
        slow.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)
#       sAlt.rhs!(dQslow, Qslow, param, slow_stage_time, increment = true)

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

        ### for testing comment out everything below this
        # Fractional time for slow stage
        if slow_s == length(slow.RKA)
            γ = 1 - slow.RKC[slow_s]
        else
            γ = slow.RKC[slow_s + 1] - slow.RKC[slow_s]
        end

        # Determine number of substeps we need
        fast_dt = ODEs.getdt(fast)
        nsubsteps = fast_dt > 0 ? ceil(Int, γ * slow_dt / ODEs.getdt(fast)) : 1
        fast_dt = γ * slow_dt / nsubsteps

        for substep in 1:nsubsteps
            fast_time = slow_stage_time + (substep - 1) * fast_dt
            ODEs.dostep!(Qfast, fast, param, fast_time, fast_dt)
            #  ---> need to cumulate U at this time (and not at each RKB sub-time-step)
            cummulate_fast_solution!(
                fast_bl,
                fast.rhs!,
                Qfast,
                fast_time,
                fast_dt,
                total_fast_step,
            )
            total_fast_step += 1
        end

        ### later testing ignore this
        # reconcile slow equation using fast equation
        reconcile_from_fast_to_slow!(
            slow_bl,
            fast_bl,
            slow.rhs!,
            fast.rhs!,
            Qslow,
            Qfast,
            total_fast_step,
        )

    end
    return nothing
end
