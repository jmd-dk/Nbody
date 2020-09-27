# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2020 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
import interactions
cimport('from analysis import debug, measure, powerspec')
cimport('from communication import domain_subdivisions')
cimport('from graphics import render2D, render3D')
cimport(
    'from integration import   '
    '    cosmic_time,          '
    '    expand,               '
    '    hubble,               '
    '    init_time,            '
    '    remove_doppelgängers, '
    '    scale_factor,         '
    '    scalefactor_integral, '
)
cimport('from snapshot import get_initial_conditions, save')
cimport('from utilities import delegate')



# Function containing the main time loop of CO𝘕CEPT
@cython.header(
    # Locals
    a='double',
    autosave_filename=str,
    autosave_time='double',
    bottleneck=str,
    component='Component',
    components=list,
    dump_index='Py_ssize_t',
    dump_time=object,  # collections.namedtuple
    dump_times=list,
    dump_times_a=set,
    dump_times_t=set,
    interaction_name=str,
    output_filenames=dict,
    recompute_Δt_max='bint',
    static_timestepping_func=object,  # callable or None
    subtiling='Tiling',
    subtiling_computation_times=object,  # collections.defaultdict
    subtiling_name=str,
    sync_at_dump='bint',
    sync_time='double',
    t='double',
    tiling='Tiling',
    tiling_name=str,
    time_step='Py_ssize_t',
    time_step_last_sync='Py_ssize_t',
    time_step_previous='Py_ssize_t',
    time_step_type=str,
    timespan='double',
    Δt='double',
    Δt_backup='double',
    Δt_begin='double',
    Δt_min='double',
    Δt_max='double',
    returns='void',
)
def timeloop():
    # Do nothing if no dump times exist
    if not (  [nr for val in output_times['a'].values() for nr in val]
            + [nr for val in output_times['t'].values() for nr in val]):
        return
    # Print out domain decomposition
    masterprint(
        f'Domain decomposition: '
        f'{domain_subdivisions[0]}×{domain_subdivisions[1]}×{domain_subdivisions[2]}'
    )
    # Determine and set the correct initial values for the cosmic time
    # universals.t and the scale factor universals.a = a(universals.t).
    init_time()
    # Get the initial components.
    # These may be loaded from a snapshot or generated from scratch.
    masterprint('Setting up initial conditions ...')
    components = get_initial_conditions()
    if not components:
        masterprint('done')
        return
    # Get the dump times and the output filename patterns
    dump_times, output_filenames = prepare_for_output(components)
    # Stow away passive components into a separate (global) list.
    # We should always keep it such that
    #   components + passive_components
    # results in a list of all components.
    # We further store the original ordering of all components.
    components_order[:] = [component.name for component in components]
    passive_components[:] = [component for component in components if not component.is_active()]
    components = [component for component in components if component not in passive_components]
    # Realize all linear fluid variables of all components
    for component in components:
        component.realize_if_linear(0, specific_multi_index=0)        # ϱ
        component.realize_if_linear(1, specific_multi_index=0)        # J
        component.realize_if_linear(2, specific_multi_index='trace')  # 𝒫
        component.realize_if_linear(2, specific_multi_index=(0, 0))   # ς
    masterprint('done')
    # Possibly output at the beginning of simulation
    if dump_times[0].t == universals.t or dump_times[0].a == universals.a:
        dump(components, output_filenames, dump_times[0])
        dump_times.pop(0)
        # Return now if all dumps lie at the initial time
        if len(dump_times) == 0:
            return
    # Set initial time step size
    static_timestepping_func = prepare_static_timestepping()
    if Δt_begin_autosave == -1:
        # Including the current (initial) t in initial_fac_times in
        # order to scale Δt_max by Δt_initial_fac,
        # making it appropriate for use as Δt_begin.
        initial_fac_times.add(universals.t)
        Δt_max, bottleneck = get_base_timestep_size(components, static_timestepping_func)
        Δt_begin = Δt_max
        # We always want the simulation time span to be at least
        # one whole Δt_period long.
        timespan = dump_times[len(dump_times) - 1].t - universals.t
        if Δt_begin > timespan/Δt_period:
            Δt_begin = timespan/Δt_period
        # We need at least 1.5 base time steps before the first dump
        if Δt_begin > (dump_times[0].t - universals.t)/1.5:
            Δt_begin = (dump_times[0].t - universals.t)/1.5
        Δt = Δt_begin
    else:
        # Set Δt_begin and Δt to the autosaved values
        Δt_begin = Δt_begin_autosave
        Δt = Δt_autosave
    # Minimum allowed time step size.
    # If Δt needs to be lower than this, the program will terminate.
    Δt_min = 1e-4*Δt_begin
    # Record what time it is, for use with autosaving
    autosave_time = time()
    # Populate the global ᔑdt_scalar and ᔑdt_rungs dicts
    # with integrand keys
    get_time_step_integrals(0, 0, components + passive_components)
    # Construct initial rung populatations by carrying out an initial
    # short kick, but without applying the momentum updates.
    initialize_rung_populations(components, Δt)
    # Mapping from (short-range) interaction names
    # to (subtile) computation times.
    subtiling_computation_times = collections.defaultdict(lambda: collections.defaultdict(float))
    # The main time loop
    masterprint('Beginning of main time loop')
    time_step = initial_time_step
    time_step_previous = time_step - 1
    bottleneck = ''
    time_step_type = 'init'
    sync_time = ထ
    time_step_last_sync = 0
    recompute_Δt_max = True
    Δt_backup = -1
    for dump_index, dump_time in enumerate(dump_times):
        # Break out of this loop when a dump has been performed
        while True:
            # Things to do at the beginning and end of each time step
            if time_step > time_step_previous:
                time_step_previous = time_step
                # If at an "init" time step, all short-range rungs will
                # be synchronized. Here, re-assign a short-range rung to
                # each particle based on their short-range acceleration,
                # disregarding their currently assigned rung and flagged
                # rung jumps. Any flagged rung jumps will be nullified.
                if time_step_type == 'init':
                    for component in components:
                        component.assign_rungs(Δt, fac_softening)
                # Update subtile computation times
                for component in components:
                    for subtiling_name, subtiling in component.tilings.items():
                        match = re.search(r'(.*) \(subtiles\)', subtiling_name)
                        if not match:
                            continue
                        subtiling_computation_times[component][match.group(1)
                            ] += subtiling.computation_time_total
                # Print out message at the end of each time step
                if time_step > initial_time_step:
                    print_timestep_footer(components)
                # Reset all computation_time_total tiling attributes
                for component in components:
                    for tiling in component.tilings.values():
                        tiling.computation_time_total = 0
                # Update universals.time_step. This is only ever done
                # here, and so in general you should not count on
                # universals.time_step being exactly equal to time_step.
                universals.time_step = time_step
                # Print out message at the beginning of each time step
                print_timestep_heading(time_step, Δt,
                    bottleneck if time_step_type == 'init' else '', components)
            # Analyze and print out debugging information, if required
            with unswitch:
                if enable_debugging:
                    debug(components)
            # Handle the time step.
            # This is either of type "init" or "full".
            if time_step_type == 'init':
                # An init step is always followed by a full step
                time_step_type = 'full'
                # This is not a full base time step. Half a long-range
                # kick will be applied, i.e. fluid interactions,
                # long-range particle interactions and internal
                # sources terms. Each particle rung will be kicked by
                # half a sub-step. It is assumed that the drifting and
                # kicking of all components is synchronized. As this
                # does not count as an actual time step,
                # the universal time will not be updated.
                # Apply initial half kick to fluids, initial half
                # long-range kick to particles and inital half
                # application of internal sources.
                kick_long(components, Δt, sync_time, 'init')
                # Sort particles in memory so that the order matches
                # the visiting order when iterating through all subtiles
                # within tiles, improving the performance of
                # CPU caching. For the in-memory sorting, the Δmom
                # buffers will be used. It is then important that these
                # do not currently contain information needed later.
                # Note that for N_rungs = 1, the tiles and subtiles are
                # not yet instantiated if this is the first time step,
                # as no fake short-range kick has been performed prior
                # to the main time loop.
                if 𝔹[particle_reordering]:
                    for component in components:
                        if not subtiling_computation_times[component]:
                            continue
                        if 𝔹[particle_reordering == 'deterministic']:
                            # If multiple tilings+subtilings exist on a
                            # component, the sorting will be done with
                            # respect to the first subtiling
                            # encountered.
                            for subtiling_name in component.tilings:
                                match = re.search(r'(.*) \(subtiles\)', subtiling_name)
                                if not match:
                                    continue
                                interaction_name = match.group(1)
                                break
                        else:
                            # If multiple tilings+subtilings exist on a
                            # component, the sorting will be done with
                            # respect to the subtiling with the highest
                            # recorded computation time. Note that the
                            # same component might then be sorted
                            # according to different subtilings on
                            # different processes.
                            interaction_name = collections.Counter(
                                subtiling_computation_times[component]
                            ).most_common(1)[0][0]
                        tiling_name    = f'{interaction_name} (tiles)'
                        subtiling_name = f'{interaction_name} (subtiles)'
                        component.tile_sort(tiling_name, None, -1, subtiling_name)
                        # Reset subtile computation time
                        subtiling_computation_times[component].clear()
                # Initial half short-range kick of particles on
                # all rungs. No rung jumps will occur, as any such
                # have been nullified by the call
                # to Component.assign_rungs() above.
                kick_short(components, Δt)
                # Check whether next dump is within 1.5*Δt
                if dump_time.t - universals.t <= 1.5*Δt:
                    # Next base step should synchronize at dump time
                    sync_time = dump_time.t
                    continue
                # Check whether the base time step needs to be reduced
                Δt_max, bottleneck = get_base_timestep_size(components, static_timestepping_func)
                if Δt > Δt_max:
                    # Next base step should synchronize.
                    # Thereafter we can lower the base time step size.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
            elif time_step_type == 'full':
                # This is a full base time step of size Δt.
                # All components will be drifted and kicked Δt.
                # The kicks will start and end half a time step ahead
                # of the drifts.
                # Drift fluids.
                drift_fluids(components, Δt, sync_time)
                # Continually perform interlaced drift and kick
                # operations of the short-range particle rungs, until
                # the particles are drifted forward to the exact time of
                # the next base time step (Δt away) and kicked half a
                # sub-step (of size Δt/2**(rung_index + 1)) into the
                # next base time step.
                driftkick_short(components, Δt, sync_time)
                # All drifting is now exactly at the next base time
                # step, while the long-range kicks are lagging behind.
                # Before doing the long-range kicks, set the universal
                # time and scale factor to match the current position of
                # the long-range kicks, so that various time averages
                # will be over the kick step.
                universals.t += 0.5*Δt
                if universals.t + Δt_reltol*Δt + ℝ[2*machine_ϵ] > sync_time:
                    universals.t = sync_time
                universals.a = scale_factor(universals.t)
                # Apply full kick to fluids, full long-range kick to
                # particles and fully apply internal sources.
                kick_long(components, Δt, sync_time, 'full')
                # Set universal time and scale factor to match end of
                # this base time step (the location of drifts).
                universals.t += 0.5*Δt
                if universals.t + Δt_reltol*Δt + ℝ[2*machine_ϵ] > sync_time:
                    universals.t = sync_time
                universals.a = scale_factor(universals.t)
                # Check whether we are at sync time
                if universals.t == sync_time:
                    # We are at sync time. Base time step completed.
                    # Reset time_step_type and sync_time
                    time_step_type = 'init'
                    sync_time = ထ
                    # If Δt has been momentarily lowered just to reach
                    # the sync time, the true value is stored in
                    # Δt_backup. Here we undo this lowering.
                    if Δt_backup != -1:
                        if Δt < Δt_backup:
                            Δt = Δt_backup
                        Δt_backup = -1
                    # Reduce base time step if necessary.
                    # If not, increase it as allowed.
                    if recompute_Δt_max:
                        Δt_max, bottleneck = get_base_timestep_size(
                            components, static_timestepping_func,
                        )
                    recompute_Δt_max = True
                    Δt, bottleneck = update_base_timestep_size(
                        Δt, Δt_min, Δt_max, bottleneck, time_step, time_step_last_sync,
                        tolerate_danger=(bottleneck == bottleneck_static_timestepping),
                    )
                    # Update time step counters
                    time_step += 1
                    time_step_last_sync = time_step
                    # If it is time, perform autosave
                    with unswitch:
                        if autosave_interval > 0:
                            if bcast(time() - autosave_time > ℝ[autosave_interval/units.s]):
                                autosave(components, time_step, Δt, Δt_begin)
                                autosave_time = time()
                    # Dump output if at dump time
                    if universals.t == dump_time.t:
                        if dump(components, output_filenames, dump_time, Δt):
                            # The "dump" was really the activation of a
                            # component. This new component might need
                            # a reduced time step size.
                            initial_fac_times.add(universals.t)
                            Δt_max, bottleneck = get_base_timestep_size(
                                components, static_timestepping_func,
                            )
                            Δt, bottleneck = update_base_timestep_size(
                                Δt, Δt_min, Δt_max, bottleneck,
                                allow_increase=False, tolerate_danger=True,
                            )
                        # Ensure that we have at least 1.5
                        # base time steps before the next dump.
                        if dump_index != len(dump_times) - 1:
                            Δt_max = (dump_times[dump_index + 1].t - universals.t)/1.5
                            if Δt > Δt_max:
                                # We are now lowering Δt in order to
                                # reach the next dump time exactly. Once
                                # the dump is completed, this lowering
                                # of Δt should be undone, and so we take
                                # a backup of the actual Δt.
                                Δt_backup = Δt
                                Δt = Δt_max
                        # Break out of the infinite loop,
                        # proceeding to the next dump time.
                        break
                    # Not at dump time.
                    # Ensure that we have at least 1.5
                    # base time steps before we reach the dump time.
                    Δt_max = (dump_time.t - universals.t)/1.5
                    if Δt > Δt_max:
                        # We are now lowering Δt in order to reach the
                        # next dump time exactly. Once the dump is
                        # completed, this lowering of Δt should be
                        # undone, and so we take a backup
                        # of the actual Δt.
                        Δt_backup = Δt
                        Δt = Δt_max
                    # Go to init step
                    continue
                # Check whether next dump is within 1.5*Δt
                if dump_time.t - universals.t <= 1.5*Δt:
                    # We need to synchronize at dump time
                    sync_time = dump_time.t
                    continue
                # Check whether the base time step needs to be reduced
                Δt_max, bottleneck = get_base_timestep_size(components, static_timestepping_func)
                if Δt > Δt_max:
                    # We should synchronize, whereafter the
                    # base time step size can be lowered.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
                # Check whether the base time step should be increased
                if (Δt_max > ℝ[1 + Δt_period_increase_min_fac]*Δt
                    and (time_step + 1 - time_step_last_sync) >= Δt_period
                ):
                    # We should synchronize, whereafter the
                    # base time step size can be raised.
                    sync_time = universals.t + 0.5*Δt
                    recompute_Δt_max = False
                    continue
                # Base time step completed
                time_step += 1
    # All dumps completed; end of main time loop
    print_timestep_footer(components)
    print_timestep_heading(time_step, Δt, bottleneck, components, end=True)
    # Remove dumped autosave snapshot, if any
    if master:
        autosave_filename = f'{autosave_dir}/autosave_{jobid}.hdf5'
        if os.path.isfile(autosave_filename):
            os.remove(autosave_filename)
# Set of (cosmic) times at which the maximum time step size Δt_max
# should be further scaled by Δt_initial_fac, which are at the initial
# time step as well as right after component activations. This set is
# populated by timeloop() and queried by get_base_timestep_size().
cython.declare(initial_fac_times=set)
initial_fac_times = set()
# Dict containing list of scale factor values at which to activate or
# terminate components (the 't' list is never used).
# The list is populated by the prepare_for_output() function.
cython.declare(activation_termination_times=dict)
activation_termination_times = {'t': (), 'a': ()}
# List of currently passive components,
# and list of all component names in order.
# These will be populated by the timeloop() function,
# and passive_components will be mutated
# by the activate_terminate() function.
cython.declare(
    passive_components=list,
    components_order=list,
)
passive_components = []
components_order = []

# Function preparing for static time-stepping. If static time-stepping
# is to be used, this will return a newly constructed
# callable Δt_max(a). Otherwise, None is returned.
def prepare_static_timestepping():
    static_timestepping_func = None
    if not master:
        # Ask master whether static time-stepping is to be used
        if bcast():
            # Static time-stepping is to be used.
            # Only the master process holds the time-stepping data,
            # so the function to return should simply receive a result
            # from the master.
            static_timestepping_func = lambda a=-1: bcast()
        return static_timestepping_func
    # Only the master process handles the below preparation
    apply_static_timestepping = False
    if static_timestepping is None:
        # Do not use static time-stepping
        pass
    elif isinstance(static_timestepping, str):
        if os.path.exists(static_timestepping):
            # The static_timestepping parameter holds the path to an
            # existing file. This file should have been produced by
            # a previous simulation and store (a, Δa) data.
            apply_static_timestepping = True
            # Load and clean up data
            static_timestepping_a, static_timestepping_Δa = np.loadtxt(
                static_timestepping,
                unpack=True,
            )
            static_timestepping_a  = static_timestepping_a .copy()  # ensure contiguousness
            static_timestepping_Δa = static_timestepping_Δa.copy()  # ensure contiguousness
            static_timestepping_a, static_timestepping_Δa = remove_doppelgängers(
                static_timestepping_a, static_timestepping_Δa,
                rel_tol=Δt_reltol,
            )
            # Construct scale factor intervals
            # of monotonically increasing Δa.
            mask = (np.diff(static_timestepping_Δa) < 0)
            for index in range(1, len(mask)):
                mask[index] &= not mask[index - 1]
            mask[-1] = False
            interval_indices = list(np.where(mask)[0] + 1)
            a_intervals = []
            a_right = 0
            for index in interval_indices:
                a_left, a_right = a_right, static_timestepping_a[index]
                a_intervals.append((a_left, a_right))
            interval_indices.append(static_timestepping_a.shape[0])
            a_left, a_right = a_right, ထ
            a_intervals.append((a_left, a_right))
            # Create linear spline for each interval
            static_timestepping_interps = []
            index_left = 0
            for index_right in interval_indices:
                static_timestepping_interps.append(
                    lambda a, *, f=scipy.interpolate.interp1d(
                        np.log(static_timestepping_a [index_left:index_right]),
                        np.log(static_timestepping_Δa[index_left:index_right]),
                        'linear',
                        fill_value='extrapolate',
                    ): exp(float(f(log(a))))
                )
                index_left = index_right
            # Create function Δt(a) implementing the static
            # time-stepping using the above splines.
            def static_timestepping_func(a=-1):
                if a == -1:
                    a, t = universals.a, universals.t
                else:
                    t = cosmic_time(a)
                # Find the scale factor tabulation interval
                # and look up the interpolated function.
                # As the number of intervals ought always to
                # be small, a simple linear search is sufficient.
                for (a_left, a_right), static_timestepping_interp in zip(
                    a_intervals, static_timestepping_interps,
                ):
                    # With a very close to a_right, it is a good guess
                    # that we really ought to have a == a_right, but
                    # that floating-point imprecisions have ruined the
                    # exact equality. If so, this a really belongs to
                    # the next interval.
                    if a_right != ထ and isclose(float(a), float(a_right)):
                        continue
                    if isclose(float(a), float(a_left + machine_ϵ)):
                        a = a_left
                    if a_left <= a < a_right:
                        break
                else:
                    abort(f'static_timestepping_func(): a = {a} not in any interval')
                # Do the interpolation and convert Δa to Δt
                Δa = static_timestepping_interp(a)
                a_next = a + Δa
                Δt = cosmic_time(a_next) - t if a_next <= 1 else ထ
                return bcast(Δt)
        else:
            # The static_timestepping parameter does not refer to an
            # existing path. Interpret it as a path to a not yet
            # existing file. The time stepping of this simulation
            # will be written to this file.
            pass
    elif callable(static_timestepping):
        # Create function Δt(a) implementing the static
        # time-stepping using the callable.
        apply_static_timestepping = True
        def static_timestepping_func(a=-1):
            if a == -1:
                a, t = universals.a, universals.t
            else:
                t = cosmic_time(a)
            # Compute Δa using the user-supplied callable
            Δa = static_timestepping(a)
            # Convert to Δt
            a_next = a + Δa
            Δt = cosmic_time(a_next) - t if a_next <= 1 else ထ
            return bcast(Δt)
    else:
        abort(
            f'Could not interpret static_timestepping = {static_timestepping} '
            f'of type {type(static_timestepping)}'
        )
    # Inform slaves whether static timestepping is to be applied
    bcast(apply_static_timestepping)
    return static_timestepping_func

# Function for computing the size of the base time step
@cython.header(
    # Arguments
    components=list,
    static_timestepping_func=object,  # callable or None
    # Locals
    H='double',
    a='double',
    bottleneck=str,
    component='Component',
    extreme_force=str,
    force=str,
    gridsize='Py_ssize_t',
    key=tuple,
    measurements=dict,
    method=str,
    n='int',
    resolution='Py_ssize_t',
    scale='double',
    t='double',
    v_max='double',
    v_rms='double',
    Δa_max='double',
    Δt_courant='double',
    Δt_decay='double',
    Δt_dynamical='double',
    Δt_hubble='double',
    Δt_max='double',
    Δt_pm='double',
    Δt_ẇ='double',
    Δx_max='double',
    ρ_bar='double',
    ρ_bar_component='double',
    returns=tuple,
)
def get_base_timestep_size(components, static_timestepping_func=None):
    """This function computes the maximum allowed size
    of the base time step Δt. The time step limiters come in three
    categories; global limiters, component limiters and
    particle/fluid element limiters. For each limiter, the value of Δt
    should not exceed a small fraction of the following.
    Background limiters:
      Global background limiters:
      - The dynamical time scale.
      - The Hubble time (≃ present age of the universe)
        if Hubble expansion is enabled.
      Component background limiters:
      - 1/abs(ẇ) for every component, so that the transition from
        relativistic to non-relativistic happens smoothly.
      - The reciprocal decay rate of each matter component, weighted
        with their current total mass (or background density) relative
        to all matter.
    Non-linear limiters:
    - For fluid components (with a Boltzmann hierarchy closed after J
      (velocity)): The time it takes for the fastest fluid element to
      traverse a fluid cell, i.e. the Courant condition.
    - For particle/fluid components using the PM method: The time it
      would take to traverse a PM grid cell for a particle/fluid element
      with the rms velocity of all particles/fluid elements within a
      given component.
    - For particle components using the P³M method: The time it
      would take to traverse the long/short-range force split scale for
      a particle with the rms velocity of all particles within a
      given component.
    The return value is a tuple containing the maximum allowed Δt and a
    str stating which limiter is the bottleneck.
    If a callable static_timestepping_func is given, this is used obtain
    Δt_max directly, ignoring all time step limiters.
    """
    t = universals.t
    a = universals.a
    # If a static_timestepping_func is given, use this to
    # determine Δt_max, short-circuiting this function.
    if static_timestepping_func is not None:
        Δt_max = static_timestepping_func(a)
        bottleneck = bottleneck_static_timestepping
        return Δt_max, bottleneck
    H = hubble(a)
    Δt_max = ထ
    bottleneck = ''
    # Local cache for calls to measure()
    measurements = {}
    # The dynamical time scale
    ρ_bar = 0
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        ρ_bar += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
    Δt_dynamical = fac_dynamical/(sqrt(G_Newton*ρ_bar) + machine_ϵ)
    if Δt_dynamical < Δt_max:
        Δt_max = Δt_dynamical
        bottleneck = 'the dynamical timescale'
    # The Hubble time
    if enable_Hubble:
        Δt_hubble = fac_hubble/H
        if Δt_hubble < Δt_max:
            Δt_max = Δt_hubble
            bottleneck = 'the Hubble time'
    # 1/abs(ẇ)
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        Δt_ẇ = fac_ẇ/(abs(cast(component.ẇ(a=a), 'double')) + machine_ϵ)
        if Δt_ẇ < Δt_max:
            Δt_max = Δt_ẇ
            bottleneck = f'ẇ of {component.name}'
    # Reciprocal decay rate
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        ρ_bar_component = component.ϱ_bar*a**(-3*(1 + component.w_eff(a=a)))
        Δt_decay = fac_Γ/(abs(component.Γ(a)) + machine_ϵ)*ρ_bar/ρ_bar_component
        if Δt_decay < Δt_max:
            Δt_max = Δt_decay
            bottleneck = f'decay rate of {component.name}'
    # Courant condition for fluid elements
    for component in components:
        if component.representation == 'particles':
            continue
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        # Find maximum propagation speed of fluid
        key = (component, 'v_max')
        v_max = measurements[key] = (
            measurements[key] if key in measurements else measure(component, 'v_max')
        )
        # In the odd case of a completely static component,
        # set v_max to be just above 0.
        if v_max == 0:
            v_max = machine_ϵ
        # The Courant condition
        Δx_max = boxsize/component.gridsize
        Δt_courant = fac_courant*Δx_max/v_max
        if Δt_courant < Δt_max:
            Δt_max = Δt_courant
            bottleneck = f'the Courant condition for {component.name}'
    # PM limiter
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        # Find PM resolution for this component
        resolution = 0
        for force, method in component.forces.items():
            if method != 'pm':
                continue
            for method, gridsizes in component.potential_gridsizes[force].items():
                if method != 'pm':
                    continue
                gridsize = np.max(gridsizes)
                if gridsize > resolution:
                    resolution = gridsize
                    extreme_force = force
        if resolution == 0:
            continue
        # Find rms bulk velocity, i.e. do not add the sound speed
        key = (component, 'v_rms')
        v_rms = measurements[key] = (
            measurements[key] if key in measurements else measure(component, 'v_rms')
        )
        if component.representation == 'fluid':
            v_rms -= light_speed*sqrt(component.w(a=a))/a
        # In the odd case of a completely static component,
        # set v_rms to be just above 0.
        if v_rms < machine_ϵ:
            v_rms = machine_ϵ
        # The PM limiter
        Δx_max = boxsize/resolution
        Δt_pm = fac_pm*Δx_max/v_rms
        if Δt_pm < Δt_max:
            Δt_max = Δt_pm
            bottleneck = f'the PM method of the {extreme_force} force for {component.name}'
    # P³M limiter
    for component in components:
        if component.representation == 'fluid' and component.is_linear(0):
            continue
        # Find P³M resolution for this component.
        # The P³M method is only implemented for gravity.
        scale = ထ
        for force, method in component.forces.items():
            if method != 'p3m':
                continue
            if force != 'gravity':
                abort(
                    f'Force "{force}" with method "P³M" unknown to get_base_timestep_size()'
                )
            if ℝ[shortrange_params['gravity']['scale']] < scale:
                scale = ℝ[shortrange_params['gravity']['scale']]
                extreme_force = 'gravity'
        if scale == ထ:
            continue
        # Find rms velocity
        key = (component, 'v_rms')
        v_rms = measurements[key] = (
            measurements[key] if key in measurements else measure(component, 'v_rms')
        )
        # In the odd case of a completely static component,
        # set v_rms to be just above 0.
        if v_rms < machine_ϵ:
            v_rms = machine_ϵ
        # The P³M limiter
        Δx_max = scale
        Δt_p3m = fac_p3m*Δx_max/v_rms
        if Δt_p3m < Δt_max:
            Δt_max = Δt_p3m
            bottleneck = f'the P³M method of the {extreme_force} force for {component.name}'
    # Reduce the found Δt_max by Δt_initial_fac
    # if we are at a time which demands this reduction.
    if t in initial_fac_times:
        Δt_max *= Δt_initial_fac
    # Record static time-stepping to disk
    if master and isinstance(static_timestepping, str):
        if t + Δt_max < cosmic_time(1):
            Δa_max = scale_factor(t + Δt_max) - a
            n = int(ceil(log10(1/Δt_reltol) + 0.5))
            with open(static_timestepping, 'a', encoding='utf-8') as f:
                if f.tell() == 0:
                    f.write(unicode(
                        '#{}a{}Δa\n'.format(' '*((n + 3)//2), ' '*(n + 6))
                    ))
                f.write(f'{{:.{n}e}} {{:.{n}e}}\n'.format(a, Δa_max))
    # Return maximum allowed base time step size and the bottleneck
    return Δt_max, bottleneck
# Constant used by the get_base_timestep_size() function
cython.declare(bottleneck_static_timestepping=str)
bottleneck_static_timestepping = 'static time-stepping'

# Function for computing the updated value of Δt.
# This has to be a pure Python function due to the use
# of keyword-only arguments.
def update_base_timestep_size(
    Δt, Δt_min, Δt_max, bottleneck,
    time_step=-1, time_step_last_sync=-1,
    *, allow_increase=True, tolerate_danger=False,
):
    # Reduce base time step size Δt if necessary
    if Δt > Δt_max:
        Δt_new = Δt_reduce_fac*Δt_max
        Δt_ratio = Δt_new/Δt
        # Inform about reduction to the base time step
        message = f'Rescaling time step size by a factor {Δt_ratio:.1g} due to {bottleneck}'
        if Δt_ratio < Δt_ratio_abort:
            if tolerate_danger:
                masterwarn(message)
            else:
                message = (
                    f'Due to {bottleneck}, the time step size '
                    f'needs to be rescaled by a factor {Δt_ratio:.1g}. '
                    f' This extreme change is unacceptable.'
                )
                abort(message)
        elif Δt_ratio < Δt_ratio_warn:
            if tolerate_danger:
                masterprint(message)
            else:
                masterwarn(message)
        if Δt_new < Δt_min:
            # Never tolerate this
            abort(
                f'Time evolution effectively halted with a time step size '
                f'of {Δt_new} {unit_time} (at the start of the simulation '
                f'the time step size was {Δt_begin} {unit_time})'
            )
        # Apply reduction
        Δt = Δt_new
        return Δt, bottleneck
    if not allow_increase:
        return Δt, bottleneck
    # Construct new base time step size Δt_new,
    # making sure that its relative change is not too big.
    Δt_new = Δt_increase_fac*Δt_max
    if Δt_new < Δt:
        Δt_new = Δt
    period_frac = (time_step + 1 - time_step_last_sync)*ℝ[1/Δt_period]
    if period_frac > 1:
        period_frac = 1
    elif period_frac < 0:
        period_frac = 0
    Δt_tmp = (1 + period_frac*Δt_period_increase_max_fac)*Δt
    if Δt_new > Δt_tmp:
        Δt_new = Δt_tmp
    # If close to a = 1, leave Δt as is
    if enable_Hubble and universals.t + Δt_new > cosmic_time(1):
        bottleneck = 'a ≈ 1'
        return Δt, bottleneck
    # Reject Δt_new if accepting it brings Δa above Δa_max_increasing
    if enable_Hubble and Δa_max_increasing > 0:
        Δa_new = scale_factor(universals.t + Δt_new) - universals.a
        if Δa_new > Δa_max_increasing:
            bottleneck = 'Δa_max_increasing'
            return Δt, bottleneck
    # Accept Δt_new. As the base time step size
    # has been increased, there is no bottleneck.
    bottleneck = ''
    return Δt_new, bottleneck

# Function for computing all time step integrals
# between two specified cosmic times.
@cython.header(
    # Arguments
    t_start='double',
    t_end='double',
    components=list,
    # Locals
    component='Component',
    component_name=str,
    component_names=tuple,
    enough_info='bint',
    integrals='double[::1]',
    integrand=object,  # str or tuple
    integrands=tuple,
    returns=dict,
)
def get_time_step_integrals(t_start, t_end, components):
    # The first time this function is called, the global ᔑdt_scalar
    # and ᔑdt_rungs gets populated.
    if not ᔑdt_scalar:
        integrands = (
            # Global integrands
            '1',
            'a**2',
            'a**(-1)',
            'a**(-2)',
            'ȧ/a',
            # Single-component integrands
            *[(integrand, component.name) for component, in itertools.product(*[components]*1)
                for integrand in (
                    'a**(-3*w_eff)',
                    'a**(-3*(1+w_eff))',
                    'a**(-3*w_eff-1)',
                    'a**(3*w_eff-2)',
                    'a**(2-3*w_eff)',
                    'a**(-3*w_eff)*Γ/H',
                )
            ],
            # Two-component integrands
            *[(integrand, component_0.name, component_1.name)
                for component_0, component_1 in itertools.product(*[components]*2)
                for integrand in (
                    'a**(-3*w_eff₀-3*w_eff₁-1)',
                )
            ]
        )
        # Populate scalar dict
        for integrand in integrands:
            ᔑdt_scalar[integrand] = 0
        # For the rungs dict, we need an integral for each rung,
        # of which there are N_rungs. Additionally, we need a value
        # of the integral for jumping down/up a rung, for each rung,
        # meaning that we need 3*N_rungs integrals. The integral
        # for a normal kick of rung rung_index is then stored in
        # ᔑdt_rungs[integrand][rung_index], while the integral for
        # jumping down from rung rung_index to rung_index - 1 is stored
        # in ᔑdt_rungs[integrand][rung_index + N_rungs], while the
        # integral for jumping up from rung_index to rung_index + 1 is
        # stored in ᔑdt_rungs[integrand][rung_index + 2*N_rungs]. Since
        # a particle at rung 0 cannot jump down and a particle at rung
        # N_rungs - 1 cannot jump up, indices 0 + N_rungs = N_rungs
        # and N_rungs - 1 + 2*N_rungs = 3*N_rungs - 1 are unused.
        # We allocate 3*N_rungs - 1 integrals, leaving the unused
        # index N_rungs be, while the unused index 3*N_rungs - 1
        # will be out of bounce.
        for integrand in integrands:
            ᔑdt_rungs[integrand] = zeros(3*N_rungs - 1, dtype=C2np['double'])
    # Fill ᔑdt_scalar with integrals
    for integrand in ᔑdt_scalar.keys():
        # If the passed components are only a subset of all components
        # present in the simulation, some integrals cannot be computed.
        # This is OK, as presumably the caller it not interested in
        # these anyway. Store NaN if the current integrand cannot be
        # computed for this reason.
        if isinstance(integrand, tuple):
            enough_info = True
            component_names = integrand[1:]
            for component_name in component_names:
                for component in components:
                    if component_name == component.name:
                        break
                else:
                    enough_info = False
                    break
            if not enough_info:
                ᔑdt_scalar[integrand] = NaN
                continue
        # Compute integral
        with unswitch:
            if t_start == t_end:
                ᔑdt_scalar[integrand] = 0
            else:
                with unswitch:
                    if not enable_class_background:
                        expand(
                            scale_factor(t_start),
                            t_start,
                            ℝ[t_end - t_start],
                        )
                ᔑdt_scalar[integrand] = scalefactor_integral(
                    integrand, t_start, ℝ[t_end - t_start], components,
                )
    # Return the global ᔑdt_scalar
    return ᔑdt_scalar
# Dict returned by the get_time_step_integrals() function,
# storing a single time step integral for each integrand.
cython.declare(ᔑdt_scalar=dict)
ᔑdt_scalar = {}
# Dict storing time step integrals for each rung,
# indexed as ᔑdt_rungs[integrand][rung_index].
cython.declare(ᔑdt_rungs=dict)
ᔑdt_rungs = {}

# Function which perform long-range kicks on all components
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    sync_time='double',
    step_type=str,
    # Locals
    a_start='double',
    a_end='double',
    component='Component',
    force=str,
    method=str,
    printout='bint',
    receivers=list,
    suppliers=list,
    t_end='double',
    t_start='double',
    ᔑdt=dict,
    returns='void',
)
def kick_long(components, Δt, sync_time, step_type):
    """We take into account three different cases of long-range kicks:
    - Internal source terms (fluid and particle components).
    - Interactions acting on fluids (only PM implemented).
    - Long-range interactions acting on particle components,
      i.e. PM and the long-range part of P³M.
    This function can operate in two separate modes:
    - step_type == 'init':
      The kick is over the first half of the base time step of size Δt.
    - step_type == 'full':
      The kick is over the second half of the base time step of size Δt
      as well as over an equally sized portion of the next time step.
      Here it is expected that universals.t and universals.t matches the
      long-range kicks, so that it is in between the current and next
      time step.
    """
    # Get time step integrals over half ('init')
    # or whole ('full') time step.
    t_start = universals.t
    t_end = t_start + (Δt/2 if step_type == 'init' else Δt)
    if t_end + Δt_reltol*Δt + ℝ[2*machine_ϵ] > sync_time:
        t_end = sync_time
    if t_start == t_end:
        return
    ᔑdt = get_time_step_integrals(t_start, t_end, components)
    # Realize all linear fluid scalars which are not components
    # of a tensor. This comes down to ϱ and 𝒫.
    a_start = universals.a
    a_end = scale_factor(t_end)
    for component in components:
        component.realize_if_linear(0,  # ϱ
            specific_multi_index=0, a=a_start, a_next=a_end,
        )
        component.realize_if_linear(2,  # 𝒫
            specific_multi_index='trace', a=a_start, a_next=a_end,
        )
    # Apply the effect of all internal source terms
    for component in components:
        component.apply_internal_sources(ᔑdt, a_end)
    # Find all long-range interactions
    interactions_list = interactions.find_interactions(components, 'long-range')
    # Invoke each long-range interaction sequentially
    printout = True
    for force, method, receivers, suppliers in interactions_list:
        getattr(interactions, force)(method, receivers, suppliers, ᔑdt, 'long-range', printout)

# Function which kicks all short-range rungs a single time
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    fake='bint',
    # Locals
    component='Component',
    force=str,
    highest_populated_rung='signed char',
    integrand=object,  # str or tuple
    interactions_instantaneous_list=list,
    interactions_list=list,
    interactions_noninstantaneous_list=list,
    method=str,
    particle_components=list,
    printout='bint',
    receivers=list,
    receivers_all=set,
    rung_index='signed char',
    suppliers=list,
    t_end='double',
    t_start='double',
    tiling='Tiling',
    ᔑdt_rung=dict,
    returns='void',
)
def kick_short(components, Δt, fake=False):
    """The kick is over the first half of the sub-step for each rung.
    A sub-step for rung rung_index is 1/2**rung_index as long as the
    base step of size Δt, and so half a sub-step is
    1/2**(rung_index + 1) of the base step. If fake is True, the kick is
    still carried out, but no momentum updates will be applied.
    """
    # Collect all particle components. Do nothing if none exists.
    particle_components = [
        component for component in components if component.representation == 'particles'
    ]
    if not particle_components:
        return
    # Find all short-range interactions. Do nothing if none exists.
    interactions_instantaneous_list = interactions.find_interactions(
        particle_components, 'short-range', instantaneous=True,
    )
    interactions_noninstantaneous_list = interactions.find_interactions(
        particle_components, 'short-range', instantaneous=False,
    )
    interactions_list = interactions_instantaneous_list + interactions_noninstantaneous_list
    if not interactions_list:
        return
    # As we only do a single, simultaneous interaction for all rungs,
    # we must flag all (populated) rungs as active.
    for component in particle_components:
        component.lowest_active_rung = component.lowest_populated_rung
    # Get the highest populated rung amongst all components
    # and processes.
    highest_populated_rung = allreduce(
        np.max([component.highest_populated_rung for component in particle_components]),
        op=MPI.MAX,
    )
    # Though the size of the time interval over which to kick is
    # different for each rung, we only perform a single interaction
    # for each pair of components and short-range forces.
    # We then need to know all time step integrals for
    # each integrand simultaneously.
    # We store these in the global ᔑdt_rungs.
    t_start = universals.t
    for rung_index in range(highest_populated_rung + 1):
        t_end = t_start + Δt/2**(rung_index + 1)
        ᔑdt_rung = get_time_step_integrals(t_start, t_end, particle_components)
        for integrand, integral in ᔑdt_rung.items():
            ᔑdt_rungs[integrand][rung_index] = integral
    # Invoke short-range interactions, assign rungs and apply momentum
    # updates depending on whether this is a fake call or not.
    printout = True
    receivers_all = {  # Really only receivers of non-instantaneous interactions
        receiver
        for force, method, receivers, suppliers in interactions_noninstantaneous_list
        for receiver in receivers
    }
    if fake:
        # Carry out fake non-instantaneous interactions in order to
        # determine the particle rungs. Skip instantaneous interactions.
        for component in particle_components:
            component.nullify_Δ('mom')
        for force, method, receivers, suppliers in interactions_noninstantaneous_list:
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt_rungs, 'short-range', printout,
            )
        for component in receivers_all:
            component.convert_Δmom_to_acc(ᔑdt_rungs)
        for component in particle_components:
            component.assign_rungs(Δt, fac_softening)
        # Reset all computation_time_total tiling attributes,
        # as the above non-instantaneous-only interactions
        # should not be counted.
        for component in particle_components:
            for tiling in component.tilings.values():
                tiling.computation_time_total = 0
    else:
        # Carry out instantaneous short-range interactions.
        # Any momentum updates will be applied.
        for force, method, receivers, suppliers in interactions_instantaneous_list:
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt_rungs, 'short-range', printout,
            )
        # Ensure nullified Δ buffers on all particle components
        for component in particle_components:
            component.nullify_Δ('mom')
        # Carry out non-instantaneous short-range interactions
        for force, method, receivers, suppliers in interactions_noninstantaneous_list:
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt_rungs, 'short-range', printout,
            )
        for component in receivers_all:
            component.apply_Δmom()
            component.convert_Δmom_to_acc(ᔑdt_rungs)

# Function which drifts all fluid components
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    sync_time='double',
    # Locals
    a_end='double',
    component='Component',
    fluid_components=list,
    t_end='double',
    t_start='double',
    ᔑdt=dict,
    returns='void',
)
def drift_fluids(components, Δt, sync_time):
    """This function always drift over a full base time step.
    """
    # Collect all fluid components. Do nothing if none exists.
    fluid_components = [
        component for component in components if component.representation == 'fluid'
    ]
    if not fluid_components:
        return
    # Get time step integrals over entire time step
    t_start = universals.t
    t_end = t_start + Δt
    if t_end + Δt_reltol*Δt + ℝ[2*machine_ϵ] > sync_time:
        t_end = sync_time
    if t_start == t_end:
        return
    ᔑdt = get_time_step_integrals(t_start, t_end, fluid_components)
    # Drift all fluid components sequentially
    a_end = scale_factor(t_end)
    for component in fluid_components:
        component.drift(ᔑdt, a_end)

# Function which performs interlaced drift and kick operations
# on the short-range rungs.
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    sync_time='double',
    # Locals
    any_kicks='bint',
    any_rung_jumps='bint',
    any_rung_jumps_arr='int[::1]',
    component='Component',
    driftkick_index='Py_ssize_t',
    force=str,
    highest_populated_rung='signed char',
    i='Py_ssize_t',
    index_end='Py_ssize_t',
    index_start='Py_ssize_t',
    integral='double',
    integrals='double[::1]',
    integrand=object,  # str or tuple
    interactions_instantaneous_list=list,
    interactions_list=list,
    interactions_noninstantaneous_list=list,
    lines=list,
    lowest_active_rung='signed char',
    message=list,
    method=str,
    n='Py_ssize_t',
    n_interactions=object,  # collections.defaultdict
    pair=set,
    pairs=list,
    particle_components=list,
    printout='bint',
    receiver='Component',
    receivers=list,
    receivers_all=set,
    rung_index='signed char',
    suppliers=list,
    t_end='double',
    t_start='double',
    text=str,
    ᔑdt=dict,
    ᔑdt_rung=dict,
    returns='void',
)
def driftkick_short(components, Δt, sync_time):
    """Every rung is fully drifted and kicked over a complete base time
    step of size Δt. Rung rung_index will be kicked 2**rung_index times.
    All rungs will be drifted synchronously in steps
    of Δt/2**(N_rungs - 1), i.e. each drift is over two half sub-steps.
    The first drift will start at the beginning of the base step.
    The kicks will vary in size for the different rungs. Rung rung_index
    will be kicked Δt/2**rung_index in each kick operation, i.e. a whole
    sub-step for the highest rung (N_rungs - 1), two sub-steps for the
    rung below, four sub-steps for the rung below that, and so on.
    It as assumed that all rungs have already been kicked so that
    these are half a kick-sized step ahead of the drifts. Thus, the
    kick position of the highest rung is already half a sub-step into
    the base time step, the rung below is two half sub-steps into the
    base time step, the rung below that is four half sub-steps into the
    base step, and so on.
    The drifts and kicks follow this rhythm:
      - drift all
      - kick rung  (N_rungs - 1)
      - drift all
      - kick rungs (N_rungs - 1), (N_rungs - 2)
      - drift all
      - kick rung  (N_rungs - 1)
      - drift all
      - kick rungs (N_rungs - 1), (N_rungs - 2), (N_rungs - 3)
      - ...
    Thus the highest rung participates in all kicks, the one below only
    in every other kick, the one below that only in every fourth kick,
    and so on.
    """
    # Collect all particle components. Do nothing if none exists.
    particle_components = [
        component for component in components if component.representation == 'particles'
    ]
    if not particle_components:
        return
    # Find all short-range interactions
    interactions_instantaneous_list = interactions.find_interactions(
        particle_components, 'short-range', instantaneous=True,
    )
    interactions_noninstantaneous_list = interactions.find_interactions(
        particle_components, 'short-range', instantaneous=False,
    )
    interactions_list = interactions_instantaneous_list + interactions_noninstantaneous_list
    # In case of no short-range interactions among the particles at all,
    # we may drift the particles in one go, after which we are done
    # within this function, as the long-range kicks
    # are handled elsewhere.
    if not interactions_list:
        # Get time step integrals over entire time step
        t_start = universals.t
        t_end = t_start + Δt
        if t_end + Δt_reltol*Δt + ℝ[2*machine_ϵ] > sync_time:
            t_end = sync_time
        if t_start == t_end:
            return
        ᔑdt = get_time_step_integrals(t_start, t_end, particle_components)
        # Drift all particle components and return
        for component in particle_components:
            masterprint(f'Drifting {component.name} ...')
            component.drift(ᔑdt)
            masterprint('done')
        return
    # We have short-range interactions.
    # Prepare progress message.
    message = [
        f'Intertwining drifts of {particle_components[0].name} with '
        f'the following particle interactions:'
        if len(particle_components) == 1 else (
           'Intertwining drifts of {{{}}} with the following particle interactions:'
            .format(', '.join([component.name for component in particle_components]))
        )
    ]
    for force, method, receivers, suppliers in interactions_list:
        text = interactions.shortrange_progress_message(force, method, receivers)
        message.append(text[0].upper() + text[1:])
    printout = True
    # Container holding interaction counts
    # for instantaneous interactions.
    n_interactions = collections.defaultdict(lambda: collections.defaultdict(int))
    # Perform the interlaced drifts and kicks
    any_kicks = True
    for driftkick_index in range(ℤ[2**(N_rungs - 1)]):
        # For each value of driftkick_index, a drift and a kick should
        # be performed. The time step integrals needed are contructed
        # using index_start and index_end, which index into a
        # (non-existing) array or half sub-steps. That is, an index
        # corresponds to a time via
        # t = universals.t + Δt*index/2**N_rungs.
        if any_kicks:
            index_start = 2*driftkick_index
        # Determine the lowest active rung
        # (the lowest rung which should receive a kick).
        # All rungs above this should be kicked as well.
        for rung_index in range(N_rungs):
            if ℤ[driftkick_index + 1] % 2**(ℤ[N_rungs - 1] - rung_index) == 0:
                lowest_active_rung = rung_index
                break
        # Set lowest active rung for each component
        # and check if any kicks are to be performed.
        any_kicks = False
        for component in particle_components:
            # There is no need to have the lowest active rung
            # be below the lowest populated rung.
            if lowest_active_rung < component.lowest_populated_rung:
                component.lowest_active_rung = component.lowest_populated_rung
            else:
                component.lowest_active_rung = lowest_active_rung
            # Flag if any particles exist on active rungs
            if component.highest_populated_rung >= component.lowest_active_rung:
                any_kicks = True
        any_kicks = allreduce(any_kicks, op=MPI.LOR)
        # Skip the kick if no particles at all occupy active rungs.
        # The drift is not skipped, as t_start stays the same in the
        # next iteration.
        if not any_kicks:
            continue
        # A kick is to be performed. First we should do the drift,
        # for which we need the time step integrals.
        index_end = 2*driftkick_index + 2
        t_start = universals.t + Δt*(float(index_start)/ℤ[2**N_rungs])
        if t_start + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
            t_start = sync_time
        t_end = universals.t + Δt*(float(index_end)/ℤ[2**N_rungs])
        if t_end + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
            t_end = sync_time
        # If the time step size is zero, meaning that we are already
        # at a sync time regarding the drifts, we skip the drift but
        # do not return, as the kicks may still not be at the sync time.
        if t_end > t_start:
            ᔑdt = get_time_step_integrals(t_start, t_end, particle_components)
            for component in particle_components:
                component.drift(ᔑdt)
                # Reset lowest active rung, as process exchange of
                # particles after drifting may alter the lowest
                # populated rung.
                if lowest_active_rung < component.lowest_populated_rung:
                    component.lowest_active_rung = component.lowest_populated_rung
                else:
                    component.lowest_active_rung = lowest_active_rung
        # Get the highest populated rung amongst all components
        # and processes.
        highest_populated_rung = allreduce(
            np.max([component.highest_populated_rung for component in particle_components]),
            op=MPI.MAX,
        )
        # Particles on rungs from lowest_active_rung to
        # highest_populated_rung (inclusive) should be kicked.
        # Though the size of the time interval over which to kick is
        # different for each rung, we perform the kicks using a single
        # interaction for each pair of components and short-range
        # forces. We then need to know all of the
        # (highest_populated_rung - lowest_active_rung) time step
        # integrals for each integrand simultaneously. Here we store
        # these as ᔑdt_rungs[integrand][rung_index].
        for rung_index in range(lowest_active_rung, ℤ[highest_populated_rung + 1]):
            index_start = (
                ℤ[2**(N_rungs - 1 - rung_index)]
                + (driftkick_index//ℤ[2**(N_rungs - 1 - rung_index)]
                    )*ℤ[2**(N_rungs - rung_index)]
            )
            index_end = index_start + ℤ[2**(N_rungs - rung_index)]
            t_start = universals.t + Δt*(float(index_start)/ℤ[2**N_rungs])
            if t_start + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
                t_start = sync_time
            t_end = universals.t + Δt*(float(index_end)/ℤ[2**N_rungs])
            if t_end + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
                t_end = sync_time
            ᔑdt_rung = get_time_step_integrals(t_start, t_end, particle_components)
            for integrand, integral in ᔑdt_rung.items():
                ᔑdt_rungs[integrand][rung_index] = integral
            # We additionally need the integral for jumping down
            # from rung_index to rung_index - 1. We store this using
            # index (rung_index + N_rungs). For any given rung, such
            # a down-jump is only allowed every second kick. When
            # disallowed, we store -1.
            if rung_index > 0 and (
                (ℤ[driftkick_index + 1] - ℤ[2**(N_rungs - 1 - rung_index)]
                    ) % 2**(N_rungs - rung_index) == 0
            ):
                index_end = index_start + ℤ[2**(N_rungs - 1 - rung_index)]
                t_end = universals.t + Δt*(float(index_end)/ℤ[2**N_rungs])
                if t_end + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
                    t_end = sync_time
                ᔑdt_rung = get_time_step_integrals(t_start, t_end, particle_components)
                for integrand, integral in ᔑdt_rung.items():
                    ᔑdt_rungs[integrand][rung_index + N_rungs] = integral
            else:
                for integrals in ᔑdt_rungs.values():
                    integrals[rung_index + N_rungs] = -1
            # We additionally need the integral for jumping up
            # from rung_index to rung_index + 1.
            if rung_index < ℤ[N_rungs - 1]:
                index_end = index_start + 3*2**(ℤ[N_rungs - 2] - rung_index)
                t_end = universals.t + Δt*(float(index_end)/ℤ[2**N_rungs])
                if t_end + ℝ[Δt_reltol*Δt + ℝ[2*machine_ϵ]] > sync_time:
                    t_end = sync_time
                ᔑdt_rung = get_time_step_integrals(t_start, t_end, particle_components)
                for integrand, integral in ᔑdt_rung.items():
                    ᔑdt_rungs[integrand][rung_index + ℤ[2*N_rungs]] = integral
        # Perform short-range kicks, unless the time step size is zero
        # for all active rungs (i.e. they are all at a sync time),
        # in wich case we go to the next (drift) sub-step.  We cannot
        # just return, as all kicks may still not be at the sync time.
        integrals = ᔑdt_rungs['1']
        if sum(integrals[lowest_active_rung:ℤ[highest_populated_rung + 1]]) == 0:
            continue
        # Print out progress message if this is the first kick
        if printout:
            masterprint(message[0])
            for text in message[1:]:
                masterprint(text, indent=4, bullet='•')
            masterprint('...', indent=4, wrap=False)
            printout = False
        # Flag rung jumps and nullify Δmom.
        # Only particles currently on active rungs will be affected.
        # Whether or not any rung jumping takes place in a given
        # particle component is stored in any_rung_jumps_arr.
        any_rung_jumps_arr = zeros(len(particle_components), dtype=C2np['int'])
        for i, component in enumerate(particle_components):
            any_rung_jumps_arr[i] = (
                component.flag_rung_jumps(Δt, Δt_jump_fac, fac_softening, ᔑdt_rungs)
            )
        Allreduce(MPI.IN_PLACE, any_rung_jumps_arr, op=MPI.LOR)
        # Carry out instantaneous short-range interactions.
        # Any momentum updates will be applied.
        for force, method, receivers, suppliers in interactions_instantaneous_list:
            # Nullify interaction tally
            for receiver in receivers:
                receiver.n_interactions.clear()
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt_rungs, 'short-range', printout,
            )
            for receiver in receivers:
                for supplier_name, n in receiver.n_interactions.items():
                    n_interactions[force, receiver][supplier_name] += n
        # Ensure nullified Δ buffers on all particle components
        for component in particle_components:
            component.nullify_Δ('mom')
        # Carry out non-instantaneous short-range interactions
        receivers_all = {
            receiver
            for force, method, receivers, suppliers in interactions_noninstantaneous_list
            for receiver in receivers
        }
        for force, method, receivers, suppliers in interactions_noninstantaneous_list:
            getattr(interactions, force)(
                method, receivers, suppliers, ᔑdt_rungs, 'short-range', printout,
            )
        for receiver in receivers_all:
            receiver.apply_Δmom()
            i = particle_components.index(receiver)
            any_rung_jumps = any_rung_jumps_arr[i]
            receiver.convert_Δmom_to_acc(ᔑdt_rungs, any_rung_jumps)
        for i, component in enumerate(particle_components):
            if any_rung_jumps_arr[i]:
                component.apply_rung_jumps()
    # Finalize the progress message. If printout is True, no message
    # was ever printed (because there were no kicks).
    if not printout:
        # Print out number of instantaneous interactions
        if n_interactions:
            for i, (force, method, receivers, suppliers,
            ) in enumerate(interactions_instantaneous_list):
                masterprint(message[1 + i][:message[1 + i].index(' for ')] + ' count:')
                lines = []
                pairs = []
                for receiver in receivers:
                    for supplier in suppliers:
                        pair = {receiver, supplier}
                        if pair in pairs:
                            continue
                        pairs.append(pair)
                        n = allreduce(n_interactions[force, receiver][supplier.name], op=MPI.SUM)
                        lines.append(f'{receiver.name} ⟷ {supplier.name}: ${n}')
                masterprint('\n'.join(align_text(lines, indent=4)))
        # Finish progress message
        masterprint('done')

# Function for assigning initial rung populations
@cython.header(
    # Arguments
    components=list,
    Δt='double',
    # Locals
    component='Component',
    i='Py_ssize_t',
    plural=str,
    rung_components=list,
    returns='void',
)
def initialize_rung_populations(components, Δt):
    if Δt == 0:
        abort('Cannot initialize rung populations with Δt = 0')
    # Collect all components that makes use of rungs.
    # Do nothing if none exists.
    rung_components = [component for component in components if component.use_rungs]
    if not rung_components:
        return
    plural = ('' if len(rung_components) == 1 else 's')
    masterprint(f'Determining rung population{plural} ...')
    # Assign all particles to rung 0 and update rungs_N,
    # resulting in rungs_N = [N_local, 0, 0, ...].
    for component in rung_components:
        for i in range(component.N_local):
            component.rung_indices[i] = 0
        component.set_rungs_N()
    # Do a fake short kick, which computes momentum updates and use
    # these to set the rungs, but does not apply these momentum updates.
    kick_short(components, Δt, fake=True)
    masterprint('done')

# Function which dump all types of output
@cython.header(
    # Arguments
    components=list,
    output_filenames=dict,
    dump_time=object,  # collections.namedtuple
    Δt='double',
    # Locals
    act=str,
    any_activations='bint',
    filename=str,
    time_param=str,
    time_value='double',
    returns='bint',
)
def dump(components, output_filenames, dump_time, Δt=0):
    time_param = dump_time.time_param
    time_value = {'t': dump_time.t, 'a': dump_time.a}[time_param]
    any_activations = False
    # Activate or termiante component before dumps
    for act in 𝕆[life_output_order[:life_output_order.index('dump')]]:
        if time_value in activation_termination_times[time_param]:
            any_activations |= (
                activate_terminate(components, time_value, Δt, act) and act == 'activate'
            )
    # Dump snapshot
    if time_value in snapshot_times[time_param]:
        filename = output_filenames['snapshot'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        save(components, filename)
    # Dump power spectrum
    if time_value in powerspec_times[time_param]:
        filename = output_filenames['powerspec'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        powerspec(components, filename)
    # Dump render3D
    if time_value in render3D_times[time_param]:
        filename = output_filenames['render3D'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        render3D(components, filename)
    # Dump render2D
    if time_value in render2D_times[time_param]:
        filename = output_filenames['render2D'].format(time_param, time_value)
        if time_param == 't':
            filename += unit_time
        render2D(components, filename)
    # Activate or terminate components after dumps
    for act in 𝕆[life_output_order[life_output_order.index('dump')+1:]]:
        if time_value in activation_termination_times[time_param]:
            any_activations |= (
                activate_terminate(components, time_value, Δt, act) and act == 'activate'
            )
    return any_activations

# Function for terminating an existing component
# or activating a new one.
@cython.header(
    # Arguments
    components=list,
    a='double',
    Δt='double',
    act=str,
    # Locals
    activated_components=list,
    active_components=list,
    active_components_order=list,
    component='Component',
    terminated_components=list,
    universal_a_backup='double',
    returns='bint',
)
def activate_terminate(components, a, Δt, act='activate terminate'):
    """This function mutates the passed list of components
    as well as the global passive_components, keeping their
    collective contents constant.
    The return value is a boolean signalling
    whether any component was activated.
    """
    # Terminations
    if 'terminate' in act:
        terminated_components = [
            component
            for component in components
            if component.life[1] == a
        ]
        if terminated_components:
            for component in terminated_components:
                masterprint(f'Terminating "{component.name}" ...')
                component.cleanup()
                masterprint('done')
            # Remove terminated components
            # from the list of current components.
            components[:] = [
                component
                for component in components
                if component not in terminated_components
            ]
            # Add terminated components
            # to the list of passive components.
            passive_components[:] = passive_components + terminated_components
    # Activations
    activated_components = []
    if 'activate' in act:
        activated_components = [
            component
            for component in passive_components
            if component.life[0] == a
        ]
        if activated_components:
            # For realization it is important that universals.a matches
            # the given a exactly. These really ought to be idential,
            # but may not be due to floating-point imprecisions.
            universal_a_backup = universals.a
            universals.a = a
            for component in activated_components:
                masterprint(f'Activating "{component.name}" ...')
                component.realize()
                component.realize_if_linear(0, specific_multi_index=0)        # ϱ
                component.realize_if_linear(1, specific_multi_index=0)        # J
                component.realize_if_linear(2, specific_multi_index='trace')  # 𝒫
                component.realize_if_linear(2, specific_multi_index=(0, 0))   # ς
                masterprint('done')
            universals.a = universal_a_backup
            # Remove newly activated components from the list
            # of passive components.
            passive_components[:] = [
                component
                for component in passive_components
                if component not in activated_components
            ]
            # Add newly activated components to the list of
            # current components, keeping the original ordering.
            active_components = components + activated_components
            active_components_order = [component.name for component in active_components]
            components[:] = [
                active_components[active_components_order.index(name)]
                for name in components_order
                if name in active_components_order
            ]
            # If a particle component has been activated, we need to
            # assign an initial rung population. Also, the current rung
            # populations of old particle components needs to updated
            # due to the new particle component.
            if any([
                component.representation == 'particles'
                for component in activated_components
            ]):
                initialize_rung_populations(components, Δt)
    return bool(activated_components)

# Function which dump all types of output
@cython.header(
    # Arguments
    components=list,
    time_step='Py_ssize_t',
    Δt='double',
    Δt_begin='double',
    # Locals
    autosave_params_filename=str,
    autosave_filename=str,
    remaining_output_times=dict,
    param_lines=list,
    present='double',
    time_param=str,
    returns='void',
)
def autosave(components, time_step, Δt, Δt_begin):
    masterprint('Autosaving ...')
    autosave_filename        = f'{autosave_dir}/autosave_{jobid}.hdf5'
    autosave_params_filename = f'{paths["params_dir"]}/autosave_{jobid}.params'
    # Save parameter file corresponding to the snapshot
    if master:
        masterprint(f'Writing parameter file "{autosave_params_filename}" ...')
        with disable_numpy_summarization():
            param_lines = []
            # Header
            param_lines += [
                f'# This parameter file is the result of an autosave of job {jobid},',
                f'# which uses the parameter file "{paths["params"]}".',
                f'# The autosave was carried out {datetime.datetime.now()}.',
                f'# The following is a copy of this original parameter file.',
            ]
            param_lines += ['']*2
            # Original parameter file
            param_lines += params_file_content.split('\n')
            param_lines += ['']*2
            # IC snapshot
            param_lines += [
                f'# The autosaved snapshot file was saved to',
                f'initial_conditions = "{autosave_filename}"',
            ]
            # Present time
            param_lines.append(f'# The autosave happened at time')
            if enable_Hubble:
                param_lines.append(f'a_begin = {universals.a:.16e}')
            else:
                param_lines.append(f't_begin = {universals.t:.16e}*{unit_time}')
            # Time step, current and original time step size
            param_lines += [
                f'# The time step and time step size was',
                f'initial_time_step = {time_step + 1}',
                f'{unicode("Δt_autosave")} = {Δt:.16e}*{unit_time}',
                f'# The time step size at the beginning of the simulation was',
                f'{unicode("Δt_begin_autosave")} = {Δt_begin:.16e}*{unit_time}',
            ]
            # All output times
            param_lines += [
                f'# All output times',
                f'output_times_full = {output_times}',
            ]
            # Remaining output times
            remaining_output_times = {'a': {}, 't': {}}
            for time_param, present in zip(('a', 't'), (universals.a, universals.t)):
                for output_kind, output_time in output_times[time_param].items():
                    remaining_output_times[time_param][output_kind] = [
                        ot for ot in output_time if ot >= present
                    ]
            param_lines += [
                f'# Remaining output times',
                f'output_times = {remaining_output_times}',
            ]
        # Write to parameter file
        with open(autosave_params_filename, 'w', encoding='utf-8') as autosave_params_file:
            print('\n'.join(param_lines), file=autosave_params_file)
        masterprint('done')
    # Save standard snapshot. Include all components regardless
    # of the snapshot_select user parameter.
    save(components, autosave_filename, snapshot_type='standard', save_all_components=True)
    # If this simulation run was started from an autosave snapshot
    # with a different name from the one just saved, remove this
    # now superfluous autosave snapshot.
    if master:
        if (    isinstance(initial_conditions, str)
            and re.search(r'^autosave_\d+\.hdf5$', os.path.basename(initial_conditions))
            and os.path.abspath(initial_conditions) != os.path.abspath(autosave_filename)
            and os.path.isfile(initial_conditions)
        ):
            os.remove(initial_conditions)
    masterprint('done')

# Function which prints out basic information
# about the current time step.
@cython.header(
    # Arguments
    time_step='Py_ssize_t',
    Δt='double',
    bottleneck=str,
    components=list,
    end='bint',
    # Locals
    component='Component',
    components_with_rungs=list,
    components_with_w=list,
    header_lines=list,
    i='Py_ssize_t',
    last_populated_rung='signed char',
    line=list,
    lines=list,
    part=str,
    parts=list,
    rung_index='signed char',
    rung_N='Py_ssize_t',
    width='Py_ssize_t',
    width_max='Py_ssize_t',
    returns='void',
)
def print_timestep_heading(time_step, Δt, bottleneck, components, end=False):
    # This function builds up its output as strings in the parts list
    parts = ['\nEnd of main time loop' if end else terminal.bold(f'\nTime step {time_step}')]
    # Create the header lines (current scale factor, time and time
    # step), ensuring proper alignment.
    header_lines = []
    if enable_Hubble:
        header_lines.append(
            [
                '\nScale factor',
                significant_figures(universals.a, 4, fmt='unicode'),
                '',
            ]
        )
    header_lines.append(
        [
            '\nCosmic time' if enable_Hubble else '\nTime',
            significant_figures(universals.t, 4, fmt='unicode'),
            unit_time,
        ]
    )
    if not end:
        header_lines.append(
            [
                '\nStep size',
                significant_figures(Δt, 4, fmt='unicode'),
                unit_time + (f' (limited by {bottleneck})' if bottleneck else ''),
            ]
        )
    header_maxlength0 = np.max([len(line[0]) for line in header_lines])
    header_maxdot1 = np.max([line[1].index('.') for line in header_lines])
    for line in header_lines:
        line[0] += ':' + ' '*(header_maxlength0 - len(line[0]) + 1)
        line[1] = ' '*(header_maxdot1 - line[1].index('.')) + line[1]
    header_maxlength1 = np.max([len(line[1]) for line in header_lines])
    for line in header_lines:
        if line[2]:
            line[2] = ' '*(header_maxlength1 - len(line[1]) + 1) + line[2]
    parts += [''.join(line) for line in header_lines]
    # Equation of state of each component
    components_with_w = [
        component for component in components if (
            component.w_type != 'constant'
            and 'metric' not in component.class_species
            and 'lapse'  not in component.class_species
        )
    ]
    if components_with_w:
        parts.append(f'\nEoS w:\n')
        lines = []
        for component in components_with_w:
            lines.append(
                f'{component.name}: $' + significant_figures(component.w(), 4, fmt='unicode')
            )
        parts.append('\n'.join(align_text(lines, indent=4)))
    # Rung population for each component
    components_with_rungs = [
        component for component in components if component.use_rungs
    ]
    if components_with_rungs:
        parts.append(f'\nRung population:\n')
        lines = []
        for component in components_with_rungs:
            rung_population = []
            last_populated_rung = 0
            for rung_index in range(N_rungs):
                rung_N = allreduce(component.rungs_N[rung_index], op=MPI.SUM)
                rung_population.append(str(rung_N))
                if rung_N > 0:
                    last_populated_rung = rung_index
            lines.append(
                f'{component.name}: $' + ', $'.join(rung_population[:last_populated_rung+1])
            )
        parts.append('\n'.join(align_text(lines, indent=4)))
    # Print out the combined heading
    masterprint(''.join(parts))

# Function which prints out debugging information at the end of each
# time step, if such output is requested.
@cython.header(
    # Arguments
    components=list,
    # Locals
    component='Component',
    decimals='Py_ssize_t',
    direct_summation_time='double',
    direct_summation_time_mean='double',
    direct_summation_time_total='double',
    imbalance='double',
    imbalance_str=str,
    line=str,
    lines=list,
    message=list,
    other_rank='int',
    rank_max_load='int',
    sign=str,
    tiling='Tiling',
    value_bad='double',
    value_miserable='double',
    returns='void',
)
def print_timestep_footer(components):
    # Print out the load imbalance, measured purely over
    # direct summation interactions and stored on the Tiling's.
    if 𝔹[print_load_imbalance and nprocs > 1]:
        # Decimals to show (of percentage)
        decimals = 1
        # Values at which to change color
        value_bad       = 0.3
        value_miserable = 1.0
        # Tally up computation times
        direct_summation_time = 0
        for component in components:
            for tiling in component.tilings.values():
                direct_summation_time += tiling.computation_time_total
        if allreduce(direct_summation_time > 0, op=MPI.LOR):
            Gather(asarray([direct_summation_time]), direct_summation_times)
            if master:
                direct_summation_time_total = sum(direct_summation_times)
                direct_summation_time_mean = direct_summation_time_total/nprocs
                for other_rank in range(nprocs):
                    imbalances[other_rank] = (
                        direct_summation_times[other_rank]/direct_summation_time_mean - 1
                    )
                rank_max_load = np.argmax(imbalances)
                if 𝔹[print_load_imbalance == 'full']:
                    # We want to print out the load imbalance
                    # for each process individually.
                    masterprint('Load imbalance:')
                    lines = []
                    for other_rank in range(nprocs):
                        imbalance = imbalances[other_rank]
                        sign = '+' if imbalance >= 0 else '-'
                        lines.append(
                            f'Process ${other_rank}: ${sign}${{:.{decimals}f}}%'
                            .format(abs(100*imbalance))
                        )
                    lines = align_text(lines, indent=4)
                    for other_rank, line in enumerate(lines):
                        if other_rank != rank_max_load:
                            continue
                        imbalance = imbalances[other_rank]
                        first, last = line.split(':')
                        if imbalance >= value_miserable:
                            last = terminal.bold_red(last)
                        elif imbalance >= value_bad:
                            last = terminal.bold_yellow(last)
                        else:
                            last = terminal.bold(last)
                        lines[other_rank] = f'{first}:{last}'
                    masterprint('\n'.join(lines))
                else:
                    # We want to print out only the
                    # worst case load imbalance.
                    imbalance = imbalances[rank_max_load]
                    imbalance_str = f'{{:.{decimals}f}}%'.format(100*imbalance)
                    if imbalance >= value_miserable:
                        imbalance_str = terminal.bold_red(imbalance_str)
                    elif imbalance >= value_bad:
                        imbalance_str = terminal.bold_yellow(imbalance_str)
                    masterprint(f'Load imbalance: {imbalance_str} (process {rank_max_load})')
    elif ...:
        ...
# Arrays used by the print_timestep_footer() function
cython.declare(direct_summation_times='double[::1]', imbalances='double[::1]')
direct_summation_times = empty(nprocs, dtype=C2np['double']) if master else None
imbalances = empty(nprocs, dtype=C2np['double']) if master else None

# Function which checks the sanity of the user supplied output times,
# creates output directories and defines the output filename patterns.
@cython.header()
def prepare_for_output(components=None):
    """As this function uses universals.t and universals.a as the
    initial values of the cosmic time and the scale factor, you must
    initialize these properly before calling this function.
    """
    output_times_all = output_times.copy()
    # If a list of components is passed, we need to first run this
    # Function without these components, processing the output times.
    # Then we can insert 'life' times based on the 'life'
    # attributes of the components, together with the current and the
    # final output time.
    if components:
        dump_times, output_filenames = prepare_for_output()
        a_final = dump_times[len(dump_times) - 1].a
        if a_final is None:
            a_final = ထ
        output_times_all['t']['life'] = ()
        output_times_all['a']['life'] = tuple(sorted({
            a
            for component in components
            for a in component.life
            if universals.a < a < a_final
        }))
        for time_param in ('a', 't'):
            activation_termination_times[time_param] = output_times[time_param]['life']
    # Check that the output times are legal
    for time_param, at_begin in zip(('a', 't'), (universals.a, universals.t)):
        for output_kind, output_time in output_times[time_param].items():
            if output_time and np.min(output_time) < at_begin:
                message = [
                    f'Cannot produce a {output_kind} at {time_param} '
                    f'= {np.min(output_time):.6g}'
                ]
                if time_param == 't':
                    message.append(f' {unit_time}')
                message.append(f', as the simulation starts at {time_param} = {at_begin:.6g}')
                if time_param == 't':
                    message.append(f' {unit_time}')
                message.append('.')
                abort(''.join(message))
    # Create output directories if necessary
    if master:
        for time_param in ('a', 't'):
            for output_kind, output_time in output_times[time_param].items():
                # Do not create directory if this kind of output
                # should never be dumped to the disk.
                if not output_time or not output_kind in output_dirs:
                    continue
                # Create directory
                output_dir = output_dirs[output_kind]
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
    Barrier()
    # Construct the patterns for the output file names. This involves
    # determining the number of digits of the scalefactor in the output
    # filenames. There should be enough digits so that adjacent dumps do
    # not overwrite each other, and so that the name of the first dump
    # differs from that of the IC, should it use the same
    # naming convention.
    output_filenames = {}
    for time_param, at_begin in zip(('a', 't'), (universals.a, universals.t)):
        # Here the output_times_full dict is used rather than just the
        # output_times dict. These dicts are equal, except after
        # starting from an autosave, where output_times will contain
        # the remaining dump times only, whereas output_times_full
        # will contain all the original dump times.
        # We use output_times_full so as to stick to the original naming
        # format used before restarting from the autosave.
        for output_kind, output_time in output_times_full[time_param].items():
            # This kind of output does not matter if
            # it should never be dumped to the disk.
            if not output_time or not output_kind in output_dirs:
                continue
            # Compute number of digits
            times = sorted(set((at_begin, ) + output_time))
            ndigits = 0
            while True:
                fmt = f'{{:.{ndigits}f}}'
                if (len(set([fmt.format(ot) for ot in times])) == len(times)
                    and (fmt.format(times[0]) != fmt.format(0) or not times[0])):
                    break
                ndigits += 1
            fmt = f'{{}}={fmt}'
            # Use the format (that is, either the format from the a
            # output times or the t output times) with the largest
            # number of digits.
            if output_kind in output_filenames:
                if int(re.search(
                    '[0-9]+',
                    re.search('{.+?}', output_filenames[output_kind]).group(),
                ).group()) >= ndigits:
                    continue
            # Store output name patterns
            output_dir = output_dirs[output_kind]
            output_base = output_bases[output_kind]
            sep = '_' if output_base else ''
            output_filenames[output_kind] = f'{output_dir}/{output_base}{sep}{fmt}'
    # Lists of sorted dump times of both kinds
    a_dumps = sorted(set([nr for val in output_times_all['a'].values() for nr in val]))
    t_dumps = sorted(set([nr for val in output_times_all['t'].values() for nr in val]))
    # Combine a_dumps and t_dumps into a single list of named tuples
    dump_times =  [DumpTime('t', t=t_dump, a=None) for t_dump in t_dumps]
    dump_times += [DumpTime('a', a=a_dump, t=None) for a_dump in a_dumps]
    if enable_Hubble:
        a_lower = t_lower = machine_ϵ
        for i, dump_time in enumerate(dump_times):
            if dump_time.time_param == 't' and dump_time.a is None:
                a = scale_factor(dump_time.t)
                dump_time = DumpTime('t', t=dump_time.t, a=a)
            elif dump_time.time_param == 'a' and dump_time.t is None:
                t = cosmic_time(dump_time.a, a_lower, t_lower)
                dump_time = DumpTime('a', a=dump_time.a, t=t)
                a_lower, t_lower = dump_time.a, dump_time.t
            dump_times[i] = dump_time
    # Sort the list according to the cosmic time
    dump_times = sorted(dump_times, key=operator.attrgetter('t'))
    # Two dump times at the same or very near the same time
    # should count as one.
    if len(dump_times) > 1:
        dump_time = dump_times[0]
        dump_times_unique = [dump_time]
        t_previous = dump_time.t
        for dump_time in dump_times[1:]:
            if not np.isclose(dump_time.t, t_previous, rtol=1e-6, atol=0):
                dump_times_unique.append(dump_time)
                t_previous = dump_time.t
        dump_times = dump_times_unique
    return dump_times, output_filenames
# Container used by the prepare_for_output() and timeloop() function
DumpTime = collections.namedtuple(
    'DumpTime', ('time_param', 't', 'a'),
)


# Here we set various values used for the time integration. These are
# purely numerical in character. For factors used to control the time
# step size Δt based on various physical time scales, see the fac_*
# variables further down.
cython.declare(
    Δt_initial_fac='double',
    Δt_reduce_fac='double',
    Δt_increase_fac='double',
    Δt_period_increase_max_fac='double',
    Δt_period_increase_min_fac='double',
    Δt_ratio_warn='double',
    Δt_ratio_abort='double',
    Δt_jump_fac='double',
    Δt_reltol='double',
    Δt_period='Py_ssize_t',
)
# The initial time step size Δt will be set to the maximum allowed
# value times this factor. The same goes for Δt right after activation
# of a component. As newly added components may need a somewhat lower
# time step than predicted, this factor should be below unity.
Δt_initial_fac = 0.95
# When reducing Δt, set it to the maximum allowed value
# times this factor.
Δt_reduce_fac = 0.94
# When increasing Δt, set it to the maximum allowed value
# times this factor.
Δt_increase_fac = 0.96
# The maximum allowed fractional increase in Δt
# after Δt_period time steps with constant time step size.
Δt_period_increase_max_fac = 0.33
# The minimum fractional increase in Δt needed before it is deemed
# worth it to synchronize drifts/kicks and update Δt.
Δt_period_increase_min_fac = 0.01
# Ratios between old and new Δt, below which the program
# will show a warning or abort, respectively.
Δt_ratio_warn  = 0.7
Δt_ratio_abort = 0.01
# When using adaptive time stepping (N_rungs > 1), the particles may
# jump from their current rung to the rung just above or below,
# depending on their (short-range) acceleration and the time step
# size Δt. To ensure that particles with accelerations right at the
# border between two rungs does not jump between these rungs too
# often (which would degrade the symplecticity), we introduce
# Δt_jump_fac so that in order to jump up (get assigned a smaller
# individual time step size), a particle has to belong to the rung
# above even if the time step size had been Δt*Δt_jump_fac < Δt.
# Likewise, to jump down (get assigned a larger individual time step
# size), a particle has to belong to the rung below even if the time
# step size had been Δt/Δt_jump_fac > Δt. The factor Δt_jump_fac
# should then be somewhat below unity.
Δt_jump_fac = 0.95
# Due to floating-point imprecisions, universals.t may not land
# exactly at sync_time when it should, which is needed to detect
# whether we are at a synchronization time or not. To fix this,
# we consider the universal time to be at the synchronization time
# if they differ by less than Δt_reltol times the
# base time step size Δt.
Δt_reltol = 1e-9
# The number of time steps before the base time step size Δt is
# allowed to increase. Choosing a multiple of 8 prevents the
# formation of spurious anisotropies when evolving fluids with the
# MacCormack method, as each of the 8 flux directions are then
# used with the same time step size (in the simple case of no
# reduction to Δt and no synchronizations due to dumps).
Δt_period = 1*8

# Here we set the values for the various factors used when determining
# the time step size. The values given below has been tuned by hand as
# to achieve a matter power spectrum at a = 1 that has converged to
# within ~1% on all relevant scales, for
# Δt_base_background_factor = Δt_base_nonlinear_factor = Δt_rung_factor = 1.
# For further specification of each factor,
# consult the get_base_timestep_size() function.
cython.declare(
    fac_dynamical='double',
    fac_hubble='double',
    fac_ẇ='double',
    fac_Γ='double',
    fac_courant='double',
    fac_pm='double',
    fac_p3m='double',
    fac_softening='double',
)
# The base time step should be below the dynamic time scale
# times this factor.
fac_dynamical = 0.057*Δt_base_background_factor
# The base time step should be below the current Hubble time scale
# times this factor.
fac_hubble = 0.16*Δt_base_background_factor
# The base time step should be below |ẇ|⁻¹ times this factor,
# for all components. Here w is the equation of state parameter.
fac_ẇ = 0.0017*Δt_base_background_factor
# The base time step should be below |Γ|⁻¹ times this factor,
# for all components. Here Γ is the decay rate.
fac_Γ = 0.0028*Δt_base_background_factor
# The base time step should be below that set by the 1D Courant
# condition times this factor, for all fluid components.
fac_courant = 0.21*Δt_base_nonlinear_factor
# The base time step should be small enough so that particles
# participating in interactions using the PM method do not drift further
# than the size of one PM grid cell times this factor in a single
# time step. The same condition is applied to fluids, where the bulk
# velocity is what counts (i.e. we ignore the sound speed).
fac_pm = 0.13*Δt_base_nonlinear_factor
# The base time step should be small enough so that particles
# participating in interactions using the P³M method do not drift
# further than the long/short-range force split scale times this factor
# in a single time step.
fac_p3m = 0.14*Δt_base_nonlinear_factor
# When using adaptive time stepping (N_rungs > 1), the individual time
# step size for a given particle must not be so large that it drifts
# further than its softening length times this factor, due to its
# (short-range) acceleration (i.e. its current velocity is not
# considered). If it does become large enough for this, the particle
# jumps to the rung just above its current rung.
# In GADGET2, this same factor is called ErrTolIntAccuracy (or η)
# and has a value of 0.025.
fac_softening = 0.025*Δt_rung_factor

# If this module is run properly (detected by jobid being set),
# launch the CO𝘕CEPT run.
if jobid != -1:
    if 'special' in special_params:
        # Instead of running a simulation, run some utility
        # as defined by the special_params dict.
        delegate()
    else:
        # Run the time loop
        timeloop()
        # Simulation done
        universals.any_warnings = allreduce(universals.any_warnings, op=MPI.LOR)
        if universals.any_warnings:
            masterprint(f'{esc_concept} run {jobid} finished')
        else:
            masterprint(
                f'{esc_concept} run {jobid} finished successfully',
                fun=terminal.bold_green,
            )
    # Shutdown CO𝘕CEPT properly
    abort(exit_code=0)
