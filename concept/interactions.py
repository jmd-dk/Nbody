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
cimport(
    'from communication import     '
    '    communicate_ghosts,       '
    '    domain_subdivisions,      '
    '    get_buffer,               '
    '    rank_neighbouring_domain, '
    '    sendrecv_component,       '
)
cimport('from ewald import get_ewald_grid')
cimport(
    'from mesh import                         '
    '    diff_domaingrid,                     '
    '    domain_decompose,                    '
    '    fft,                                 '
    '    get_fftw_slab,                       '
    '    get_gridshape_local,                 '
    '    interpolate_components,              '
    '    interpolate_domaingrid_to_particles, '
    '    interpolate_grid_to_grid,            '
    '    interpolate_upstream,                '
    '    nullify_modes,                       '
    '    slab_decompose,                      '
    '    slab_fourier_loop,                   '
)
cimport(
    'from species import                        '
    '    accept_or_reject_subtiling_refinement, '
    '    tentatively_refine_subtiling,          '
)

# Function pointer types used in this module
pxd("""
ctypedef void (*func_interaction)(
    str,              # interaction_name
    Component,        # receiver
    Component,        # supplier
    dict,             # ᔑdt_rungs
    int,              # rank_supplier
    bint,             # only_supply
    str,              # pairing_level
    Py_ssize_t[::1],  # tile_indices_receiver
    Py_ssize_t**,     # tile_indices_supplier_paired
    Py_ssize_t*,      # tile_indices_supplier_paired_N
    dict,             # interaction_extra_args
)
""")



# Generic function implementing component-component pairing
@cython.header(
    # Arguments
    interaction_name=str,
    receivers=list,
    suppliers=list,
    interaction=func_interaction,
    ᔑdt_rungs=dict,
    pairing_level=str,
    interaction_extra_args=dict,
    # Locals
    affected=list,
    anticipate_refinement='bint',
    anticipation_period='Py_ssize_t',
    attempt_refinement='bint',
    computation_time='double',
    dependent=list,
    deterministic='bint',
    index='Py_ssize_t',
    judge_refinement='bint',
    judgement_period='Py_ssize_t',
    lowest_active_rung='signed char',
    only_supply='bint',
    other_rank='int',
    pair=set,
    pairs=list,
    receiver='Component',
    refinement_offset='Py_ssize_t',
    refinement_period='Py_ssize_t',
    rung_index='signed char',
    subtiles_computation_times_N_interaction='Py_ssize_t[::1]',
    subtiles_computation_times_interaction='double[::1]',
    subtiles_computation_times_sq_interaction='double[::1]',
    subtiling='Tiling',
    subtiling_shape_judged='Py_ssize_t[::1]',
    subtiling_name=str,
    subtiling_name_2=str,
    supplier='Component',
    tile_sorted=set,
    tiling_name=str,
    returns='void',
)
def component_component(
    interaction_name, receivers, suppliers, interaction, ᔑdt_rungs,
    pairing_level, interaction_extra_args={},
):
    """This function takes care of pairings between all receiver and
    supplier components. It then calls doman_domain.
    """
    # Lookup basic information for this interaction
    interaction_info = interactions_registered[interaction_name]
    dependent     = interaction_info.dependent
    affected      = interaction_info.affected
    deterministic = interaction_info.deterministic
    # The names used to refer to the domain and tile level tiling
    # (tiles and subtiles). In the case of pairing_level == 'domain',
    # no actual tiling will take place, but we still need the
    # tile + subtile structure. For this, the trivial tiling,
    # spanning the box, is used.
    if 𝔹[pairing_level == 'tile']:
        tiling_name      = f'{interaction_name} (tiles)'
        subtiling_name   = f'{interaction_name} (subtiles)'
        subtiling_name_2 = f'{interaction_name} (subtiles 2)'
    else:  # pairing_level == 'domain':
        tiling_name = subtiling_name = 'trivial'
    # Set flags anticipate_refinement, attempt_refinement and
    # judge_refinement. The first signals whether a tentative subtiling
    # refinement attempt is comming up soon, in which case we should
    # be collecting computation time data of the current subtiling.
    # The second specifies whether a tentative refinement of the
    # subtilings in use should be performed now, meaning prior to
    # the interaction. The third specifies whether the previously
    # performed tentative refinement should be concluded, resulting in
    # either accepting or rejecting the refinement.
    anticipate_refinement = attempt_refinement = judge_refinement = False
    if 𝔹[pairing_level == 'tile'
        and shortrange_params[interaction_name]['subtiling'][0] == 'automatic'
    ]:
        # The anticipation_period and judgement_period specifies the
        # number of time steps spend collecting computation time data
        # before and after a tentative sutiling refinement.
        # The refinement will be judged after the first interaction of
        # the time step after judgement_period time steps has gone by
        # after the tentative refinement (there may be many more
        # interactions in this time step, depending on N_rungs).
        # Note that changes to anticipation_period or judgement_period
        # need to be reflected in the subtiling_refinement_period_min
        # variable, defined in the commons module. The relation is
        # subtiling_refinement_period_min = (
        #     anticipation_period + judgement_period + 1)
        anticipation_period = 4
        judgement_period = 2
        subtiles_computation_times_interaction    = subtiles_computation_times   [interaction_name]
        subtiles_computation_times_sq_interaction = subtiles_computation_times_sq[interaction_name]
        subtiles_computation_times_N_interaction  = subtiles_computation_times_N [interaction_name]
        for receiver in receivers:
            subtiling = receiver.tilings.get(subtiling_name)
            if subtiling is None:
                continue
            refinement_period = subtiling.refinement_period
            refinement_offset = subtiling.refinement_offset
            if refinement_period == 0:
                abort(
                    f'The subtiling "{subtiling_name}" is set to use automatic subtiling '
                    f'refinement, but it has a refinement period of {refinement_period}.'
                )
            # We judge the attempted refinement after 2 whole time steps
            # has gone by; this one and the next. The refinement will
            # then be judged after the first interaction on the third
            # time step (there may be many more interactions
            # if N_rungs > 1).
            if interaction_name in subtilings_under_tentative_refinement:
                anticipate_refinement = True
                judge_refinement = (
                    ℤ[universals.time_step + refinement_offset + 1] % refinement_period == 0
                )
            else:
                attempt_refinement = (
                    (ℤ[universals.time_step + refinement_offset + 1] + judgement_period
                        ) % refinement_period == 0
                )
                # We begin storing the computation time data of the
                # current subtiling 4 time steps before we tentatively
                # apply the new subtiling.
                anticipate_refinement = (
                    (ℤ[universals.time_step + refinement_offset + 1] + judgement_period
                        ) % refinement_period >= refinement_period - anticipation_period
                )
            break
    # Do the tentative subtiling refinement, if required
    if attempt_refinement:
        # Copy the old computation times to new locations in
        # subtiles_computation_times_interaction, making room for the
        # new computation times.
        for rung_index in range(N_rungs):
            subtiles_computation_times_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_interaction[rung_index]
            )
            subtiles_computation_times_sq_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_sq_interaction[rung_index]
            )
            subtiles_computation_times_N_interaction[N_rungs + rung_index] = (
                subtiles_computation_times_N_interaction[rung_index]
            )
            subtiles_computation_times_interaction   [rung_index] = 0
            subtiles_computation_times_sq_interaction[rung_index] = 0
            subtiles_computation_times_N_interaction [rung_index] = 0
        # Replace the subtilings with slighly refined versions
        subtilings_under_tentative_refinement.add(interaction_name)
        tentatively_refine_subtiling(interaction_name)
    # Pair each receiver with all suppliers and let them interact
    pairs = []
    tile_sorted = set()
    computation_time = 0  # Total tile-tile computation time for this call to component_component()
    for receiver in receivers:
        for supplier in suppliers:
            pair = {receiver, supplier}
            if pair in pairs:
                continue
            pairs.append(pair)
            # Make sure that the tile sorting of particles
            # in the two components are up-to-date.
            with unswitch(1):
                if receiver not in tile_sorted:
                    receiver.tile_sort(tiling_name)
                    tile_sorted.add(receiver)
                    # Also ensure existence of subtiling
                    receiver.init_tiling(subtiling_name)
            if supplier not in tile_sorted:
                supplier.tile_sort(tiling_name)
                tile_sorted.add(supplier)
                # Also ensure existence of subtiling
                supplier.init_tiling(subtiling_name)
            # Flag specifying whether the supplier should only supply
            # forces to the receiver and not receive any force itself.
            only_supply = (supplier not in receivers)
            # Pair up domains for the current
            # receiver and supplier component.
            domain_domain(
                interaction_name,
                receiver,
                supplier,
                interaction,
                ᔑdt_rungs,
                dependent,
                affected,
                only_supply,
                deterministic,
                pairing_level,
                interaction_extra_args,
            )
        # The interactions between the receiver and all suppliers are
        # now done. Add the accumulated computation time to the local
        # computation_time variable, then nullify the computation time
        # stored on the subtiling, so that it is ready for new data.
        # To keep the total computation time tallied up over the entire
        # time step present on the subtiling, add the currently stored
        # computation time to the computation_time_total attribute
        # before doing the nullification.
        subtiling = receiver.tilings[subtiling_name]
        computation_time += subtiling.computation_time
        subtiling.computation_time_total += subtiling.computation_time
        subtiling.computation_time = 0
    # All interactions are now done. If the measured computation time
    # should be used for automatic subtiling refinement, store this
    # outside of this function.
    if 𝔹[pairing_level == 'tile'
        and shortrange_params[interaction_name]['subtiling'][0] == 'automatic'
    ]:
        # The computation time depends drastically on which rungs are
        # currently active. We therefore store the total computation
        # time according to the current lowest active rung.
        if anticipate_refinement or attempt_refinement or judge_refinement:
            lowest_active_rung = ℤ[N_rungs - 1]
            for receiver in receivers:
                if receiver.lowest_active_rung < lowest_active_rung:
                    lowest_active_rung = receiver.lowest_active_rung
                    if lowest_active_rung == 0:
                        break
            subtiles_computation_times_interaction   [lowest_active_rung] += computation_time
            subtiles_computation_times_sq_interaction[lowest_active_rung] += computation_time**2
            subtiles_computation_times_N_interaction [lowest_active_rung] += 1
        # If it is time to judge a previously attempted refinement,
        # do so and reset the computation time.
        if judge_refinement:
            subtilings_under_tentative_refinement.remove(interaction_name)
            subtiling_shape_judged = accept_or_reject_subtiling_refinement(
                interaction_name,
                subtiles_computation_times_interaction,
                subtiles_computation_times_sq_interaction,
                subtiles_computation_times_N_interaction,
            )
            subtiles_computation_times_interaction   [:] = 0
            subtiles_computation_times_sq_interaction[:] = 0
            subtiles_computation_times_N_interaction [:] = 0
        else:
            subtiling_shape_judged = subtiling_shape_rejected
        # Gather information about the acceptance of the new subtiling
        # and print out any positive results.
        Gather(subtiling_shape_judged, subtiling_shapes_judged)
        if master:
            for other_rank in range(nprocs):
                index = 3*other_rank
                if subtiling_shapes_judged[index] == 0:
                    continue
                subtiling_shape_judged = subtiling_shapes_judged[index:index+3]
                masterprint(
                    f'Rank {other_rank}: Refined subtile decomposition ({interaction_name}):',
                    '×'.join(list(map(str, subtiling_shape_judged)))
                )
# Containers and array used by the component_component() function.
# The subtiles_computation_times and subtiles_computation_times_N are
# used to store total computation times and numbers for performed
# interations. They are indexed as
# subtiles_computation_times[interaction_name][rung_index],
# resulting in the accumulated computation time for this interaction
# when the lowest active rung corresponds to rung_index.
# The subtilings_under_tentative_refinement set contain names of
# interactions the subtilings of which are currently under
# tentative refinement.
# The subtiling_shape_rejected are used only for signalling purposes,
# while the subtiling_shapes_judged are used to gather subtiling shapes
# from all processes into the master process.
cython.declare(
    subtiles_computation_times=object,
    subtiles_computation_times_sq=object,
    subtiles_computation_times_N=object,
    subtilings_under_tentative_refinement=set,
    subtiling_shape_rejected='Py_ssize_t[::1]',
    subtiling_shapes_judged='Py_ssize_t[::1]',
)
subtiles_computation_times = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['double'])
)
subtiles_computation_times_sq = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['double'])
)
subtiles_computation_times_N = collections.defaultdict(
    lambda: zeros(ℤ[2*N_rungs], dtype=C2np['Py_ssize_t'])
)
subtilings_under_tentative_refinement = set()
subtiling_shape_rejected = zeros(3, dtype=C2np['Py_ssize_t'])
subtiling_shapes_judged = empty(3*nprocs, dtype=C2np['Py_ssize_t']) if master else None

# Generic function implementing domain-domain pairing
@cython.header(
    # Arguments
    interaction_name=str,
    receiver='Component',
    supplier='Component',
    interaction=func_interaction,
    ᔑdt_rungs=dict,
    dependent=list,
    affected=list,
    only_supply='bint',
    deterministic='bint',
    pairing_level=str,
    interaction_extra_args=dict,
    # Locals
    domain_pair_nr='Py_ssize_t',
    instantaneous='bint',
    interact='bint',
    only_supply_communication='bint',
    only_supply_passed='bint',
    rank_recv='int',
    rank_send='int',
    ranks_recv='int[::1]',
    ranks_send='int[::1]',
    supplier_extrl='Component',
    supplier_local='Component',
    tile_indices='Py_ssize_t[:, ::1]',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    tile_indices_supplier_paired='Py_ssize_t**',
    tile_indices_supplier_paired_N='Py_ssize_t*',
    tile_pairings_index='Py_ssize_t',
    returns='void',
)
def domain_domain(
    interaction_name, receiver, supplier, interaction, ᔑdt_rungs, dependent, affected,
    only_supply, deterministic, pairing_level, interaction_extra_args,
):
    """This function takes care of pairings between the domains
    containing particles/fluid elements of the passed receiver and
    supplier component.
    As the components are distributed at the domain level,
    all communication needed for the interaction will be taken care of
    by this function. The receiver will not be communicated, while the
    supplier will be sent to other processes (domains) and also received
    back from other processes. Thus both local and external versions of
    the supplier exist, called supplier_local and supplier_extrl.
    The dependent and affected arguments specify which attributes of the
    supplier and receiver component are needed to supply and receive
    the force, respectively. Only these attributes will be communicated.
    If affected is an empty list, this is not really an interaction.
    In this case, every domain will both send and receive from every
    other domain.
    """
    # To satisfy the compiler
    tile_indices_receiver = tile_indices_supplier = None
    tile_indices_supplier_paired = tile_indices_supplier_paired_N = NULL
    # Flag specifying whether or not this interaction is instantaneous.
    # For instantaneous interactions, we need to apply the updates to
    # the affected variables after each domain-domain pairing.
    instantaneous = interactions_registered[interaction_name].instantaneous
    # Get the process ranks to send to and receive from.
    # When only_supply is True, each domain will be paired with every
    # other domain, either in the entire box (pairing_level == 'domain')
    # or just among the neighbouring domains (pairing_level == 'tile').
    # When only_supply is False, the results of an interaction
    # computed on one process will be send back to the other
    # participating process and applied, cutting the number of domain
    # pairs roughly in half. Note however that even if only_supply is
    # False, we may not cut the number of domain pairs in half if the
    # receiver and supplier are separate components; all domains of
    # the receiver then need to be paired with all domains of the
    # supplier. That is, "only_supply" really serve two distinct usages:
    # (1) it is passed to the interaction() function so that
    # it knows whether to also update the supplier, (2) it determines
    # the interprocess communication pattern. As this latter usage also
    # depends upon whether the receiver and supplier is really the same
    # component, we extract usage (2) into its own flag,
    # "only_supply_communication".
    only_supply_communication = (only_supply if 𝔹[receiver is supplier] else True)
    ranks_send, ranks_recv = domain_domain_communication(pairing_level, only_supply_communication)
    # Backup of the passed only_supply boolean
    only_supply_passed = only_supply
    # Pair this process/domain with whichever other
    # processes/domains are needed. This process is paired
    # with two other processes simultaneously. This process/rank sends
    # a copy of the local supplier (from now on referred to
    # as supplier_local) to rank_send, while receiving the external
    # supplier (supplier_extrl) from rank_recv.
    # On each process, the local receiver and the external
    # (received) supplier_extrl then interact.
    supplier_local = supplier
    for domain_pair_nr in range(ranks_send.shape[0]):
        # Process ranks to send to and receive from
        rank_send = ranks_send[domain_pair_nr]
        rank_recv = ranks_recv[domain_pair_nr]
        # The passed interaction function should always update the
        # particles of the receiver component within the local domain,
        # due to the particles of the external supplier component,
        # within whatever domain they happen to be in.
        # Unless the supplier component is truly only a supplier and
        # not also a receiver (only_supply is True), the particles
        # that supply the force also need to be updated by the passed
        # interaction function. It is important that the passed
        # interaction function do not update the affected variables
        # directly (e.g. mom for gravity), but instead update the
        # corresponding buffers (e.g. Δmom for gravity). The exception
        # is when the interaction is instantaneous, in which case the
        # affeceted variables should be updated directly, while also
        # updating the corresponding buffer for the supplier. The
        # buffers are what will be communicated. Also, Δmom is used to
        # figure out which short-range rung any given particle belongs
        # to. Special cases described below may change whether or not
        # the interaction between this particular domain pair should be
        # carried out on the local process (specified by the interact
        # flag), or whether the only_supply flag should be changed.
        interact = True
        only_supply = only_supply_passed
        with unswitch:
            if 𝔹[receiver is supplier] and 𝔹[pairing_level == 'domain']:
                if rank_send == rank_recv != rank:
                    # We are dealing with the special case where the
                    # local process and some other (with a rank given by
                    # rank_send == rank_recv) both send all of their
                    # particles belonging to the same component to each
                    # other, after which the exact same interaction
                    # takes place on both processes. In such a case,
                    # even when only_supply is False, there is no need
                    # to communicate the interaction results, as these
                    # are already known to both processes. Thus, we
                    # always use only_supply = True in such cases.
                    # Note that this is not true for
                    # pairing_level == 'tile', as here not all of the
                    # particles within the domains are communicated, but
                    # rather particles within completely disjoint sets
                    # of tiles, and so the interactions taking place on
                    # the two processes will not be identical.
                    only_supply = True
                    # In the case of a non-deterministic interaction,
                    # the above logic no longer holds, as the two
                    # versions of the supposedly same interaction
                    # computed on different processes will not be
                    # identical. In such cases, perform the interaction
                    # only on one of the two processes. The process with
                    # the lower rank is chosen for the job.
                    with unswitch:
                        if not deterministic:
                            interact = (rank < rank_send)
                            only_supply = only_supply_passed
        # Communicate the dependent variables (e.g. pos for gravity) of
        # the supplier. For pairing_level == 'domain', communicate all
        # local particles. For pairing_level == 'tile', we only need to
        # communicate particles within the tiles that are going to
        # interact during the current domain-domain pairing.
        with unswitch:
            if 𝔹[pairing_level == 'tile']:
                # Find interacting tiles
                tile_indices = domain_domain_tile_indices(
                    interaction_name, receiver,
                    only_supply_communication, domain_pair_nr,
                )
                tile_indices_receiver = tile_indices[0, :]
                tile_indices_supplier = tile_indices[1, :]
            else:  # pairing_level == 'domain'
                # For domain level pairing we make use of
                # the trivial tiling, containing a single tile.
                tile_indices_receiver = tile_indices_supplier = tile_indices_trivial
                tile_indices_supplier_paired = tile_indices_trivial_paired
                tile_indices_supplier_paired_N = tile_indices_trivial_paired_N
        supplier_extrl = sendrecv_component(
            supplier_local, dependent, pairing_level, interaction_name, tile_indices_supplier,
            dest=rank_send, source=rank_recv,
        )
        # Let the local receiver interact with the external
        # supplier_extrl. This will update the affected variable buffers
        # (e.g. Δmom for gravity) of the local receiver, and of the
        # external supplier if only_supply is False.
        if interact:
            with unswitch:
                if 𝔹[pairing_level == 'tile']:
                    # Get the supplier tiles with which to pair each
                    # receiver tile and perform the interaction
                    # at the tile level.
                    tile_pairings_index = get_tile_pairings(
                        interaction_name,
                        receiver,
                        rank_recv,
                        only_supply_communication,
                        domain_pair_nr,
                        tile_indices_receiver,
                        tile_indices_supplier,
                    )
                    tile_indices_supplier_paired   = tile_pairings_cache  [tile_pairings_index]
                    tile_indices_supplier_paired_N = tile_pairings_N_cache[tile_pairings_index]
                # Perform the interaction
                interaction(
                    𝕊[interaction_name if 𝔹[pairing_level == 'tile'] else 'trivial'],
                    receiver,
                    supplier_extrl,
                    ᔑdt_rungs,
                    rank_recv,
                    only_supply,
                    pairing_level,
                    tile_indices_receiver,
                    tile_indices_supplier_paired,
                    tile_indices_supplier_paired_N,
                    interaction_extra_args,
                )
        # Send the populated buffers (e.g. Δmom for gravity) back to the
        # process from which the external supplier_extrl came. Note that
        # we should not do this in the case of a local interaction
        # (rank_send == rank) or in a case where only_supply is True.
        if rank_send != rank and not only_supply:
            # For non-instantaneous interactions, the received Δ values
            # should be added to the Δ's of the local supplier_local.
            # For instantaneous interactions, the received Δ values
            # should be added directly to the data of the
            # local supplier_local.
            sendrecv_component(
                supplier_extrl, affected, pairing_level,
                interaction_name, tile_indices_supplier,
                dest=rank_recv, source=rank_send, component_recv=supplier_local,
                use_Δ_recv=(not instantaneous),
            )
            # Nullify the Δ buffers of the external supplier_extrl,
            # leaving this with no leftover junk.
            supplier_extrl.nullify_Δ(affected, only_active=False)
# Tile indices for the trivial tiling,
# used by the domain_domain function.
cython.declare(
    tile_indices_trivial='Py_ssize_t[::1]',
    tile_indices_trivial_paired='Py_ssize_t**',
    tile_indices_trivial_paired_N='Py_ssize_t*',
)
tile_indices_trivial = zeros(1, dtype=C2np['Py_ssize_t'])
tile_indices_trivial_paired = malloc(1*sizeof('Py_ssize_t*'))
tile_indices_trivial_paired[0] = cython.address(tile_indices_trivial[:])
tile_indices_trivial_paired_N = malloc(1*sizeof('Py_ssize_t'))
tile_indices_trivial_paired_N[0] = tile_indices_trivial.shape[0]

# Function returning the process ranks with which to pair
# the local process/domain in the domain_domain function,
# depending on the pairing level and supplier only supplies
# or also receives.
@cython.header(
    # Arguments
    pairing_level=str,
    only_supply='bint',
    # Locals
    i='Py_ssize_t',
    returns=tuple,
)
def domain_domain_communication(pairing_level, only_supply):
    ranks = domain_domain_communication_dict.get((pairing_level, only_supply))
    if ranks:
        return ranks
    if pairing_level == 'domain':
        # When only_supply is True, each process should be paired with
        # all processes. When only_supply is False, advantage is taken
        # of the fact that a process is paired with two other processes
        # simultaneously, meaning that the number of pairings is cut
        # (roughly) in half. The particular order implemented below
        # is of no importance.
        N_domain_pairs = nprocs if only_supply else 1 + nprocs//2
        ranks_send = empty(N_domain_pairs, dtype=C2np['int'])
        ranks_recv = empty(N_domain_pairs, dtype=C2np['int'])
        for i in range(N_domain_pairs):
            ranks_send[i] = mod(rank + i, nprocs)
            ranks_recv[i] = mod(rank - i, nprocs)
        domain_domain_communication_dict[pairing_level, only_supply] = (ranks_send, ranks_recv)
    elif pairing_level == 'tile':
        # When only_supply is True, each domian should be paired with
        # itself and all 26 neighbouring domains. Even though we might
        # have nprocs < 27, meaning that some of the neighbouring
        # domains might be the same, we always include all of them.
        # If only_supply is False, advantage is taken of the fact that a
        # domain is simultaneously paired with two other domains along
        # the same direction (e.g. to the left and to the right),
        # cutting the number of pairings (roughly) in half. The order is
        # as specified below, and stored (as directions, not ranks) in
        # domain_domain_communication_dict[
        #     'tile', only_supply, 'domain_pair_offsets'].
        ranks_send = []
        ranks_recv = []
        offsets_list = []
        # - This domain itself
        offsets = np.array([0, 0, 0], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        # - Domains at the 6 faces
        #   (when only_supply is False, send right, forward, upward)
        direction = np.array([+1, 0, 0], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        # - Domains at the 12 edges
        #   (when only_supply is False, send
        #     {right  , forward}, {left     , forward },
        #     {forward, upward }, {backward , upward  },
        #     {right  , upward }, {rightward, downward},
        # )
        direction = np.array([+1, +1,  0], dtype=C2np['int'])
        flip      = np.array([-1, +1, +1], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
            offsets *= np.roll(flip, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        # - Domains at the 8 corners
        #   (when only_supply is False, send
        #    {right, forward , upward  },
        #    {right, forward , downward},
        #    {left , forward , upward  },
        #    {right, backward, upward  },
        # )
        offsets = np.array([+1, +1, +1], dtype=C2np['int'])
        offsets_list.append(offsets.copy())
        ranks_send.append(rank_neighbouring_domain(*(+offsets)))
        ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
        if only_supply:
            offsets_list.append(-offsets)
            ranks_send.append(rank_neighbouring_domain(*(-offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        direction = np.array([+1, +1, -1], dtype=C2np['int'])
        for i in range(3):
            offsets = np.roll(direction, i)
            offsets_list.append(offsets.copy())
            ranks_send.append(rank_neighbouring_domain(*(+offsets)))
            ranks_recv.append(rank_neighbouring_domain(*(-offsets)))
            if only_supply:
                offsets_list.append(-offsets)
                ranks_send.append(rank_neighbouring_domain(*(-offsets)))
                ranks_recv.append(rank_neighbouring_domain(*(+offsets)))
        domain_domain_communication_dict[pairing_level, only_supply] = (
            (np.array(ranks_send, dtype=C2np['int']), np.array(ranks_recv, dtype=C2np['int']))
        )
        domain_domain_communication_dict[pairing_level, only_supply, 'domain_pair_offsets'] = (
            np.array(offsets_list, dtype=C2np['Py_ssize_t'])
        )
    else:
        abort(
            f'domain_domain_communication() got '
            f'pairing_level = {pairing_level} ∉ {{"domain", "tile"}}'
        )
    return domain_domain_communication_dict[pairing_level, only_supply]
# Cached results of the domain_domain_communication function
# are stored in the dict below.
cython.declare(domain_domain_communication_dict=dict)
domain_domain_communication_dict = {}

# Function returning the indices of the tiles of the local receiver and
# supplier which take part in tile-tile interactions under the
# domain-domain pairing with number domain_pair_nr.
@cython.header(
    # Arguments
    interaction_name=str,
    component='Component',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    # Locals
    dim='int',
    domain_pair_offsets='Py_ssize_t[:, ::1]',
    domain_pair_offset='Py_ssize_t[::1]',
    key=tuple,
    sign='int',
    tile_indices='Py_ssize_t[:, ::1]',
    tile_indices_all=list,
    tile_indices_component='Py_ssize_t[::1]',
    tile_indices_list=list,
    tile_layout='Py_ssize_t[:, :, ::1]',
    tile_layout_slice_end='Py_ssize_t[::1]',
    tile_layout_slice_start='Py_ssize_t[::1]',
    tiling='Tiling',
    tiling_name=str,
    returns='Py_ssize_t[:, ::1]',
)
def domain_domain_tile_indices(interaction_name, component, only_supply, domain_pair_nr):
    key = (interaction_name, only_supply)
    tile_indices_all = domain_domain_tile_indices_dict.get(key)
    if tile_indices_all is None:
        tile_indices_all = [None]*27
        domain_domain_tile_indices_dict[key] = tile_indices_all
    else:
        tile_indices = tile_indices_all[domain_pair_nr]
        if tile_indices is not None:
            return tile_indices
    tile_layout_slice_start = empty(3, dtype=C2np['Py_ssize_t'])
    tile_layout_slice_end   = empty(3, dtype=C2np['Py_ssize_t'])
    domain_pair_offsets = domain_domain_communication_dict[
        'tile', only_supply, 'domain_pair_offsets']
    domain_pair_offset = domain_pair_offsets[domain_pair_nr, :]
    tiling_name = f'{interaction_name} (tiles)'
    tiling = component.tilings[tiling_name]
    tile_layout = tiling.layout
    tile_indices_list = []
    for sign in range(-1, 2, 2):
        for dim in range(3):
            if domain_pair_offset[dim] == -sign:
                tile_layout_slice_start[dim] = 0
                tile_layout_slice_end[dim]   = 1
            elif domain_pair_offset[dim] == 0:
                tile_layout_slice_start[dim] = 0
                tile_layout_slice_end[dim]   = tile_layout.shape[dim]
            elif domain_pair_offset[dim] == +sign:
                tile_layout_slice_start[dim] = tile_layout.shape[dim] - 1
                tile_layout_slice_end[dim]   = tile_layout.shape[dim]
        tile_indices_component = asarray(tile_layout[
            tile_layout_slice_start[0]:tile_layout_slice_end[0],
            tile_layout_slice_start[1]:tile_layout_slice_end[1],
            tile_layout_slice_start[2]:tile_layout_slice_end[2],
        ]).flatten()
        tile_indices_list.append(tile_indices_component)
    tile_indices = asarray(tile_indices_list, dtype=C2np['Py_ssize_t'])
    tile_indices_all[domain_pair_nr] = tile_indices
    return tile_indices
# Cached results of the domain_domain_tile_indices function
# are stored in the dict below.
cython.declare(domain_domain_tile_indices_dict=dict)
domain_domain_tile_indices_dict = {}

# Function that given arrays of receiver and supplier tiles
# returns them in paired format.
@cython.header(
    # Arguments
    interaction_name=str,
    component='Component',
    rank_supplier='int',
    only_supply='bint',
    domain_pair_nr='Py_ssize_t',
    tile_indices_receiver='Py_ssize_t[::1]',
    tile_indices_supplier='Py_ssize_t[::1]',
    # Locals
    dim='int',
    domain_pair_offset='Py_ssize_t[::1]',
    global_tile_layout_shape='Py_ssize_t[::1]',
    i='Py_ssize_t',
    j='Py_ssize_t',
    key=tuple,
    l='Py_ssize_t',
    l_offset='Py_ssize_t',
    l_s='Py_ssize_t',
    m='Py_ssize_t',
    m_offset='Py_ssize_t',
    m_s='Py_ssize_t',
    n='Py_ssize_t',
    n_offset='Py_ssize_t',
    n_s='Py_ssize_t',
    neighbourtile_index_3D_global='Py_ssize_t[::1]',
    pairings='Py_ssize_t**',
    pairings_N='Py_ssize_t*',
    pairs_N='Py_ssize_t',
    suppliertile_indices_3D_global_to_1D_local=dict,
    tile_index_3D_global_s=tuple,
    tile_index_r='Py_ssize_t',
    tile_index_s='Py_ssize_t',
    tile_indices_supplier_paired='Py_ssize_t[::1]',
    tile_indices_supplier_paired_ptr='Py_ssize_t*',
    tile_layout='Py_ssize_t[:, :, ::1]',
    tile_pairings_index='Py_ssize_t',
    tiling='Tiling',
    tiling_name=str,
    wraparound='bint',
    returns='Py_ssize_t',
)
def get_tile_pairings(
    interaction_name, component, rank_supplier, only_supply,
    domain_pair_nr, tile_indices_receiver, tile_indices_supplier,
):
    global tile_pairings_cache, tile_pairings_N_cache, tile_pairings_cache_size
    # Lookup index of the required tile pairings in the global cache
    key = (interaction_name, domain_pair_nr, only_supply)
    tile_pairings_index = tile_pairings_cache_indices.get(key, tile_pairings_cache_size)
    if tile_pairings_index < tile_pairings_cache_size:
        return tile_pairings_index
    # No cached results found. We will now compute the supplier tile
    # indices to be paired with each of the receiver tiles.
    # Below is a list of lists storing the supplier tile indices
    # for each receiver tile. The type of this data structure will
    # change during the computation.
    tile_indices_receiver_supplier = [[] for i in range(tile_indices_receiver.shape[0])]
    # Get the shape of the local (domain) tile layout,
    # as well as of the global (box) tile layout.
    tiling_name = f'{interaction_name} (tiles)'
    tiling = component.tilings[tiling_name]
    tile_layout = tiling.layout
    tile_layout_shape = asarray(asarray(tile_layout).shape)
    # The general computation below takes a long time when dealing with
    # many tiles. By far the worst case is when all tiles in the local
    # domain should be paired with themselves, which is the case for
    # domain_pair_nr == 0. For this case we perform a much faster,
    # more specialised computation.
    if domain_pair_nr == 0:
        if rank != rank_supplier:
            abort(
                f'get_tile_pairings() got rank_supplier = {rank_supplier} != rank = {rank} '
                f'at domain_pair_nr == 0'
            )
        if not np.all(asarray(tile_indices_receiver) == asarray(tile_indices_supplier)):
            abort(
                f'get_tile_pairings() got tile_indices_receiver != tile_indices_supplier '
                f'at domain_pair_nr == 0'
            )
        i = 0
        for         l in range(ℤ[tile_layout.shape[0]]):
            for     m in range(ℤ[tile_layout.shape[1]]):
                for n in range(ℤ[tile_layout.shape[2]]):
                    if i != tile_layout[l, m, n]:
                        abort(
                            f'It looks as though the tile layout of {component.name} is incorrect'
                        )
                    neighbourtile_indices_supplier = tile_indices_receiver_supplier[i]
                    for l_offset in range(-1, 2):
                        l_s = l + l_offset
                        if l_s == -1 or l_s == ℤ[tile_layout.shape[0]]:
                            continue
                        for m_offset in range(-1, 2):
                            m_s = m + m_offset
                            if m_s == -1 or m_s == ℤ[tile_layout.shape[1]]:
                                continue
                            for n_offset in range(-1, 2):
                                n_s = n + n_offset
                                if n_s == -1 or n_s == ℤ[tile_layout.shape[2]]:
                                    continue
                                tile_index_s = tile_layout[l_s, m_s, n_s]
                                # As domain_pair_nr == 0, all tiles in
                                # the local domain are paired with all
                                # others. To not double count, we
                                # disregard the pairing if the supplier
                                # tile index is lower than the receiver
                                # tile index (i). However, if
                                # only_supply is True, there is no
                                # double counting to be considered (the
                                # two components are presumably
                                # different), and so here we do not
                                # disregard the pairing.
                                with unswitch:
                                    if not only_supply:
                                        if tile_index_s < i:
                                            continue
                                neighbourtile_indices_supplier.append(tile_index_s)
                    tile_indices_receiver_supplier[i] = asarray(
                        neighbourtile_indices_supplier, dtype=C2np['Py_ssize_t'],
                    )
                    i += 1
    else:
        # Get relative offsets of the domains currently being paired
        domain_pair_offset = domain_domain_communication_dict[
            'tile', only_supply, 'domain_pair_offsets'][domain_pair_nr, :]
        # Get the indices of the global domain layout matching the
        # receiver (local) domain and supplier domain.
        domain_layout_receiver_indices = asarray(
            np.unravel_index(rank, domain_subdivisions)
        )
        domain_layout_supplier_indices = asarray(
            np.unravel_index(rank_supplier, domain_subdivisions)
        )
        global_tile_layout_shape = asarray(
            asarray(domain_subdivisions)*tile_layout_shape,
            dtype=C2np['Py_ssize_t'],
        )
        tile_index_3D_r_start = domain_layout_receiver_indices*tile_layout_shape
        tile_index_3D_s_start = domain_layout_supplier_indices*tile_layout_shape
        # Construct dict mapping global supplier 3D indices to their
        # local 1D counterparts.
        suppliertile_indices_3D_global_to_1D_local = {}
        for j in range(tile_indices_supplier.shape[0]):
            tile_index_s = tile_indices_supplier[j]
            tile_index_3D_s = asarray(tiling.tile_index3D(tile_index_s))
            tile_index_3D_global_s = tuple(tile_index_3D_s + tile_index_3D_s_start)
            suppliertile_indices_3D_global_to_1D_local[tile_index_3D_global_s] = tile_index_s
        # Pair each receiver tile with all neighbouring supplier tiles
        for i in range(tile_indices_receiver.shape[0]):
            neighbourtile_indices_supplier = tile_indices_receiver_supplier[i]
            # Construct global 3D index of this receiver tile
            tile_index_r = tile_indices_receiver[i]
            tile_index_3D_r = asarray(tiling.tile_index3D(tile_index_r))
            tile_index_3D_global_r = tile_index_3D_r + tile_index_3D_r_start
            # Loop over all neighbouring receiver tiles
            # (including the tile itself).
            for         l in range(-1, 2):
                for     m in range(-1, 2):
                    for n in range(-1, 2):
                        neighbourtile_index_3D_global = asarray(
                            tile_index_3D_global_r + asarray((l, m, n)),
                            dtype=C2np['Py_ssize_t'],
                        )
                        # For domain_pair_nr == 0, all tiles in the
                        # local domain are paired with all others.
                        # Here we must not take the periodicity into
                        # account, as such interacions are performed by
                        # future domain pairings.
                        with unswitch:
                            if domain_pair_nr == 0:
                                wraparound = False
                                for dim in range(3):
                                    if not (
                                        0 <= neighbourtile_index_3D_global[dim]
                                          <  global_tile_layout_shape[dim]
                                    ):
                                        wraparound = True
                                        break
                                if wraparound:
                                    continue
                        # Take the periodicity of the domain layout
                        # into account. This should only be done
                        # along the direction(s) connecting
                        # the paired domains.
                        with unswitch:
                            if 𝔹[domain_pair_offset[0] != 0]:
                                neighbourtile_index_3D_global[0] = mod(
                                    neighbourtile_index_3D_global[0],
                                    global_tile_layout_shape[0],
                                )
                        with unswitch:
                            if 𝔹[domain_pair_offset[1] != 0]:
                                neighbourtile_index_3D_global[1] = mod(
                                    neighbourtile_index_3D_global[1],
                                    global_tile_layout_shape[1],
                                )
                        with unswitch:
                            if 𝔹[domain_pair_offset[2] != 0]:
                                neighbourtile_index_3D_global[2] = mod(
                                    neighbourtile_index_3D_global[2],
                                    global_tile_layout_shape[2],
                                )
                        # Check if a supplier tile sits at the location
                        # of the current neighbour tile.
                        tile_index_s = suppliertile_indices_3D_global_to_1D_local.get(
                            tuple(neighbourtile_index_3D_global),
                            -1,
                        )
                        if tile_index_s != -1:
                            # For domain_pair_nr == 0, all tiles in the
                            # local domain are paired with all others.
                            # To not double count, we disregard the
                            # pairing if the supplier tile index is
                            # lower than the receiver tile index.
                            # However, if only_supply is True, there is
                            # no double counting to be considered (the
                            # two components are presumably different),
                            # and so here we do not disregard
                            # the pairing.
                            with unswitch:
                                if domain_pair_nr == 0 and not only_supply:
                                    if tile_index_s < tile_index_r:
                                        continue
                            neighbourtile_indices_supplier.append(tile_index_s)
            # Convert the neighbouring supplier tile indices from a list
            # to an Py_ssize_t[::1] array.
            # We also sort the indices, though this is not necessary.
            neighbourtile_indices_supplier = asarray(
                neighbourtile_indices_supplier, dtype=C2np['Py_ssize_t'],
            )
            neighbourtile_indices_supplier.sort()
            tile_indices_receiver_supplier[i] = neighbourtile_indices_supplier
    # Transform tile_indices_receiver_supplier to an object array,
    # the elements of which are arrays of dtype Py_ssize_t.
    tile_indices_receiver_supplier = asarray(tile_indices_receiver_supplier, dtype=object)
    # If all arrays in tile_indices_receiver_supplier are of the
    # same size, it will not be stored as an object array of Py_ssize_t
    # arrays, but instead a 2D object array. In compiled mode this leads
    # to a crash, as elements of tile_indices_receiver_supplier must be
    # compatible with Py_ssize_t[::1]. We can convert it to a 2D
    # Py_ssize_t array instead, single-index elements of which exactly
    # are the Py_ssize_t[::1] arrays we need. When the arrays in
    # tile_indices_receiver_supplier are not of the same size, such a
    # conversion will fail. Since we do not care about the conversion
    # in such a case anyway, we always just attempt to do
    # the conversion. If it succeed, it was needed. If not, it was not
    # needed anyway.
    try:
        tile_indices_receiver_supplier = asarray(
            tile_indices_receiver_supplier, dtype=C2np['Py_ssize_t'])
    except ValueError:
        pass
    # Cache the result. This cache is not actually used, but it ensures
    # that Python will not garbage collect the data.
    tile_indices_receiver_supplier_dict[key] = tile_indices_receiver_supplier
    # Now comes the caching that is actually used, where we use pointers
    # rather than Python objects.
    pairs_N = tile_indices_receiver_supplier.shape[0]
    pairings   = malloc(pairs_N*sizeof('Py_ssize_t*'))
    pairings_N = malloc(pairs_N*sizeof('Py_ssize_t'))
    for i in range(pairs_N):
        tile_indices_supplier_paired = tile_indices_receiver_supplier[i]
        tile_indices_supplier_paired_ptr = cython.address(tile_indices_supplier_paired[:])
        pairings[i] = tile_indices_supplier_paired_ptr
        pairings_N[i] = tile_indices_supplier_paired.shape[0]
    tile_pairings_cache_indices[key] = tile_pairings_index
    tile_pairings_cache_size += 1
    tile_pairings_cache = realloc(
        tile_pairings_cache,
        tile_pairings_cache_size*sizeof('Py_ssize_t**'),
    )
    tile_pairings_N_cache = realloc(
        tile_pairings_N_cache,
        tile_pairings_cache_size*sizeof('Py_ssize_t*'),
    )
    tile_pairings_cache  [tile_pairings_index] = pairings
    tile_pairings_N_cache[tile_pairings_index] = pairings_N
    return tile_pairings_index
# Caches used by the get_tile_pairings function
cython.declare(
    tile_indices_receiver_supplier_dict=dict,
    tile_pairings_cache_indices=dict,
    tile_pairings_cache_size='Py_ssize_t',
    tile_pairings_cache='Py_ssize_t***',
    tile_pairings_N_cache='Py_ssize_t**',
)
tile_indices_receiver_supplier_dict = {}
tile_pairings_cache_indices = {}
tile_pairings_cache_size = 0
tile_pairings_cache   = malloc(tile_pairings_cache_size*sizeof('Py_ssize_t**'))
tile_pairings_N_cache = malloc(tile_pairings_cache_size*sizeof('Py_ssize_t*'))

# Function responsible for constructing pairings between subtiles within
# the supplied subtiling, including the corresponding subtiles in the 26
# neighbour tiles. Subtiles further away than the supplied forcerange
# will not be paired.
@cython.header(
    # Arguments
    subtiling='Tiling',
    forcerange='double',
    only_supply='bint',
    # Locals
    all_pairings='Py_ssize_t***',
    all_pairings_N='Py_ssize_t**',
    dim='int',
    extent_over_range_dim='double',
    key=tuple,
    key_quick=tuple,
    pairing_index='Py_ssize_t',
    pairings='Py_ssize_t**',
    pairings_N='Py_ssize_t*',
    pairings_r='Py_ssize_t*',
    r_dim='Py_ssize_t',
    r2='double',
    same_tile='bint',
    shape='Py_ssize_t[::1]',
    size='Py_ssize_t',
    subtile_index_r='Py_ssize_t',
    subtile_index_s='Py_ssize_t',
    subtile_index3D='Py_ssize_t[::1]',
    subtile_index3D_r='Py_ssize_t[::1]',
    subtile_index3D_s='Py_ssize_t[::1]',
    subtile_pairings_index='Py_ssize_t',
    tile_extent='double[::1]',
    tile_pair_index='Py_ssize_t',
    tiles_offset='Py_ssize_t[::1]',
    tiles_offset_i='Py_ssize_t',
    tiles_offset_j='Py_ssize_t',
    tiles_offset_k='Py_ssize_t',
    returns='Py_ssize_t',
)
def get_subtile_pairings(subtiling, forcerange, only_supply):
    global subtile_pairings_cache, subtile_pairings_N_cache, subtile_pairings_cache_size
    # Lookup index of the required subtile pairings in the global cache.
    # We first try a quick lookup using a key containing the passed
    # subtiling instance. The attributes (e.g. shape and extent)
    # on a subtiling instance must then never be redefined.
    key_quick = (subtiling, forcerange, only_supply)
    subtile_pairings_index = subtile_pairings_cache_indices.get(
        key_quick,
        subtile_pairings_cache_size,
    )
    if subtile_pairings_index < subtile_pairings_cache_size:
        return subtile_pairings_index
    # The subtile pairings was not found in the cache. It is possible
    # that a different subtiling instance with the same shape and the
    # same extent in units of the forcerange is present in the cache.
    # All results are therefore also stored using keys containing the
    # shape and extent/forcerange. Try this more involved lookup.
    for dim in range(3):
        extent_over_range_dim = subtiling.extent[dim]*ℝ[1/forcerange]
        extent_over_range[dim] = float(f'{extent_over_range_dim:.12g}')
    shape = subtiling.shape
    key = (tuple(shape), tuple(extent_over_range), forcerange, only_supply)
    subtile_pairings_index = subtile_pairings_cache_indices.get(key, subtile_pairings_cache_size)
    if subtile_pairings_index < subtile_pairings_cache_size:
        # Found in cache. Add the missing, quick key.
        subtile_pairings_cache_indices[key_quick] = subtile_pairings_index
        return subtile_pairings_index
    # No cached results found. Create subtile pairings
    # for each of the 27 cases of neighbour tiles.
    size = subtiling.size
    tile_extent = subtiling.tile_extent
    all_pairings   = malloc(27*sizeof('Py_ssize_t**'))
    all_pairings_N = malloc(27*sizeof('Py_ssize_t*'))
    tiles_offset      = empty(3, dtype=C2np['Py_ssize_t'])
    subtile_index3D_r = empty(3, dtype=C2np['Py_ssize_t'])
    same_tile = False
    for tiles_offset_i in range(-1, 2):
        tiles_offset[0] = tiles_offset_i
        for tiles_offset_j in range(-1, 2):
            tiles_offset[1] = tiles_offset_j
            for tiles_offset_k in range(-1, 2):
                tiles_offset[2] = tiles_offset_k
                # Does the tile offset correspond to
                # a tile being paired with itself?
                with unswitch:
                    if not only_supply:
                        same_tile = (tiles_offset_i == tiles_offset_j == tiles_offset_k == 0)
                # Get 1D tile pair index from the 3D offset
                tile_pair_index = get_neighbourtile_pair_index(tiles_offset)
                # Allocate memory for subtile pairings
                # for this particular tile pair.
                pairings   = malloc(size*sizeof('Py_ssize_t*'))
                pairings_N = malloc(size*sizeof('Py_ssize_t'))
                all_pairings  [tile_pair_index] = pairings
                all_pairings_N[tile_pair_index] = pairings_N
                # Loop over all receiver subtiles
                for subtile_index_r in range(size):
                    # Get 3D subtile index. As the tile_index3D() method
                    # return a view over internal data and we mutate
                    # subtile_index3D_r below, we take a copy of the
                    # returned data.
                    subtile_index3D = subtiling.tile_index3D(subtile_index_r)
                    for dim in range(3):
                        subtile_index3D_r[dim] = subtile_index3D[dim]
                    # The receiver and supplier subtiles belong to
                    # (potentially) diffent tiles, with a relative
                    # offset given by tiles_offset_*, so that the
                    # supplier tile is at the receiver tile location
                    # plus tiles_offset_*. We now subtract this offset
                    # from the receiver 3D subtile index, so that the
                    # difference in subtile indices between the receiver
                    # and supplier subtile is proportional to their
                    # physical separation. Note that subtile_index3D_r
                    # no longer represents the actual index in memory.
                    for dim in range(3):
                        subtile_index3D_r[dim] -= tiles_offset[dim]*shape[dim]
                    # Allocate memory for subtile pairings with this
                    # particular receiver subtile.
                    # We give it the maximum possible needed memory.
                    pairings_r = malloc(size*sizeof('Py_ssize_t'))
                    pairings[subtile_index_r] = pairings_r
                    # Pair receiver subtile with every supplier subtile,
                    # unless the tile is being paired with itself.
                    # In that case, we need to not double count the
                    # subtile pairing (while still pairing every subtile
                    # with themselves).
                    pairing_index = 0
                    for subtile_index_s in range(subtile_index_r if same_tile else 0, size):
                        subtile_index3D_s = subtiling.tile_index3D(subtile_index_s)
                        # Measure (squared) distance between the subtile
                        # pair and reject if larger than the passed
                        # forcerange.
                        r2 = 0
                        for dim in range(3):
                            # Distance between the same point in the two
                            # subtiles along the dim'th dimenson,
                            # in subtile grid units.
                            r_dim = abs(subtile_index3D_r[dim] - subtile_index3D_s[dim])
                            if r_dim > 0:
                                # The two subtiles are offset along the
                                # dim'th dimension. Subtract one unit
                                # from the length, making the length
                                # between the closest two points
                                # in the two subtiles.
                                r_dim -= 1
                            r2 += (r_dim*tile_extent[dim])**2
                        if r2 > ℝ[forcerange**2]:
                            continue
                        # Add this supplier subtile to the list of
                        # pairing partners for this receiver subtile.
                        pairings_r[pairing_index] = subtile_index_s
                        pairing_index += 1
                    # All pairs found for this receiver subtile.
                    # Truncate the allocated memory as to only contain
                    # the used chunk.
                    pairings[subtile_index_r] = realloc(
                        pairings_r, pairing_index*sizeof('Py_ssize_t'),
                    )
                    # Record the size of this pairing array
                    pairings_N[subtile_index_r] = pairing_index
    # Store results in global caches
    subtile_pairings_cache_indices[key_quick] = subtile_pairings_index
    subtile_pairings_cache_indices[key      ] = subtile_pairings_index
    subtile_pairings_cache_size += 1
    subtile_pairings_cache = realloc(
        subtile_pairings_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t***'),
    )
    subtile_pairings_N_cache = realloc(
        subtile_pairings_N_cache, subtile_pairings_cache_size*sizeof('Py_ssize_t**'),
    )
    subtile_pairings_cache  [subtile_pairings_index] = all_pairings
    subtile_pairings_N_cache[subtile_pairings_index] = all_pairings_N
    # Return cached results in form of the cache index
    return subtile_pairings_index
# Caches used by the get_subtile_pairings function
cython.declare(
    extent_over_range='double[::1]',
    subtile_pairings_cache_indices=dict,
    subtile_pairings_cache_size='Py_ssize_t',
    subtile_pairings_cache='Py_ssize_t****',
    subtile_pairings_N_cache='Py_ssize_t***',
)
extent_over_range = empty(3, dtype=C2np['double'])
subtile_pairings_cache_indices = {}
subtile_pairings_cache_size = 0
subtile_pairings_cache   = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t***'))
subtile_pairings_N_cache = malloc(subtile_pairings_cache_size*sizeof('Py_ssize_t**'))

# Helper function for the get_subtile_pairings function
@cython.header(
    # Arguments
    tiles_offset='Py_ssize_t[::1]',
    # Locals
    dim='int',
    returns='Py_ssize_t',
)
def get_neighbourtile_pair_index(tiles_offset):
    # The passed tiles_offset is the relative offset between a pair of
    # neighbouring tiles, and so each of its three elements has to be
    # in {-1, 0, +1}. If any element is outside this range, it is due
    # to the periodic boundaries. Fix this now, as we do not care about
    # whether the tile pair is connected through the box boundary.
    for dim in range(3):
        if tiles_offset[dim] > 1:
            tiles_offset[dim] = -1
        elif tiles_offset[dim] < -1:
            tiles_offset[dim] = +1
    # Compute 1D index from a 3×3×3 shape. We add 1 to each element,
    # as they range from -1 to +1.
    return ((tiles_offset[0] + 1)*3 + (tiles_offset[1] + 1))*3 + (tiles_offset[2] + 1)

# Generic function implementing particle-particle pairing.
# Note that this function returns a generator and so should only be
# called within a loop.
@cython.iterator(
    depends=[
        # Global variables used by particle_particle()
        'tile_location_r',
        'tile_location_r_ptr',
        'tile_location_s',
        'tile_location_s_ptr',
        'tiles_offset',
        'tiles_offset_ptr',
        # Functions used by particle_particle()
        'get_subtile_pairings',
            # Global variables used by get_subtile_pairings()
            'extent_over_range',
            'subtile_pairings_cache_indices',
            'subtile_pairings_cache_size',
            'subtile_pairings_cache',
            'subtile_pairings_N_cache',
        'get_neighbourtile_pair_index',
    ]
)
def particle_particle(
    receiver, supplier, pairing_level,
    tile_indices_receiver, tile_indices_supplier_paired, tile_indices_supplier_paired_N,
    rank_supplier, interaction_name, only_supply,
):
    # Cython declarations for variables used for the iteration,
    # not including those to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        N_subtiles='Py_ssize_t',
        all_subtile_pairings='Py_ssize_t***',
        all_subtile_pairings_N='Py_ssize_t**',
        dim='int',
        forcerange='double',
        highest_populated_rung_r='signed char',
        highest_populated_rung_s='signed char',
        lowest_active_rung_r='signed char',
        lowest_active_rung_s='signed char',
        lowest_populated_rung_r='signed char',
        lowest_populated_rung_s='signed char',
        only_supply_communication='bint',
        posx_r='double*',
        posx_s='double*',
        posy_r='double*',
        posy_s='double*',
        posz_r='double*',
        posz_s='double*',
        rung_index_s_start='signed char',
        rung_particle_index_r='Py_ssize_t',
        rung_particle_index_s='Py_ssize_t',
        rung_particle_index_s_start='Py_ssize_t',
        rung_N_r='Py_ssize_t',
        rung_N_s='Py_ssize_t',
        rung_index_r='signed char',
        rung_index_s='signed char',
        rung_jump_r='signed char',
        rung_jump_s='signed char',
        rung_jumps_r='signed char*',
        rung_jumps_s='signed char*',
        rung_r='Py_ssize_t*',
        rung_s='Py_ssize_t*',
        rungs_N_r='Py_ssize_t*',
        rungs_N_s='Py_ssize_t*',
        subtile_contain_particles_r='signed char',
        subtile_contain_particles_s='signed char',
        subtile_index_r='Py_ssize_t',
        subtile_index_s='Py_ssize_t',
        subtile_pairings='Py_ssize_t**',
        subtile_pairings_N='Py_ssize_t*',
        subtile_pairings_N_r='Py_ssize_t',
        subtile_pairings_index='Py_ssize_t',
        subtile_pairings_r='Py_ssize_t*',
        subtile_r='Py_ssize_t**',
        subtile_s='Py_ssize_t**',
        subtiles_contain_particles_r='signed char*',
        subtiles_contain_particles_s='signed char*',
        subtiles_r='Py_ssize_t***',
        subtiles_rungs_N_r='Py_ssize_t**',
        subtiles_rungs_N_s='Py_ssize_t**',
        subtiles_s='Py_ssize_t***',
        subtiling_name=str,
        subtiling_name_2=str,
        subtiling_s='Tiling',
        subtiling_s_2='Tiling',
        tile_contain_particles_r='signed char',
        tile_contain_particles_s='signed char',
        tile_extent='double*',
        tile_index_r='Py_ssize_t',
        tile_index_s='Py_ssize_t',
        tile_index3D='Py_ssize_t[::1]',
        tile_index3D_r='Py_ssize_t*',
        tile_index3D_s='Py_ssize_t*',
        tile_indices_supplier='Py_ssize_t*',
        tile_indices_supplier_N='Py_ssize_t',
        tile_pair_index='Py_ssize_t',
        tiles_contain_particles_r='signed char*',
        tiles_contain_particles_s='signed char*',
        tiles_r='Py_ssize_t***',
        tiles_s='Py_ssize_t***',
        tiling_location_r='double*',
        tiling_location_s='double*',
        tiling_name=str,
        tiling_r='Tiling',
        tiling_s='Tiling',
        xi='double',
        yi='double',
        zi='double',
    )
    # Extract particle variables from the receiver component
    posx_r = receiver.posx
    posy_r = receiver.posy
    posz_r = receiver.posz
    lowest_active_rung_r     = receiver.lowest_active_rung
    lowest_populated_rung_r  = receiver.lowest_populated_rung
    highest_populated_rung_r = receiver.highest_populated_rung
    # Extract particle variables from the supplier
    # (the external) component.
    posx_s = supplier.posx
    posy_s = supplier.posy
    posz_s = supplier.posz
    lowest_active_rung_s     = supplier.lowest_active_rung
    lowest_populated_rung_s  = supplier.lowest_populated_rung
    highest_populated_rung_s = supplier.highest_populated_rung
    rung_jumps_r = receiver.rung_jumps
    rung_jumps_s = supplier.rung_jumps
    # The names used to refer to the domain and tile level tiling
    # (tiles and subtiles). In the case of pairing_level == 'domain',
    # we always use the trivial tiling.
    if 𝔹[pairing_level == 'tile']:
        tiling_name    = f'{interaction_name} (tiles)'
        subtiling_name = f'{interaction_name} (subtiles)'
    else:  # pairing_level == 'domain':
        tiling_name = subtiling_name = 'trivial'
    # Extract tiling variables from receiver
    tiling_r = receiver.tilings[tiling_name]
    tiling_location_r         = cython.address(tiling_r.location[:])
    tile_extent               = cython.address(tiling_r.tile_extent[:])  # The same for receiver and supplier
    tiles_r                   = tiling_r.tiles
    tiles_contain_particles_r = tiling_r.contain_particles
    subtiling_r = receiver.tilings[subtiling_name]
    subtiles_r                   = subtiling_r.tiles
    subtiles_contain_particles_r = subtiling_r.contain_particles
    N_subtiles                   = subtiling_r.size  # The same for receiver and supplier
    # Extract tiling variables from supplier
    tiling_s = supplier.tilings[tiling_name]
    tiling_location_s         = cython.address(tiling_s.location[:])
    tiles_s                   = tiling_s.tiles
    tiles_contain_particles_s = tiling_s.contain_particles
    subtiling_s = supplier.tilings[subtiling_name]
    subtiles_s                   = subtiling_s.tiles
    subtiles_contain_particles_s = subtiling_s.contain_particles
    # When the receiver and supplier components are the same
    # and the receiver and supplier domains are also the same,
    # we now have a case where (tiling_r is tiling_s) and
    # (subtiling_r is subtiling_s) are both True. This is OK for
    # the coarse tiling, but not for the subtiling, as here we need
    # to re-sort the particles during the iteration below. That is,
    # we need to keep track of the sorting of the receiver tiles
    # into subtiles while also keeping track of the sorting of the
    # supplier tiles into subtiles. We thus always need two separate
    # subtiling_{r/s} instances, which we do not have in the case
    # mentioned. When this is the case, we make use of a second,
    # separate Tiling instance. If however the subtiling in use is the
    # trivial tiling, the re-sorting has no effect, and so we do not
    # have to worry.
    if 𝔹[receiver.name == supplier.name and rank == rank_supplier and subtiling_name != 'trivial']:
        subtiling_name_2 = f'{interaction_name} (subtiles 2)'
        if subtiling_name_2 not in supplier.tilings:
            supplier.tilings.pop(subtiling_name)
            subtiling_s_2 = supplier.init_tiling(subtiling_name)
            supplier.tilings[subtiling_name  ] = subtiling_s
            supplier.tilings[subtiling_name_2] = subtiling_s_2
        subtiling_s = supplier.tilings[subtiling_name_2]
        subtiles_s                   = subtiling_s.tiles
        subtiles_contain_particles_s = subtiling_s.contain_particles
    # Get subtile pairings between each
    # of the 27 possible tile pairings.
    only_supply_communication = (only_supply if receiver.name == supplier.name else True)
    forcerange = get_shortrange_param((receiver, supplier), interaction_name, 'range')
    subtile_pairings_index = get_subtile_pairings(
        subtiling_r, forcerange, only_supply_communication,
    )
    all_subtile_pairings = subtile_pairings_cache[subtile_pairings_index]
    all_subtile_pairings_N = subtile_pairings_N_cache[subtile_pairings_index]
    # Flags specifying whether the force betweeen particle i and j
    # should be applied to i and j. If only_supply is True,
    # the values below are correct. Otherwise, other values
    # will be set further down.
    apply_to_i = True
    apply_to_j = False
    # The current time. This is yielded back to the caller,
    # where time() - particle_particle_t_begin should be added to the
    # computation_time of the receiver subtiling. This is used for the
    # automatic subtiling refinement and the load imbalancing printout.
    particle_particle_t_begin = time()
    # Loop over the requested tiles in the receiver
    for tile_index_r in range(ℤ[tile_indices_receiver.shape[0]]):
        # Lookup supplier tile indices with which to pair the current
        # receiver tile.
        tile_indices_supplier   = tile_indices_supplier_paired  [tile_index_r]
        tile_indices_supplier_N = tile_indices_supplier_paired_N[tile_index_r]
        # Now make tile_index_r an actual receiver tile index
        tile_index_r = tile_indices_receiver[tile_index_r]
        # Skip tile if it does not contain any particles at all,
        # or only inactive particles when only_supply is True.
        tile_contain_particles_r = tiles_contain_particles_r[tile_index_r]
        with unswitch(1):
            if 𝔹[not only_supply]:
                if tile_contain_particles_r == 0:
                    continue
            else:
                if tile_contain_particles_r < 2:
                    continue
        # Sort particles within the receiver tile into subtiles
        tile_index3D = tiling_r.tile_index3D(tile_index_r)
        tile_index3D_r = cython.address(tile_index3D[:])
        for dim in range(3):
            tile_location_r_ptr[dim] = (
                tiling_location_r[dim] + tile_index3D_r[dim]*tile_extent[dim]
            )
        subtiling_r.relocate(tile_location_r)
        subtiling_r.sort(tiling_r, tile_index_r)
        subtiles_rungs_N_r = subtiling_r.tiles_rungs_N
        # Loop over the requested tiles in the supplier
        for tile_index_s in range(tile_indices_supplier_N):
            tile_index_s = tile_indices_supplier[tile_index_s]
            # Skip tile if it does not contain any particles at all
            tile_contain_particles_s = tiles_contain_particles_s[tile_index_s]
            if tile_contain_particles_s == 0:
                continue
            # If both the receiver and supplier tile contains particles
            # on inactive rows only, we skip this tile pair.
            if True:  # with unswitch(1):
                if tile_contain_particles_r == 1:
                    if tile_contain_particles_s == 1:
                        continue
            # Sort particles within the supplier tile into subtiles
            tile_index3D = tiling_s.tile_index3D(tile_index_s)
            tile_index3D_s = cython.address(tile_index3D[:])
            for dim in range(3):
                tile_location_s_ptr[dim] = (
                    tiling_location_s[dim] + tile_index3D_s[dim]*tile_extent[dim]
                )
            subtiling_s.relocate(tile_location_s)
            subtiling_s.sort(tiling_s, tile_index_s)
            subtiles_rungs_N_s = subtiling_s.tiles_rungs_N
            # Get the needed subtile pairings for the selected receiver
            # and supplier tiles (which should be neighbour tiles).
            for dim in range(3):
                tiles_offset_ptr[dim] = tile_index3D_s[dim] - tile_index3D_r[dim]
            tile_pair_index = get_neighbourtile_pair_index(tiles_offset)
            subtile_pairings   = all_subtile_pairings  [tile_pair_index]
            subtile_pairings_N = all_subtile_pairings_N[tile_pair_index]
            # Loop over all subtiles in the selected receiver tile
            for subtile_index_r in range(N_subtiles):
                # Skip subtile if it does not contain
                # any particles at all, or only inactive particles
                # when only_supply is True.
                subtile_contain_particles_r = subtiles_contain_particles_r[subtile_index_r]
                with unswitch(3):
                    if 𝔹[not only_supply]:
                        if subtile_contain_particles_r == 0:
                            continue
                    else:
                        if subtile_contain_particles_r < 2:
                            continue
                subtile_r = subtiles_r[subtile_index_r]
                rungs_N_r = subtiles_rungs_N_r[subtile_index_r]
                subtile_pairings_r   = subtile_pairings  [subtile_index_r]
                subtile_pairings_N_r = subtile_pairings_N[subtile_index_r]
                # Loop over the needed supplier subtiles
                for subtile_index_s in range(subtile_pairings_N_r):
                    subtile_index_s = subtile_pairings_r[subtile_index_s]
                    # Skip subtile if it does not contain
                    # any particles at all.
                    subtile_contain_particles_s = subtiles_contain_particles_s[subtile_index_s]
                    if subtile_contain_particles_s == 0:
                        continue
                    # If both the receiver and supplier subtile contains
                    # particles on inactive rows only, we skip this
                    # subtile pair.
                    if True:  # with unswitch(1):
                        if subtile_contain_particles_r == 1:
                            if subtile_contain_particles_s == 1:
                                continue
                    subtile_s = subtiles_s[subtile_index_s]
                    rungs_N_s = subtiles_rungs_N_s[subtile_index_s]
                    # Loop over all rungs in the receiver subtile
                    for rung_index_r in range(
                        ℤ[lowest_active_rung_r if only_supply else lowest_populated_rung_r],
                        ℤ[highest_populated_rung_r + 1],
                    ):
                        rung_N_r = rungs_N_r[rung_index_r]
                        if rung_N_r == 0:
                            continue
                        rung_r = subtile_r[rung_index_r]
                        # We need to pair all active receiver rungs
                        # with all supplier rungs. All inactive
                        # receiver rungs need only to be paired with
                        # the active supplier rungs (i.e. we do not need
                        # to pair up two inacive rungs).
                        # If only_supply is True, the values already set
                        # will be used.
                        rung_index_s_start = lowest_populated_rung_s
                        with unswitch(5):
                            if 𝔹[not only_supply]:
                                if rung_index_r < lowest_active_rung_r:
                                    # Only the supplier should receive
                                    # a kick.
                                    apply_to_i = False
                                    rung_index_s_start = lowest_active_rung_s
                                else:
                                    # The receiver and the supplier
                                    # should receive a kick.
                                    apply_to_i = True
                        # We need to make sure not to double count the
                        # rung pairs for local interactions. Here,
                        # local means that the current components,
                        # domains, tiles and subtiles for the receiver
                        # and supplier are all the same.
                        with unswitch(5):
                            if 𝔹[receiver.name == supplier.name and rank == rank_supplier]:
                                with unswitch(3):
                                    if 𝔹[tile_index_r == tile_index_s]:
                                        with unswitch(1):
                                            if 𝔹[subtile_index_r == subtile_index_s]:
                                                if rung_index_s_start < rung_index_r:
                                                    rung_index_s_start = rung_index_r
                        # Loop over the needed supplier rungs
                        for rung_index_s in range(
                            rung_index_s_start, ℤ[highest_populated_rung_s + 1]
                        ):
                            rung_N_s = rungs_N_s[rung_index_s]
                            if rung_N_s == 0:
                                continue
                            rung_s = subtile_s[rung_index_s]
                            # Flag whether we need to apply the force to
                            # the supplier particles in this rung (if
                            # not, we still apply the force to the
                            # receiver particles).
                            with unswitch(6):
                                if 𝔹[not only_supply]:
                                    apply_to_j = (rung_index_s >= lowest_active_rung_s)
                            # Loop over all particles
                            # in the receiver rung.
                            for rung_particle_index_r in range(rung_N_r):
                                # Get receiver particle index
                                i = rung_r[rung_particle_index_r]
                                # Construct rung_index_i. This is equal
                                # to rung_index_r, except for when a
                                # particle jump to another rung.
                                # In this case, rung_index_i no longer
                                # corresponds to any actual rung,
                                # but it can be used to correctly index
                                # into arrays of time step integrals.
                                if True:  #with unswitch(7):
                                    if 𝔹[receiver.use_rungs]:
                                        if True:  # with unswitch(4):
                                            if subtile_contain_particles_r == 3:
                                                rung_jump_r = rung_jumps_r[i]
                                                if rung_jump_r == 0:
                                                    rung_index_i = rung_index_r
                                                else:
                                                    rung_index_i = rung_index_r + rung_jump_r
                                            else:
                                                rung_index_i = rung_index_r
                                    else:
                                        rung_index_i = rung_index_r
                                # Get coordinates of receiver particle
                                xi = posx_r[i]
                                yi = posy_r[i]
                                zi = posz_r[i]
                                # We need to make sure not to double
                                # count the particle pairs for local
                                # interactions. Here, local means that
                                # the current components, domains,
                                # tiles, subtiles and rungs for the
                                # receiver and supplier are all
                                # the same.
                                rung_particle_index_s_start = 0
                                with unswitch(7):
                                    if 𝔹[receiver.name == supplier.name and rank == rank_supplier]:
                                        with unswitch(5):
                                            if 𝔹[tile_index_r == tile_index_s]:
                                                with unswitch(3):
                                                    if 𝔹[subtile_index_r == subtile_index_s]:
                                                        with unswitch(1):
                                                            if 𝔹[rung_index_r == rung_index_s]:
                                                                rung_particle_index_s_start = (
                                                                    rung_particle_index_r + 1
                                                                )
                                # Loop over the needed particles
                                # in the supplier rung.
                                for rung_particle_index_s in range(
                                    rung_particle_index_s_start, rung_N_s,
                                ):
                                    # Get supplier particle index
                                    j = rung_s[rung_particle_index_s]
                                    # Construct rung_index_j
                                    if True:  # with unswitch(8):
                                        if 𝔹[not only_supply and supplier.use_rungs]:
                                            if True:  # with unswitch(4):
                                                if subtile_contain_particles_s == 3:
                                                    rung_jump_s = rung_jumps_s[j]
                                                    if rung_jump_s == 0:
                                                        rung_index_j = rung_index_s
                                                    else:
                                                        rung_index_j = rung_index_s + rung_jump_s
                                                else:
                                                    rung_index_j = rung_index_s
                                        else:
                                            rung_index_j = rung_index_s
                                    # "Vector" from particle j
                                    # to particle i.
                                    x_ji = xi - posx_s[j]
                                    y_ji = yi - posy_s[j]
                                    z_ji = zi - posz_s[j]
                                    # Yield the needed variables
                                    yield i, j, rung_index_i, rung_index_j, x_ji, y_ji, z_ji, apply_to_i, apply_to_j, particle_particle_t_begin, subtiling_r
# Variables used by the particle_particle function
cython.declare(
    tile_location_r='double[::1]',
    tile_location_r_ptr='double*',
    tile_location_s='double[::1]',
    tile_location_s_ptr='double*',
    tiles_offset='Py_ssize_t[::1]',
    tiles_offset_ptr='Py_ssize_t*',
)
tile_location_r = empty(3, dtype=C2np['double'])
tile_location_s = empty(3, dtype=C2np['double'])
tiles_offset    = empty(3, dtype=C2np['Py_ssize_t'])
tile_location_r_ptr = cython.address(tile_location_r[:])
tile_location_s_ptr = cython.address(tile_location_s[:])
tiles_offset_ptr    = cython.address(tiles_offset[:])

# Generic function implementing particle-mesh interactions
# for both particle and fluid componenets.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    gridsize_global='Py_ssize_t',
    quantity=str,
    force=str,
    method=str,
    potential=str,
    interpolation_order='int',
    interlace='bint',
    differentiate_global='bint',
    differentiation_order='int',
    ᔑdt=dict,
    ᔑdt_key=object,  # str or tuple
    # Locals
    Jᵢ='FluidScalar',
    Jᵢ_ptr='double*',
    any_fluid_receivers='bint',
    any_fluid_suppliers='bint',
    any_particles_receivers='bint',
    any_particles_suppliers='bint',
    buffer_names=dict,
    deconv='double',
    dim='int',
    factor='double',
    fft_factor='double',
    grid_global='double[:, :, ::1]',
    gridshape_local=tuple,
    gridsize_downstream='Py_ssize_t',
    grids_global=dict,
    index='Py_ssize_t',
    k2='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    potential_factor='double',
    receiver='Component',
    receiver_collections=dict,
    receiver_group=list,
    receiver_groups=object,  # collections.defaultdict(list)
    representation=str,
    slab_fluid='double[:, :, ::1]',
    slab_fluid_ptr='double*',
    slab_particles='double[:, :, ::1]',
    slab_particles_ptr='double*',
    slabs=dict,
    submsg=str,
    supplier='Component',
    suppliers_gridsizes_upstream=list,
    suppliers_name=str,
    Δx='double',
    ϱ_ptr='double*',
    𝒫_ptr='double*',
    ᐁᵢgrid_downstream='double[:, :, ::1]',
    ᐁᵢgrid_global='double[:, :, ::1]',
    ᐁᵢgrid_receiver='double[:, :, ::1]',
    ᐁᵢgrid_receiver_ptr='double*',
    returns='void',
)
def particle_mesh(
    receivers, suppliers, gridsize_global, quantity, force, method, potential,
    interpolation_order, interlace, differentiate_global, differentiation_order,
    ᔑdt, ᔑdt_key,
):
    """
    This function will update the momenta of all receiver components due
    to an interaction with the supplier components. The steps taken are
    outlined below.
    - Interpolate the specified quantity of all supplier components onto
      local (component specific) grids. For particle components, this is
      done through particle (e.g. CIC) interpolation, while pixel mixing
      is used for fluids.
    - Pixel mix the local grids onto a common, global grid of grid
      size gridsize_global.
    - Convert the global grid to a potential:
      - Transform the grid to Fourier space.
      - Multiply the grid values by the specified potential function.
      - While in Fourier space, perform deconvolutions needed due to
        particle interpolations. This results in two versions of the
        global potential grid; one applicable to particle components and
        one applicable to fluid components, both of which contain the
        total potential from all suppliers.
      - Transform back to real space.
    - For each dimension, differentiate the global potentials to get the
      (particle and fluid) force.
    - Pixel mix the global force grid back to the local grids.
    - Interpolate the local grids onto the components, applying the
      force. The force application uses the prescription
      Δmom = -component.mass*∂ⁱφ*ᔑdt[ᔑdt_key].
    """
    suppliers_name = '{{{}}}'.format(', '.join([supplier.name for supplier in suppliers]))
    if len(suppliers) == 1:
        suppliers_name = suppliers_name.strip('{}')
    masterprint(
        f'Constructing potential of grid size {gridsize_global} due to {suppliers_name} ...'
    )
    # Flags specifying the existence of components
    any_particles_receivers = any([receiver.representation == 'particles' for receiver in receivers])
    any_fluid_receivers     = any([receiver.representation == 'fluid'     for receiver in receivers])
    any_particles_suppliers = any([supplier.representation == 'particles' for supplier in suppliers])
    any_fluid_suppliers     = any([supplier.representation == 'fluid'     for supplier in suppliers])
    if not any_particles_suppliers:
        interlace = False
    # Interpolate suppliers onto global grids by first interpolating
    # them onto individual upstream grids and then pixel mix these onto
    # global grids of size gridsize_global. Separate global grids will
    # be used for particle and fluid components.
    suppliers_gridsizes_upstream = [
        supplier.potential_gridsizes[force][method].upstream
        for supplier in suppliers
    ]
    grids_global = interpolate_upstream(
        suppliers, suppliers_gridsizes_upstream, gridsize_global, quantity,
        interpolation_order, interlace, ᔑdt,
        do_ghost_communication=False,
    )
    # Slab decompose the global grids
    slabs = slab_decompose(grids_global, prepare_fft=True)
    # Do a forward in-place Fourier transform of the slabs
    fft(slabs, 'forward')
    # Currently the 'particles' and 'fluid' grids/slabs hold data from
    # the particles and fluid suppliers, respectively. In the end these
    # will be transformed such that they both hold the total potential
    # from all suppliers, their difference being the amount of
    # deconvolution applied. The 'particles' grid then supplies the
    # force to particle receivers while the 'fluid' grid supplies the
    # force to fluid receivers. In the case where we have a
    # particles/fluid receiver but no supplier of the same
    # representation, no corresponding grid/slab currently exist.
    # Allocate them.
    gridshape_local = ()
    if any_particles_receivers and not any_particles_suppliers:
        if not gridshape_local:
            gridshape_local = get_gridshape_local(gridsize_global)
        grids_global['particles'] = get_buffer(gridshape_local, 'global_grid_particles')
        slabs['particles'] = get_fftw_slab(gridsize_global, 'slab_particles')
    if any_fluid_receivers and not any_fluid_suppliers:
        if not gridshape_local:
            gridshape_local = get_gridshape_local(gridsize_global)
        grids_global['fluid'] = get_buffer(gridshape_local, 'global_grid_fluid')
        slabs['fluid'] = get_fftw_slab(gridsize_global, 'slab_fluid')
    # Store the slabs as separate variables
    slab_particles = slabs.get('particles')
    if slab_particles is not None:
        slab_particles_ptr = cython.address(slab_particles[:, :, :])
    slab_fluid = slabs.get('fluid')
    if slab_fluid is not None:
        slab_fluid_ptr = cython.address(slab_fluid[:, :, :])
    # Multiplicative factor needed after a
    # forward and a backward Fourier transformation.
    fft_factor = float(gridsize_global)**(-3)
    # Loop over the complex numbers in the slabs
    deconv_order = interpolation_order*(any_particles_suppliers or any_particles_receivers)
    for index, ki, kj, kk, deconv in slab_fourier_loop(
        slabs,
        deconv_order=deconv_order,
        do_interlacing=interlace,
    ):
        k2 = ℤ[ℤ[kj**2] + ki**2] + kk**2
        # The potential factor, for converting the slab values
        # to the desired potential. Note that we further attach
        # the FFT normalisation factor to this.
        with unswitch(3):
            # The physical squared length of the wave
            # vector is given by |k|² = (2π/boxsize)**2*k2.
            if 𝔹[potential == 'gravity']:
                # The Poisson equation, the factor of which is
                #   -4πG/|k|².
                potential_factor = ℝ[fft_factor*ℝ[-boxsize**2*G_Newton/π]]/k2
            elif 𝔹[potential == 'gravity long-range']:
                # The Poisson equation with a Gaussian
                # cutoff, resulting in the factor
                #   -4πG/|k|² * exp(-rₛ²*|k|²).
                potential_factor = (
                    ℝ[fft_factor*ℝ[-boxsize**2*G_Newton/π]]/k2
                    *exp(k2*ℝ[-(2*π/boxsize*shortrange_params['gravity']['scale'])**2])
                )
            else:
                abort(f'potential "{potential}" not implemented in particle_mesh()')
                potential_factor = 1  # To satisfy the compiler
        # Combine the fluid and particles slab values into the
        # total potential. The result is {re, im} and
        # corresponds to the fluid potential, from which the
        # particles potential is obtained through a single
        # deconvolution.
        with unswitch(3):
            if 𝔹[any_particles_suppliers and any_fluid_suppliers]:
                re = potential_factor*(
                    deconv*slab_particles_ptr[index    ] + slab_fluid_ptr[index    ]
                )
                im = potential_factor*(
                    deconv*slab_particles_ptr[index + 1] + slab_fluid_ptr[index + 1]
                )
            elif any_particles_suppliers:
                factor = potential_factor*deconv
                re = factor*slab_particles_ptr[index    ]
                im = factor*slab_particles_ptr[index + 1]
            else:  # any_fluid_suppliers
                re = potential_factor*slab_fluid_ptr[index    ]
                im = potential_factor*slab_fluid_ptr[index + 1]
        # Insert the final potential values into the slabs.
        # The unswitched block below is somewhat repetitive,
        # but it leads to fewer source lines after unswitching compared
        # to having individual unswitching for any_particles_receivers
        # and any_fluid_receivers.
        with unswitch(3):
            if 𝔹[any_particles_receivers and any_fluid_receivers]:
                slab_particles_ptr[index    ] = deconv*re
                slab_particles_ptr[index + 1] = deconv*im
                slab_fluid_ptr    [index    ] =        re
                slab_fluid_ptr    [index + 1] =        im
            elif any_particles_receivers:
                slab_particles_ptr[index    ] = deconv*re
                slab_particles_ptr[index + 1] = deconv*im
            else:  # any_fluid_receivers
                slab_fluid_ptr    [index    ] =        re
                slab_fluid_ptr    [index + 1] =        im
    # The shifted particles slabs and global grid are never needed
    # from this point on, and so we remove these. We further remove the
    # slabs and global grids for which we have no receivers of the same
    # representation.
    grids_global.pop('particles_shifted', None)
    slabs       .pop('particles_shifted', None)
    if not any_particles_receivers:
        grids_global.pop('particles', None)
        slabs       .pop('particles', None)
    if not any_fluid_receivers:
        grids_global.pop('fluid', None)
        slabs       .pop('fluid', None)
    if not slabs or not grids_global:
        abort(
            'Something went wrong in the particle_mesh() function, '
            'as it appears that neither particles nor fluids should receive the force '
            'due to the potential.'
        )
    # Ensure nullified Nyquist planes and origin
    nullify_modes(slabs, 'nyquist, dc')
    # Fourier transform the slabs back to coordinate space
    fft(slabs, 'backward')
    # Domain-decompose the slabs
    domain_decompose(slabs, grids_global, do_ghost_communication=differentiate_global)
    masterprint('done')
    # Get and apply the force from the global potential grids
    buffer_names = {
        'differentiate'    : 0,
        'interpolate'      : 1,
        'interpolate_fluid': 2,
    }
    if differentiate_global:
        # We should first differentiate the global potential grids to
        # obtain global force grids, then pixel mix these to downstream
        # force grids and apply them.
        # Group receivers according to their representation and
        # downstream grid size.
        receiver_collections = {
            representation: collections.defaultdict(list)
            for representation in grids_global.keys()
        }
        for receiver in receivers:
            gridsize_downstream = (
                receiver.potential_gridsizes[force][method].downstream
            )
            receiver_group = (
                receiver_collections[receiver.representation][gridsize_downstream]
            )
            receiver_group.append(receiver)
        # Physical grid spacing of global potential grids
        Δx = boxsize/gridsize_global
        # Differentiate the global potential to get the global force
        # field, pixel mix to downstream forces and apply them, for each
        # representation, direction and receiver.
        submsg = (r'({representation}, {xyz})' if len(grids_global) == 2 else r'({xyz})')
        for representation, grid_global in grids_global.items():
            receiver_groups = receiver_collections[representation]
            for dim in range(3):
                masterprint(
                    f'Obtaining and applying the {submsg} force ...'
                    .format(representation=representation, xyz='xyz'[dim])
                )
                # Differentiate the global potential grid
                # along the dim'th dimension.
                ᐁᵢgrid_global = diff_domaingrid(
                    grid_global, dim, differentiation_order,
                    Δx, ℤ[buffer_names['differentiate']],
                    do_ghost_communication=False,
                )
                for gridsize_downstream, receiver_group in receiver_groups.items():
                    # Pixel mix global force downstream
                    # to get local force.
                    ᐁᵢgrid_downstream = interpolate_grid_to_grid(
                        ᐁᵢgrid_global,
                        ℤ[buffer_names['interpolate']],
                        gridsize_downstream,
                    )
                    # For particle interpolation we need the
                    # ghost points to be properly set.
                    with unswitch(2):
                        if 𝔹[representation == 'particles']:
                            communicate_ghosts(ᐁᵢgrid_downstream, '=')
                    # Apply force
                    for receiver in receiver_group:
                        masterprint(f'Applying force to {receiver.name} ...')
                        with unswitch(3):
                            if 𝔹[isinstance(ᔑdt_key, tuple)]:
                                ᔑdt_key = (ᔑdt_key[0], receiver.name)
                        with unswitch(3):
                            if 𝔹[representation == 'particles']:
                                # Update the dim'th momentum component
                                # of all particles through interpolation
                                # in ᐁᵢgrid_downstream. To convert from
                                # force to momentum change we should
                                # multiply by -mass*Δt (minus as the
                                # force is the negative gradient of the
                                # potential), where Δt = ᔑdt['1']. Here
                                # this integral over the time step is
                                # generalised and supplied
                                # by the caller.
                                interpolate_domaingrid_to_particles(
                                    ᐁᵢgrid_downstream, receiver, 'mom', dim, interpolation_order,
                                    receiver.mass*ℝ[-ᔑdt[ᔑdt_key]],
                                )
                            else:  # representation == 'fluid'
                                # Make sure that the force grid has
                                # identical grid size as the receiver.
                                # After this we may simply add the force
                                # grid directly to the receiver grids,
                                # without further interpolation.
                                ᐁᵢgrid_receiver = interpolate_grid_to_grid(
                                    ᐁᵢgrid_downstream,
                                    ℤ[buffer_names['interpolate_fluid']],
                                    receiver.gridsize,
                                )
                                ᐁᵢgrid_receiver_ptr = cython.address(ᐁᵢgrid_receiver[:, :, :])
                                # The source term has the form
                                # ΔJ ∝ -(ϱ + c⁻²𝒫)*ᐁφ.
                                # The proportionality factor above is
                                # something liḱe Δt = ᔑdt['1']. Here
                                # this integral over the time step is
                                # generalised and supplied by the
                                # caller.
                                Jᵢ = receiver.J[dim]
                                Jᵢ_ptr = Jᵢ.grid
                                ϱ_ptr = receiver.ϱ.grid
                                𝒫_ptr = receiver.𝒫.grid
                                for index in range(receiver.size):
                                    Jᵢ_ptr[index] += ℝ[-ᔑdt[ᔑdt_key]]*(
                                        ϱ_ptr[index] + ℝ[light_speed**(-2)]*𝒫_ptr[index]
                                    )*ᐁᵢgrid_receiver_ptr[index]
                                # Though the above loop includes ghost
                                # points, these are not properly set in
                                # ᐁᵢgrid_receiver. Correctly set the
                                # ghost points in the momentum grid.
                                communicate_ghosts(Jᵢ.grid_mv, '=')
                        masterprint('done')
                masterprint('done')
    else:
        # We should first pixel mix the global potentials to downstream
        # potentials, then differentiate these to get downstream force
        # grids and then apply them.
        abort(
            f'It is currently only possible to obtain the force directly from the '
            f'global potential. You should set '
            f'force_from_global_potentials["{force}"]["{method}"] = True.'
        )

# Function implementing progress messages used for the short-range
# kicks intertwined with drift operations.
@cython.pheader(
    # Arguments
    force=str,
    method=str,
    receivers=list,
    extra_message=str,
    # Locals
    component='Component',
    returns=str,
)
def shortrange_progress_message(force, method, receivers, extra_message=' (short-range only)'):
    # Lookup appropriate form of the name of the force
    force = interactions_registered[force].conjugated_name
    # Print the progress message
    if method == 'p3m':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the P³M method{extra_message}'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the P³M method{extra_message}'
                .format(', '.join([component.name for component in receivers]))
            )
    elif method == 'pp':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the PP method'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the PP method'
                .format(', '.join([component.name for component in receivers]))
            )
    elif method == 'ppnonperiodic':
        if len(receivers) == 1:
            return f'{force} interaction for {receivers[0].name} via the non-periodic PP method'
        else:
            return (
                f'{force} interaction for {{{{{{}}}}}} via the non-periodic PP method'
                .format(', '.join([component.name for component in receivers]))
            )
    else:
        abort(f'The method "{method}" is unknown to shortrange_progress_message()')

# Function that given lists of receiver and supplier components of a
# one-way interaction removes any components from the supplier list that
# are also present in the receiver list.
def oneway_force(receivers, suppliers):
    return [component for component in suppliers if component not in receivers]

# Function which constructs a list of interactions from a list of
# components. The list of interactions store information about which
# components interact with one another, via what force and method.
def find_interactions(components, interaction_type='any', instantaneous='both'):
    """You may specify an interaction_type to only get
    specific interactions. The options are:
    - interaction_type == 'any':
      Include every interaction.
    - interaction_type == 'long-range':
      Include long-range interactions only, i.e. ones with a method of
      either PM and P³M. Note that P³M interactions will also be
      returned for interaction_type == 'short-range'.
    - interaction_type == 'short-range':
      Include short-range interactions only, i.e. any other than PM.
      Note that P³M interactions will also be returned
      for interaction_type == 'short-range'.
    Furthermore you may specify instantaneous to filter out interactions
    that are (not) instantaneous:
    - instantaneous == 'both':
      Include both instantaneous and non-instantaneous interactions.
    - instantaneous == True:
      Include only non-instantaneous interactions.
    - instantaneous == False:
      Include only instantaneous interactions.
    """
    # Use cached result
    key = (interaction_type, instantaneous, tuple(components))
    interactions_list = interactions_lists.get(key)
    if interactions_list is not None:
        return interactions_list
    # Find all (force, method) pairs in use. Store these as a (default)
    # dict mapping forces to lists of methods.
    forces_in_use = collections.defaultdict(set)
    for component in components:
        for force, method in component.forces.items():
            forces_in_use[force].add(method)
    # Check that all forces and methods assigned
    # to the components are implemented.
    for force, methods in forces_in_use.items():
        interaction_info = interactions_registered.get(force)
        if interaction_info is None:
            abort(f'Force "{force}" is not implemented')
        for method in interaction_info.methods:
            if not method:
                # Should never happen
                abort(f'Falsy method "{method}" registered for force "{force}')
            if method not in methods_implemented:
                abort(f'Force method "{method}" not recognized')
        for method in methods:
            if not method:
                # When the method is set to an empty string it signifies
                # that this method should be used as a supplier for the
                # given force, but not receive the force itself.
                continue
            if method not in methods_implemented:
                abort(f'Force method "{method}" not recognized')
            if method not in interaction_info.methods:
                if len(interaction_info.methods) == 1:
                    abort(
                        f'Method "{method}" for force "{force}" is not implemented. '
                        f'Did you mean "{interaction_info.methods[0]}"?'
                    )
                else:
                    abort(
                        f'Method "{method}" for force "{force}" is not implemented. '
                        f'Did you mean one of {interaction_info.methods}?'
                    )
    # Construct the interactions_list with (named) 4-tuples
    # in the format (force, method, receivers, suppliers),
    # where receivers is a list of all components which interact
    # via the force and should therefore receive momentum updates
    # computed via this force and the method given as the
    # second element. In the simple case where all components
    # interacting under some force using the same method, the suppliers
    # list holds the same components as the receivers list. When the
    # same force should be applied to several components using
    # different methods, the suppliers list still holds all components
    # as before, while the receivers list is limited to just those
    # components that should receive the force using the
    # specified method. Note that the receivers do not contribute to the
    # force unless they are also present in the suppliers list.
    interactions_list = []
    for force, interaction_info in interactions_registered.items():
        methods = interaction_info.methods
        for method in methods:
            if method not in forces_in_use.get(force, []):
                continue
            # Find all receiver and supplier components
            # for this (force, method) pair.
            receivers = []
            suppliers = []
            for component in components:
                if force in component.forces:
                    suppliers.append(component)
                    if component.forces[force] == method:
                        receivers.append(component)
            # Store the 4-tuple in the interactions_list
            interactions_list.append(Interaction(force, method, receivers, suppliers))
    # Cleanup the list of interactions
    def cleanup():
        nonlocal interactions_list
        # If fluid components are present as suppliers for interactions
        # using a method different from PM, remove them from the
        # suppliers list and create a new PM interaction instead.
        for i, interaction in enumerate(interactions_list):
            if interaction.method == 'pm':
                continue
            for component in interaction.suppliers:
                if component.representation == 'fluid':
                    interaction.suppliers.remove(component)
                    interactions_list.insert(
                        i + 1,
                        Interaction(
                            interaction.force, 'pm', interaction.receivers.copy(), [component],
                        )
                    )
                    return True
        # Remove interactions with no suppliers or no receivers
        interactions_list = [interaction for interaction in interactions_list
            if interaction.receivers and interaction.suppliers]
        # Merge interactions of identical force, method and receivers,
        # but different suppliers; or identical force,
        # method and suppliers but different receivers.
        for     i, interaction_i in enumerate(interactions_list):
            for j, interaction_j in enumerate(interactions_list[i+1:], i+1):
                if interaction_i.force != interaction_j.force:
                    continue
                if interaction_i.method != interaction_j.method:
                    continue
                if (
                        set(interaction_i.receivers) == set(interaction_j.receivers)
                    and set(interaction_i.suppliers) != set(interaction_j.suppliers)
                ):
                    for supplier in interaction_j.suppliers:
                        if supplier not in interaction_i.suppliers:
                            interaction_i.suppliers.insert(0, supplier)
                    interactions_list.pop(j)
                    return True
                if (
                        set(interaction_i.receivers) != set(interaction_j.receivers)
                    and set(interaction_i.suppliers) == set(interaction_j.suppliers)
                ):
                    for receiver in interaction_j.receivers:
                        if receiver not in interaction_i.receivers:
                            interaction_i.receivers.insert(0, receiver)
                    interactions_list.pop(j)
                    return True
    while cleanup():
        pass
    # In the case that only long-/short-range interactions should be
    # considered, remove the unwanted interactions.
    if 'long' in interaction_type:
        for interaction in interactions_list:
            if interaction.method not in {'pm', 'p3m'}:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'short' in interaction_type:
        for interaction in interactions_list:
            if interaction.method == 'pm':
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'any' not in interaction_type:
        abort(f'find_interactions(): Unknown interaction_type "{interaction_type}"')
    # In the case that only (non-)instantaneous interactions should be
    # considered, remove the unwanted interactions.
    if 'True' in str(instantaneous):
        for interaction in interactions_list:
            if not interactions_registered[interaction.force].instantaneous:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'False' in str(instantaneous):
        for interaction in interactions_list:
            if interactions_registered[interaction.force].instantaneous:
                interaction.receivers[:] = []
        while cleanup():
            pass
    elif 'both' not in str(instantaneous):
        abort(f'find_interactions(): Unknown instantaneous value "{instantaneous}"')
    # Cache the result and return it
    interactions_lists[key] = interactions_list
    return interactions_list
# Global dict of interaction lists populated by the above function
cython.declare(interactions_lists=dict)
interactions_lists = {}
# Create the Interaction type used in the above function
Interaction = collections.namedtuple(
    'Interaction', ('force', 'method', 'receivers', 'suppliers')
)

# Function for registrering interactions
def register(
    force, methods, conjugated_name=None,
    *,
    dependent=('pos', ), affected=('mom', ),
    deterministic=True, instantaneous=False,
):
    """Every implemented interaction should be registered by a call to
    this function. The order in which interactions are registered will
    be the order in which they are carried out, with the exeption that
    short-range instantaneous interactions will be carried out before
    short-range non-instantaneous interactions.
    """
    # Canonicalize force and method names
    def canonicalize(s):
        s = s.lower()
        for char in ' _-^()':
            s = s.replace(char, '')
        for n in range(10):
            s = s.replace(unicode_superscript(str(n)), str(n))
        return s
    force = canonicalize(force)
    methods = [canonicalize(method) for method in any2list(methods)]
    # If no "conjugated" version of the force name is given,
    # set it equal to the normal name of the force.
    if conjugated_name is None:
        conjugated_name = force
    # Store the information globally as an InteractionInfo instance
    interactions_registered[force] = InteractionInfo(
        force, methods, conjugated_name,
        any2list(dependent), any2list(affected),
        deterministic, instantaneous,
    )
# Global dict of interaction infos populated by the above function
cython.declare(interactions_registered=dict)
interactions_registered = {}
# Create the InteractionInfo type used in the above function
InteractionInfo = collections.namedtuple(
    'InteractionInfo',
    (
        'force', 'methods', 'conjugated_name',
        'dependent', 'affected',
        'deterministic', 'instantaneous',
    ),
)

# Function which looks up quantities defined between pairs of
# components within a passed dict.
@cython.header(
    # Arguments
    receivers=list,
    suppliers=list,
    select_dict=dict,
    name=str,
    # Locals
    key=tuple,
    pair=set,
    pairs=list,
    quantities=dict,
    quantity=object,
    quantity_r=object,
    quantity_s=object,
    receiver='Component',
    supplier='Component',
    returns=dict,
)
def get_pairwise_quantities(receivers, suppliers, select_dict, name=''):
    """The "name" argument is only used in relation with the caching.
    """
    # Attempt lookup in cache
    if name:
        key = (
            name,
            frozenset([receiver.name for receiver in receivers]),
            frozenset([supplier.name for supplier in suppliers]),
        )
        quantities = pairwise_quantities.get(key)
        if quantities is not None:
            return quantities
    # Result not present in cache. Do the lookups.
    quantities = {}
    pairs = []
    for receiver in receivers:
        for supplier in suppliers:
            pair = {receiver, supplier}
            if pair in pairs:
                continue
            pairs.append(pair)
            # Look up the quantity for this {receiver, supplier} pair
            quantity = is_selected((receiver, supplier), select_dict)
            if quantity is None:
                if receiver.name == supplier.name:
                    quantity = is_selected(receiver, select_dict)
                else:
                    quantity_r = is_selected(receiver, select_dict)
                    quantity_s = is_selected(supplier, select_dict)
                    if quantity_r == quantity_s:
                        quantity = quantity_r
            if quantity is None:
                if name:
                    abort(
                        f'get_pairwise_quantities(): No pairwise quantity "{name}" '
                        f'for {{{receiver.name}, {supplier.name}}} specified in the passed dict.'
                    )
                else:
                    abort(
                        f'get_pairwise_quantities(): No pairwise quantity '
                        f'for {{{receiver.name}, {supplier.name}}} specified in the passed dict.'
                    )
            # Store the found quantity symmetrically
            # with respect to the receiver and supplier.
            quantities[receiver.name, supplier.name] = quantity
            quantities[supplier.name, receiver.name] = quantity
    # Save the found quantities to the cache
    if name:
        pairwise_quantities[key] = quantities
    return quantities
# Cache used by the above function
cython.declare(pairwise_quantities=dict)
pairwise_quantities = {}

# Function for looking up various information needed for potential
# (i.e. particle mesh) interactions.
@cython.header(
    # Arguments
    force=str,
    method=str,
    receivers=list,
    suppliers=list,
    # Locals
    differentiate_global='bint',
    differentiation_order='int',
    gridsize='Py_ssize_t',
    gridsizes=set,
    interlace='bint',
    interpolation_order='int',
    key=tuple,
    potential_specs=object,  # PotentialForceInfo
    receiver='Component',
    supplier='Component',
    returns=object,  # PotentialForceInfo
)
def get_potential_specs(force, method, receivers, suppliers):
    # Cache lookup
    key = (force, method, tuple(receivers), tuple(suppliers))
    potential_specs = potential_specs_cache.get(key)
    if potential_specs is not None:
        return potential_specs
    # Look up the global potential grid size
    gridsize = 𝕆[potential_gridsizes['global']][force][method]
    if gridsize == -1:
        # No global grid size specified. If all supplier upstream
        # and receiver downstream grid sizes are equal, use this for
        # the global grid size.
        gridsizes = (
            {
                supplier.potential_gridsizes[force][method].upstream
                for supplier in suppliers
            } | {
                receiver.potential_gridsizes[force][method].downstream
                for receiver in receivers
            }
        )
        if len(gridsizes) != 1:
            abort(
                f'No global potential grid size specified for force "{force}" with '
                f'method "{method}". As multiple upstream and/or downstream grid sizes '
                f'are in use, the global grid size could not be set automatically.'
            )
        gridsize = gridsizes.pop()
    # Fetch interpolation, interlacing
    # and differentiation specifications.
    interpolation_order   = force_interpolations        [force][method]
    interlace             = force_interlacings          [force][method]
    differentiate_global  = force_from_global_potentials[force][method]
    differentiation_order = force_differentiations      [force][method]
    # Cache and return
    potential_specs = PotentialForceInfo(
        gridsize, interpolation_order, interlace,
        differentiate_global, differentiation_order,
    )
    potential_specs_cache[key] = potential_specs
    return potential_specs
# Cache and type used by the above function
cython.declare(potential_specs_cache=dict)
potential_specs_cache = {}
PotentialForceInfo = collections.namedtuple(
    'PotentialForceInfo',
    (
        'gridsize', 'interpolation_order', 'interlace',
        'differentiate_global', 'differentiation_order',
    ),
)



#########################################
# Implement specific interactions below #
#########################################

# Gavity
cimport('from gravity import *')
register('gravity', ['ppnonperiodic', 'pp', 'p3m', 'pm'], 'gravitational')
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    # Locals
    extra_message=str,
    force=str,
    potential=str,
    potential_specs=object,  # PotentialForceInfo
    quantity=str,
    ᔑdt_key=tuple,
)
def gravity(method, receivers, suppliers, ᔑdt, interaction_type, printout):
    force = 'gravity'
    # Set up variables used by potential/grid (PM and P³M) methods
    if method in {'pm', 'p3m'}:
        potential_specs = get_potential_specs(force, method, receivers, suppliers)
        # The gravitational potential is given by the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ,
        # summed over all suppliers.
        # The component dependent quantity is then a²ρ.
        # In the case of the long-range part only (for use with P³M),
        # though the expression for the potential is altered, we still
        # have φ ∝ a²ρ.
        quantity = 'a²ρ'
        # In the fluid description, the gravitational source term is
        # ∂ₜJⁱ = ⋯ -a**(-3*w_eff)*(ϱ + c⁻²𝒫)*∂ⁱφ
        # and so a**(-3*w_eff) should be integrated over the time step
        # to get ΔJⁱ. In the particle description, the gravitational
        # source term is
        # ∂ₜmomⁱ = -mass*∂ⁱφ.
        # In the general case of a changing mass, the current mass is
        # given by mass*a**(-3*w_eff), and so again, a**(-3*w_eff)
        # shoud be integrated over the time step
        # in order to obtain Δmomⁱ.
        ᔑdt_key = ('a**(-3*w_eff)', 'component')
    # Compute gravity via one of the following methods
    if method == 'pm':
        # The particle-mesh method
        if printout:
            masterprint(
                f'Executing gravitational interaction for {receivers[0].name} '
                f'via the PM method ...'
                if len(receivers) == 1 else (
                    'Executing gravitational interaction for {{{}}} via the PM method ...'
                    .format(', '.join([component.name for component in receivers]))
                )
            )
        potential = 'gravity'
        particle_mesh(
            receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
            potential_specs.interpolation_order, potential_specs.interlace,
            potential_specs.differentiate_global, potential_specs.differentiation_order,
            ᔑdt, ᔑdt_key,
        )
        if printout:
            masterprint('done')
    elif method == 'p3m':
        # The particle-particle-mesh method
        if printout:
            extra_message = ''
            if 𝔹['long' in interaction_type]:
                extra_message = ' (long-range only)'
            elif 𝔹['short' in interaction_type]:
                extra_message = ' (short-range only)'
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers, extra_message),
                '...',
            )
        # The long-range PM part
        if 𝔹['any' in interaction_type] or 𝔹['long' in interaction_type]:
            potential = 'gravity long-range'
            particle_mesh(
                receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
                potential_specs.interpolation_order, potential_specs.interlace,
                potential_specs.differentiate_global, potential_specs.differentiation_order,
                ᔑdt, ᔑdt_key,
            )
        # The short-range PP part
        if 𝔹['any' in interaction_type] or 𝔹['short' in interaction_type]:
            tabulate_shortrange_gravity()
            component_component(
                force, receivers, suppliers, gravity_pairwise_shortrange, ᔑdt,
                pairing_level='tile',
            )
        if printout:
            masterprint('done')
    elif method == 'pp':
        # The particle-particle method with Ewald-periodicity
        if printout:
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers),
                '...',
            )
        get_ewald_grid()
        component_component(
            force, receivers, suppliers, gravity_pairwise, ᔑdt,
            pairing_level='domain',
        )
        if printout:
            masterprint('done')
    elif method == 'ppnonperiodic':
        # The non-periodic particle-particle method
        if printout:
            masterprint(
                'Executing',
                shortrange_progress_message(force, method, receivers),
                '...',
            )
        component_component(
            force, receivers, suppliers, gravity_pairwise_nonperiodic, ᔑdt,
            pairing_level='domain',
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'gravity() was called with the "{method}" method')

# The lapse force
register('lapse', 'pm')
@cython.pheader(
    # Arguments
    method=str,
    receivers=list,
    suppliers=list,
    ᔑdt=dict,
    interaction_type=str,
    printout='bint',
    # Locals
    force=str,
    potential=str,
    potential_specs=object,  # PotentialForceInfo
    quantity=str,
    ᔑdt_key=tuple,
)
def lapse(method, receivers, suppliers, ᔑdt, interaction_type, printout):
    force = 'lapse'
    # While the receivers list stores the correct components,
    # the suppliers store the lapse component as well as all the
    # components also present as receivers. As the lapse force should be
    # supplied solely from the lapse component, we must remove these
    # additional components.
    suppliers = oneway_force(receivers, suppliers)
    if len(suppliers) == 0:
        abort('The lapse() function got no suppliers, but expected a lapse component.')
    elif len(suppliers) > 1:
        abort(
            f'The lapse() function got the following suppliers: {suppliers}, '
            f'but expected only a lapse component.'
        )
    # For the lapse force, only the PM method is implemented
    if method == 'pm':
        if printout:
            masterprint(
                f'Executing lapse interaction for {receivers[0].name} via the PM method ...'
                if len(receivers) == 1 else (
                    'Executing lapse interaction for {{{}}} via the PM method ...'
                    .format(', '.join([component.name for component in receivers]))
                )
            )
        # Get interaction specifications
        potential_specs = get_potential_specs(force, method, receivers, suppliers)
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, it obeys the Poisson equation
        # ∇²φ = 4πGa²ρ = 4πGa**(-3*w_eff - 1)ϱ,
        # with φ the lapse potential and ρ, ϱ and w_eff belonging to the
        # fictitious lapse species.
        quantity = 'a²ρ'
        potential = 'gravity'
        # As the lapse potential is implemented exactly analogous to the
        # gravitational potential, the momentum updates are again
        # proportional to a**(-3*w_eff) integrated over the time step
        # (see the gravity function for a more detailed explanation).
        # The realized lapse potential is the common lapse potential,
        # independent of the component in question which is to receive
        # momentum updates. The actual lapse potential needed for a
        # given component is obtained by multiplying the common lapse
        # potential by Γ/H, where Γ is the decay rate of the component
        # and H is the Hubble parameter. As these are time dependent,
        # the full time step integral is then a**(-3*w_eff)*Γ/H.
        ᔑdt_key = ('a**(-3*w_eff)*Γ/H', 'component')
        # Execute the lapse interaction
        particle_mesh(
            receivers, suppliers, potential_specs.gridsize, quantity, force, method, potential,
            potential_specs.interpolation_order, potential_specs.interlace,
            potential_specs.differentiate_global, potential_specs.differentiation_order,
            ᔑdt, ᔑdt_key,
        )
        if printout:
            masterprint('done')
    elif master:
        abort(f'lapse() was called with the "{method}" method')
