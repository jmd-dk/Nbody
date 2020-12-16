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



# Function for fairly partitioning data among the processes
@cython.header(# Arguments
               size='Py_ssize_t',
               # Locals
               size_local='Py_ssize_t',
               start_local='Py_ssize_t',
               rank_transition='Py_ssize_t',
               returns=tuple,
               )
def partition(size):
    """This function takes in the size (nr. of elements) of an array
    and partitions it fairly among the processes.
    Both the starting index and size of the local part are returned.
    If the given size cannot be divided evenly, one additional element
    will be given to the higher ranks.
    """
    # Size and starting index of local part of the data
    size_local = size//nprocs
    start_local = rank*size_local
    # Lowest rank which receives one extra data point
    rank_transition = nprocs + size_local*nprocs - size
    # Correct local size and starting index for processes receiving
    # the extra data point.
    if rank >= rank_transition:
        size_local += 1
        start_local += rank - rank_transition
    return start_local, size_local

# Function for communicating sizes of receive buffers
@cython.header(# Arguments
               N_send='Py_ssize_t[::1]',
               # Locals
               ID_recv='int',
               ID_send='int',
               N_recv='Py_ssize_t[::1]',
               N_recv_max='Py_ssize_t',
               j='int',
               same_N_send='bint',
               max_bfore_rank='Py_ssize_t',
               max_after_rank='Py_ssize_t',
               returns='Py_ssize_t[::1]',
               )
def find_N_recv(N_send):
    """Given the size of arrays to send, N_send, which itself has a
    length of either 1 (same data size to send to every process) or
    n_procs (individual data sizes to send to the processes), this
    function communicates this to all processes, so that everyone knows
    how much to receive from every process. The entrance number rank is
    unused (the process do not send to itself). The maximum number to
    receive is useful when allocating receive buffers, so this number is
    stored in this otherwise unused entrance.
    """
    N_recv = empty(nprocs, dtype=C2np['Py_ssize_t'])
    # Check whether N_send is the same for each process to send to
    same_N_send = (N_send.size == 1)
    # If N_send is the same for each process, an Allgather will suffice
    if same_N_send:
        Allgather(N_send, N_recv)
        # Store the max N_recv_in the unused entrance in N_recv
        max_bfore_rank = 0 if rank == 0 else max(N_recv[:rank])
        max_after_rank = 0 if rank == nprocs - 1 else max(N_recv[(rank + 1):])
        N_recv[rank] = (max_bfore_rank if max_bfore_rank > max_after_rank
                        else max_after_rank)
        return N_recv
    # Find out how many particles will be received from each process
    N_recv_max = 0
    for j in range(1, nprocs):
        # Process ranks to send/receive to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        # Send and receive nr of particles to be exchanged
        N_recv[ID_recv] = sendrecv(N_send[ID_send],
                                   dest=ID_send, source=ID_recv)
        if N_recv[ID_recv] > N_recv_max:
            N_recv_max = N_recv[ID_recv]
    # Store N_recv_max in the unused entrance in N_recv
    N_recv[rank] = N_recv_max
    return N_recv

# This function examines every particle of the supplied component and
# communicates them to the process governing the domain in which the
# particle is located.
@cython.header(
    # Arguments
    component='Component',
    reset_buffers='bint',
    # Locals
    ID_recv='int',
    ID_send='int',
    N_local='Py_ssize_t',
    N_needed='Py_ssize_t',
    N_recv='Py_ssize_t[::1]',
    N_recv_j='Py_ssize_t',
    N_recv_max='Py_ssize_t',
    N_recv_tot='Py_ssize_t',
    N_send_j='Py_ssize_t',
    N_send_max='Py_ssize_t',
    N_send_tot='Py_ssize_t',
    N_send_tot_global='Py_ssize_t',
    buffer_name=object,  # int or str
    holes_filled='Py_ssize_t',
    i='Py_ssize_t',
    index_recv_j='Py_ssize_t',
    indices_send_j='Py_ssize_t*',
    j='int',
    k='Py_ssize_t',
    k_start='Py_ssize_t',
    momx='double*',
    momx_mv='double[::1]',
    momy='double*',
    momy_mv='double[::1]',
    momz='double*',
    momz_mv='double[::1]',
    owner='int',
    posx='double*',
    posx_mv='double[::1]',
    posy='double*',
    posy_mv='double[::1]',
    posz='double*',
    posz_mv='double[::1]',
    rung_index='signed char',
    rung_indices='signed char*',
    rung_indices_buf='signed char[::1]',
    rung_indices_jumped='signed char*',
    rung_indices_mv='signed char[::1]',
    rungs_N='Py_ssize_t*',
    sendbuf_mv='double[::1]',
    Δmemory='Py_ssize_t',
    Δmomx='double*',
    Δmomx_mv='double[::1]',
    Δmomy='double*',
    Δmomy_mv='double[::1]',
    Δmomz='double*',
    Δmomz_mv='double[::1]',
    returns='void',
)
def exchange(component, reset_buffers=False):
    """This function will do an exchange of particles between processes,
    so that every particle resides on the process in charge of the
    domain where the particle is located. The variable indices_send
    holds arrays of indices of particles to send to the different
    processes, while particle data is copied to sendbuf before it is
    send. These two variables will grow in size if needed. Call with
    reset_buffers=True to reset these variables to their most basic
    forms, freeing up memory.
    """
    # No need to consider exchange of particles if running serially
    if 𝔹[nprocs == 1]:
        return
    # Only particles are exchangeable
    if component.representation != 'particles':
        return
    # Extract some variables from component
    N_local = component.N_local
    posx = component.posx
    posy = component.posy
    posz = component.posz
    # The index buffers indices_send[:] increase in size by this amount
    Δmemory = 2 + cast(0.01*N_local, 'Py_ssize_t')
    # Reset the number of particles to be sent
    for j in range(nprocs):
        N_send[j] = 0
    # Find out where to send which particle
    for i in range(N_local):
        # Rank of the process that local particle i belongs to
        owner = which_domain(posx[i], posy[i], posz[i])
        if owner != rank:
            # Particle owned by nonlocal process.
            # Append the particle index to the owner's index buffer.
            indices_send[owner][N_send[owner]] = i
            # Increase the number of particle to send to this process
            N_send[owner] += 1
            # Enlarge the index buffer indices_send[owner] if needed
            if N_send[owner] == indices_send_sizes[owner]:
                indices_send_sizes[owner] += Δmemory
                indices_send[owner] = realloc(
                    indices_send[owner],
                    indices_send_sizes[owner]*sizeof('Py_ssize_t'),
                )
    # No need to continue if no particles should be exchanged
    N_send_tot = sum(N_send)
    N_send_tot_global = allreduce(N_send_tot, op=MPI.SUM)
    if N_send_tot_global == 0:
        return
    # Grab a buffer for holding the data to be send.
    # The 'send' buffer is also used internally by smart_mpi.
    buffer_name = 'send'
    N_send_max = max(N_send)
    sendbuf_mv = get_buffer(N_send_max, buffer_name)
    # We additionally need a buffer storing signed char,
    # for the rung indices.
    if component.use_rungs:
        if rung_indices_arr.shape[0] < N_send_max:
            rung_indices_arr.resize(N_send_max, refcheck=False)
        rung_indices_buf = rung_indices_arr
    # Find out how many particles to receive
    N_recv = find_N_recv(N_send)
    # The maximum number of particles to
    # receive is stored in entrance rank.
    N_recv_max = N_recv[rank]
    N_recv_tot = sum(N_recv) - N_recv_max
    # Enlarge the component data attributes, if needed. This may not be
    # strictly necessary as more particles may be send than received.
    N_needed = N_local + N_recv_tot
    if component.N_allocated < N_needed:
        component.resize(N_needed)
        # Reextract position pointers
        posx = component.posx
        posy = component.posy
        posz = component.posz
    # Extract momenta pointers and all memory views
    # of the possibly resized data.
    momx    = component.momx
    momy    = component.momy
    momz    = component.momz
    posx_mv = component.posx_mv
    posy_mv = component.posy_mv
    posz_mv = component.posz_mv
    momx_mv = component.momx_mv
    momy_mv = component.momy_mv
    momz_mv = component.momz_mv
    # Extract Δmom pointers and memory views, storing the acceleration
    # used to determine the rung for each particle. If not using rungs,
    # we do not communicate these.
    Δmomx    = component.Δmomx
    Δmomy    = component.Δmomy
    Δmomz    = component.Δmomz
    Δmomx_mv = component.Δmomx_mv
    Δmomy_mv = component.Δmomy_mv
    Δmomz_mv = component.Δmomz_mv
    # Extract rung information
    rungs_N             = component.rungs_N
    rung_indices        = component.rung_indices
    rung_indices_mv     = component.rung_indices_mv
    rung_indices_jumped = component.rung_indices_jumped
    # Start index for received data
    index_recv_j = N_local
    # Exchange particles between processes
    for j in range(1, nprocs):
        # Process ranks to send/receive to/from
        ID_send = mod(rank + j, nprocs)
        ID_recv = mod(rank - j, nprocs)
        # Number of particles to send/receive
        N_send_j = N_send[ID_send]
        N_recv_j = N_recv[ID_recv]
        # The indices of particles to send
        indices_send_j = indices_send[ID_send]
        # Send/receive posx
        for i in range(N_send_j):
            sendbuf_mv[i] = posx[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=posx_mv[index_recv_j:],
            source=ID_recv,
        )
        # Send/receive posy
        for i in range(N_send_j):
            sendbuf_mv[i] = posy[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=posy_mv[index_recv_j:],
            source=ID_recv,
        )
        # Send/receive posz
        for i in range(N_send_j):
            sendbuf_mv[i] = posz[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=posz_mv[index_recv_j:],
            source=ID_recv,
        )
        # Send/receive momx
        for i in range(N_send_j):
            sendbuf_mv[i] = momx[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=momx_mv[index_recv_j:],
            source=ID_recv,
        )
        # Send/receive momy
        for i in range(N_send_j):
            sendbuf_mv[i] = momy[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=momy_mv[index_recv_j:],
            source=ID_recv,
        )
        # Send/receive momz
        for i in range(N_send_j):
            sendbuf_mv[i] = momz[indices_send_j[i]]
        Sendrecv(
            sendbuf_mv[:N_send_j],
            dest=ID_send,
            recvbuf=momz_mv[index_recv_j:],
            source=ID_recv,
        )
        # If using rungs, exchange the rung indices and the acceleration
        # stored in Δmom.
        with unswitch:
            if component.use_rungs:
                # Send/receive rung_indices
                for i in range(N_send_j):
                    # Add rung_index to buffer
                    rung_index = rung_indices[indices_send_j[i]]
                    rung_indices_buf[i] = rung_index
                    # Decrement rung population
                    rungs_N[rung_index] -= 1
                Sendrecv(
                    rung_indices_buf[:N_send_j],
                    dest=ID_send,
                    recvbuf=rung_indices_mv[index_recv_j:],
                    source=ID_recv,
                )
                # Increment rung population due to received particles
                # and set the jumped rung indices.
                for i in range(index_recv_j, index_recv_j + N_recv_j):
                    rung_index = rung_indices[i]
                    rungs_N[rung_index] += 1
                    # Set the jumped rung index equal to
                    # the rung index, signalling no upcoming jump.
                    rung_indices_jumped[i] = rung_index
                # Send/receive Δmomx
                for i in range(N_send_j):
                    sendbuf_mv[i] = Δmomx[indices_send_j[i]]
                Sendrecv(
                    sendbuf_mv[:N_send_j],
                    dest=ID_send,
                    recvbuf=Δmomx_mv[index_recv_j:],
                    source=ID_recv,
                )
                # Send/receive Δmomy
                for i in range(N_send_j):
                    sendbuf_mv[i] = Δmomy[indices_send_j[i]]
                Sendrecv(
                    sendbuf_mv[:N_send_j],
                    dest=ID_send,
                    recvbuf=Δmomy_mv[index_recv_j:],
                    source=ID_recv,
                )
                # Send/receive Δmomz
                for i in range(N_send_j):
                    sendbuf_mv[i] = Δmomz[indices_send_j[i]]
                Sendrecv(
                    sendbuf_mv[:N_send_j],
                    dest=ID_send,
                    recvbuf=Δmomz_mv[index_recv_j:],
                    source=ID_recv,
                )
        # Update the start index for received data
        index_recv_j += N_recv_j
        # Mark the holes in the data by setting posx[hole] = -1
        for k in range(N_send_j):
            posx[indices_send_j[k]] = -1
    # Move particle data to fill holes
    k_start = 0
    holes_filled = 0
    if N_send_tot > 0:
        # Loop backward over particles
        for i in range(index_recv_j - 1, -1, -1):
            # Index i should be a particle
            if posx[i] == -1:
                continue
            # Loop forward over holes
            for k in range(k_start, index_recv_j):
                # Index k should be a hole
                if posx[k] != -1:
                    continue
                # Particle i and hole k found.
                # Fill the hole with the particle.
                posx[k] = posx[i]
                posy[k] = posy[i]
                posz[k] = posz[i]
                momx[k] = momx[i]
                momy[k] = momy[i]
                momz[k] = momz[i]
                with unswitch:
                    if component.use_rungs:
                        rung_index = rung_indices[i]
                        rung_indices       [k] = rung_index
                        rung_indices_jumped[k] = rung_index  # no jump
                        Δmomx[k] = Δmomx[i]
                        Δmomy[k] = Δmomy[i]
                        Δmomz[k] = Δmomz[i]
                k_start = k + 1
                holes_filled += 1
                break
            # All holes have been filled
            if holes_filled == N_send_tot:
                break
    # Update N_local
    component.N_local = N_needed - N_send_tot
    # When not using rungs, all particles occupy rung 0
    if not component.use_rungs:
        rungs_N[0] = component.N_local
    # Update the rung flags
    if component.use_rungs:
        # Find and set lowest and highest populated rung
        component.set_lowest_highest_populated_rung()
        # There is no need to have the lowest active rung
        # be below the lowest populated rung.
        if component.lowest_active_rung < component.lowest_populated_rung:
            component.lowest_active_rung = component.lowest_populated_rung
    # If reset_buffers is True, reset the global indices_send and
    # sendbuf to their basic forms. This buffer will then be rebuild in
    # future calls.
    if reset_buffers:
        resize_buffer(1, buffer_name)
        for j in range(nprocs):
            indices_send[j] = realloc(indices_send[j], 1*sizeof('Py_ssize_t'))
            indices_send_sizes[j] = 1
        if component.use_rungs:
            rung_indices_arr.resize(1, refcheck=False)

# Function for communicating ghost values
# of domain grids between processes.
@cython.header(
    # Arguments
    grid_or_grids=object,  # double[:, :, ::1] or dict
    operation=str,
    # Locals
    grid='double[:, :, ::1]',
    grids=dict,
    i='int',
    index_recv_bgn_i='Py_ssize_t',
    index_recv_end_i='Py_ssize_t',
    index_send_bgn_i='Py_ssize_t',
    index_send_end_i='Py_ssize_t',
    index_recv_bgn_j='Py_ssize_t',
    index_recv_end_j='Py_ssize_t',
    index_send_bgn_j='Py_ssize_t',
    index_send_end_j='Py_ssize_t',
    index_recv_bgn_k='Py_ssize_t',
    index_recv_end_k='Py_ssize_t',
    index_send_bgn_k='Py_ssize_t',
    index_send_end_k='Py_ssize_t',
    j='int',
    k='int',
    reverse='bint',
    returns='void',
)
def communicate_ghosts(grid_or_grids, operation):
    """This function can operate in two different modes depending on the
    operation argument:
    - operation == '+=':
        Current values in the ghost points will be send to their
        designated neighbour processes, where they will be added to the
        current values of the outer (non-ghost) layer of points.
    - operation == "=":
        All local ghost points will be assigned values based on the
        values stored at the corresponding points on neighbour
        processes. Current ghost point values will be ignored.
    """
    if isinstance(grid_or_grids, dict):
        grids = grid_or_grids
        for grid in grids.values():
            communicate_ghosts(grid, operation)
        return
    grid = grid_or_grids
    if grid is None:
        return
    # Set the direction of communication dependeing on the operation
    reverse = (operation == '=')
    # Loop over all 26 neighbour domains
    for i in range(-1, 2):
        if i == -1:
            # Send left, receive right
            index_send_bgn_i = 0
            index_send_end_i = ℤ[1*nghosts]
            index_recv_bgn_i = ℤ[grid.shape[0] - 2*nghosts]
            index_recv_end_i = ℤ[grid.shape[0] - 1*nghosts]
        elif i == 0:
            # Do not send to or receive from this direction.
            # Include the entire i-dimension of the local bulk.
            index_send_bgn_i = ℤ[1*nghosts]
            index_send_end_i = ℤ[grid.shape[0] - 1*nghosts]
            index_recv_bgn_i = ℤ[1*nghosts]
            index_recv_end_i = ℤ[grid.shape[0] - 1*nghosts]
        else:  # i == -1
            # Send right, receive left
            index_send_bgn_i = ℤ[grid.shape[0] - 1*nghosts]
            index_send_end_i = ℤ[grid.shape[0]]
            index_recv_bgn_i = ℤ[1*nghosts]
            index_recv_end_i = ℤ[2*nghosts]
        for j in range(-1, 2):
            if j == -1:
                # Send backward, receive forward
                index_send_bgn_j = 0
                index_send_end_j = ℤ[1*nghosts]
                index_recv_bgn_j = ℤ[grid.shape[1] - 2*nghosts]
                index_recv_end_j = ℤ[grid.shape[1] - 1*nghosts]
            elif j == 0:
                # Do not send to or receive from this direction.
                # Include the entire j-dimension of the local bulk.
                index_send_bgn_j = ℤ[1*nghosts]
                index_send_end_j = ℤ[grid.shape[1] - 1*nghosts]
                index_recv_bgn_j = ℤ[1*nghosts]
                index_recv_end_j = ℤ[grid.shape[1] - 1*nghosts]
            else:  # j == -1
                # Send forward, receive backward
                index_send_bgn_j = ℤ[grid.shape[1] - 1*nghosts]
                index_send_end_j = ℤ[grid.shape[1]]
                index_recv_bgn_j = ℤ[1*nghosts]
                index_recv_end_j = ℤ[2*nghosts]
            for k in range(-1, 2):
                if i == j == k == 0:
                    # Do not communicate the local bulk
                    continue
                if k == -1:
                    # Send downward, receive upward
                    index_send_bgn_k = 0
                    index_send_end_k = ℤ[1*nghosts]
                    index_recv_bgn_k = ℤ[grid.shape[2] - 2*nghosts]
                    index_recv_end_k = ℤ[grid.shape[2] - 1*nghosts]
                elif k == 0:
                    # Do not send to or receive from this direction.
                    # Include the entire k-dimension of the local bulk.
                    index_send_bgn_k = ℤ[1*nghosts]
                    index_send_end_k = ℤ[grid.shape[2] - 1*nghosts]
                    index_recv_bgn_k = ℤ[1*nghosts]
                    index_recv_end_k = ℤ[grid.shape[2] - 1*nghosts]
                else:  # k == -1
                    # Send upward, receive downward
                    index_send_bgn_k = ℤ[grid.shape[2] - 1*nghosts]
                    index_send_end_k = ℤ[grid.shape[2]]
                    index_recv_bgn_k = ℤ[1*nghosts]
                    index_recv_end_k = ℤ[2*nghosts]
                # Communicate this face/edge/corner
                smart_mpi(
                    grid[
                        index_send_bgn_i:index_send_end_i,
                        index_send_bgn_j:index_send_end_j,
                        index_send_bgn_k:index_send_end_k,
                    ],
                    grid[
                        index_recv_bgn_i:index_recv_end_i,
                        index_recv_bgn_j:index_recv_end_j,
                        index_recv_bgn_k:index_recv_end_k,
                    ],
                    dest  =rank_neighbouring_domain(+i, +j, +k),
                    source=rank_neighbouring_domain(-i, -j, -k),
                    reverse=reverse,
                    mpifun='Sendrecv',
                    operation=operation,
                )

# Function for cutting out domains as rectangular boxes in the best
# possible way. The return value is an array of 3 elements; the number
# of subdivisions of the box for each dimension. When all dimensions
# cannot be equally divided, the x-dimension is subdivided the most,
# then the y-dimension and lastly the z-dimension.
@cython.header(
    # Arguments
    n='int',
    # Locals
    elongation='double',
    elongation_min='double',
    factor='int',
    factor_pair=frozenset,
    factors=list,
    factors_pairs=set,
    factors_singles=set,
    factors_triplet=tuple,
    factors_triplet_best=tuple,
    factors_x=tuple,
    factors_x_mul='int',
    factors_y=tuple,
    factors_y_mul='int',
    factors_yz=list,
    factors_z_mul='int',
    i='int',
    m='int',
    r_x='int',
    r_y='int',
    returns='int[::1]',
)
def cutout_domains(n):
    if n == 1:
        return ones(3, dtype=C2np['int'])
    if n < 1:
        abort(f'Cannot cut the box into {n} domains')
    # Factorise n into primes
    factors = []
    m = n
    while m%2 == 0:
        factors.append(2)
        m //= 2
    for i in range(3, cast(ceil(sqrt(n)), 'int') + 1, 2):
        while m%i == 0:
            factors.append(i)
            m //= i
    if m != 1:
        factors.append(m)
    # Go through all triplets of factors and find the one resulting in
    # the least elongation of the domains, i.e. the triplet with the
    # smallest ratio between the largest and smallest element.
    factors_singles = set()
    factors_pairs = set()
    elongation_min = ထ
    factors_triplet_best = ()
    for r_x in range(1, len(factors) + 1):
        for factors_x in itertools.combinations(factors, r_x):
            factors_x_mul = np.prod(factors_x)
            if factors_x_mul in factors_singles:
                continue
            factors_singles.add(factors_x_mul)
            factors_yz = factors.copy()
            for factor in factors_x:
                factors_yz.remove(factor)
            factors_yz.append(1)
            for r_y in range(1, len(factors_yz) + 1):
                for factors_y in itertools.combinations(factors_yz, r_y):
                    factors_y_mul = np.prod(factors_y)
                    factor_pair = frozenset({factors_x_mul, factors_y_mul})
                    if factor_pair in factors_pairs:
                        continue
                    factors_pairs.add(factor_pair)
                    factors_z_mul = n//(factors_x_mul*factors_y_mul)
                    factors_triplet = (factors_x_mul, factors_y_mul, factors_z_mul)
                    elongation = np.max(factors_triplet)/np.min(factors_triplet)
                    if elongation < elongation_min:
                        factors_triplet_best = factors_triplet
                        elongation_min = elongation
    if len(factors_triplet_best) != 3 or np.prod(factors_triplet_best) != n:
        abort('Something went wrong during domain decomposition')
    return asarray(sorted(factors_triplet_best, reverse=True), dtype=C2np['int'])

# This function takes coordinates as arguments and returns the rank of
# the process that governs the domain in which the coordinates reside.
@cython.header(# Arguments
               x='double',
               y='double',
               z='double',
               # Locals
               x_index='int',
               y_index='int',
               z_index='int',
               returns='int',
               )
def which_domain(x, y, z):
    x_index = int(x/domain_size_x)
    y_index = int(y/domain_size_y)
    z_index = int(z/domain_size_z)
    return domain_layout[x_index, y_index, z_index]

# This function computes the ranks of the processes governing the
# domain which is located i domains to the right, j domains forward and
# k domains up, relative to the local domain.
@cython.pheader(
    # Arguments
    i='int',
    j='int',
    k='int',
    # Locals
    returns='int',
)
def rank_neighbouring_domain(i, j, k):
    return domain_layout[
        mod(domain_layout_local_indices[0] + i, domain_subdivisions[0]),
        mod(domain_layout_local_indices[1] + j, domain_subdivisions[1]),
        mod(domain_layout_local_indices[2] + k, domain_subdivisions[2]),
    ]

# Function which communicates local component data
@cython.header(
    # Arguments
    component_send='Component',
    variables=list,  # list of str's
    pairing_level=str,
    interaction_name=str,
    tile_indices_send='Py_ssize_t[::1]',
    dest='int',
    source='int',
    component_recv='Component',
    use_Δ_recv='bint',
    # Locals
    N_particles='Py_ssize_t',
    N_particles_recv='Py_ssize_t',
    dim='int',
    domain_layout_source='int[::1]',
    i='Py_ssize_t',
    j='Py_ssize_t',
    mv_recv='double[::1]',
    mv_recv_buf='double[::1]',
    mv_recv_list=list,
    mv_send='double[::1]',
    mv_send_buf='double[::1]',
    mv_send_list=list,
    n='Py_ssize_t',
    operation=str,
    rung='Py_ssize_t*',
    rung_index='signed char',
    rung_indices='signed char*',
    rung_indices_buf='signed char[::1]',
    rung_indices_jumped='signed char*',
    rung_indices_jumped_buf='signed char[::1]',
    rungs_N='Py_ssize_t*',
    subtiling_name=str,
    tile='Py_ssize_t**',
    tile_index='Py_ssize_t',
    tiles='Py_ssize_t***',
    tiles_recv='Py_ssize_t***',
    tiles_rungs_N='Py_ssize_t**',
    tiles_rungs_N_recv='Py_ssize_t**',
    tiling='Tiling',
    tiling_name=str,
    tiling_recv='Tiling',
    use_rungs='bint',
    variable=str,
    returns='Component',
)
def sendrecv_component(
    component_send, variables, pairing_level, interaction_name,
    tile_indices_send, dest, source, component_recv=None, use_Δ_recv=True,
):
    """This function operates in two modes:
    - Communicate data (no component_recv supplied):
      The data of component_send will be send and received
      into the global component_buffer.
      The component_buffer is then returned.
    - Communicate and apply buffers (a component_recv is supplied):
      The data buffers of component_send will be send and
      received into the data buffers of component_recv. The received
      buffer data is then used to update the corresponding data
      in component_recv.
      It is assumed that the data arrays of component_recv are large
      enough to hold the data from component_send.
      The return value is the updated component_recv.
    Note that if you try to use this function locally within a single
    process (dest == rank == source), nothing will happen. Thus you
    cannot rely on this function to apply buffers to data attributes.
    The variables argument must be a list of str's designating
    which local data variables of component_send to communicate.
    The implemented variables are:
    - 'pos' (posx, posy and posz)
    - 'mom' (momx, momy and momz)
    Only particles within the tiles given by tile_indices_send will be
    communicated. The tiling used will be determined from
    interaction_name. In the case of pairing_level == 'domain',
    no actual tiling should be used, and so here we use the trivial
    tiling. Note that the passed tile_indices_send should be identical
    on all processes. After tile particles have been communicated,
    the returned buffer component will be tile sorted at the domain
    (tile, not subtile) level. Note that the particle order is not
    preserved when doing such a communication + tile sorting.
    """
    global component_buffer
    if component_send.representation != 'particles':
        abort('The sendrecv_component function is only implemented for particle components')
    # No communication is needed if the destination and source is
    # really the local process.
    if dest == rank == source:
        return component_send
    # Determine the mode of operation
    operation = '+='
    if 𝔹[component_recv is None]:
        operation = '='
    # Determine which tiling to use
    if pairing_level == 'tile':
        tiling_name = f'{interaction_name} (tiles)'
    else:  # pairing_level == 'domain'
        tiling_name = 'trivial'
    # Find out how many particles should be communicated
    tiling = component_send.tilings[tiling_name]
    tiles         = tiling.tiles
    tiles_rungs_N = tiling.tiles_rungs_N
    if 𝔹[operation == '=']:
        N_particles = 0
        for i in range(tile_indices_send.shape[0]):
            rungs_N = tiles_rungs_N[tile_indices_send[i]]
            for rung_index in range(
                ℤ[component_send.lowest_populated_rung],
                ℤ[component_send.highest_populated_rung + 1],
            ):
                N_particles += rungs_N[rung_index]
    else:
        # When operation == '+=', we always send all particles back
        # to the process from which they originally came.
        # Really we should only include particles within the tiles
        # given by tile_indices_send, and so the above loop over
        # these tiles is correct even when operation == '+='.
        # However, as long as this function has been called
        # correctly, the component_send is really just a buffer
        # component storing only particles within the specified
        # tiles, and so we can skip the counting above.
        N_particles = component_send.N_local
        # Also extract tile variables from component_recv
        tiling_recv = component_recv.tilings[tiling_name]
        tiles_recv         = tiling_recv.tiles
        tiles_rungs_N_recv = tiling_recv.tiles_rungs_N
    N_particles_recv = sendrecv(N_particles, dest=dest, source=source)
    # In communicate mode (operation == '='),
    # the global component_buffer is used as component_recv.
    if 𝔹[operation == '=']:
        # We cannot simply import Component from the species module,
        # as this would create an import loop. Instead, the first time
        # the component_buffer is needed, we grab the type of the passed
        # component_send (Component) and instantiate such an instance.
        if component_buffer is None:
            component_buffer = type(component_send)('', 'cold dark matter', N=1)
        # Adjust important meta data on the buffer component
        component_buffer.name             = component_send.name
        component_buffer.species          = component_send.species
        component_buffer.representation   = component_send.representation
        component_buffer.N                = component_send.N
        component_buffer.mass             = component_send.mass
        component_buffer.softening_length = component_send.softening_length
        component_buffer.use_rungs        = component_send.use_rungs
        # Enlarge the data arrays of the component_buffer if necessary
        component_buffer.N_local = N_particles_recv
        if component_buffer.N_allocated < component_buffer.N_local:
            # Temporarily set use_rungs = True to ensure that the
            # rung_indices and rung_indices_jumped
            # get resized as well.
            use_rungs = component_buffer.use_rungs
            component_buffer.use_rungs = True
            component_buffer.resize(component_buffer.N_local)
            component_buffer.use_rungs = use_rungs
        # Use component_buffer as component_recv
        component_recv = component_buffer
    # Operation-dependent preparations for the communication
    if 𝔹[operation == '=']:
        # In communication mode the particles within the tiles are
        # temporarily copied to the mv_send_buf buffer.
        # Make sure that this is large enough.
        mv_send_buf = get_buffer(N_particles, 'send')
    else:  # operation == '+=':
        # We need to receive the data into a buffer, and then update the
        # local data by this amount. Get the buffer.
        mv_recv_buf = get_buffer(N_particles_recv, 'recv')
    # Do the communication for each variable
    for variable in variables:
        # Get arrays to send and receive into
        if variable == 'pos':
            with unswitch:
                if 𝔹[operation == '=']:
                    mv_send_list = component_send.pos_mv
                    mv_recv_list = component_recv.pos_mv
                else:  # operation == '+='
                    abort('Δpos not implemented')
        elif variable == 'mom':
            with unswitch:
                if 𝔹[operation == '=']:
                    mv_send_list = component_send.mom_mv
                    mv_recv_list = component_recv.mom_mv
                else:  # operation == '+='
                    mv_send_list = component_send.Δmom_mv
                    if use_Δ_recv:
                        mv_recv_list = component_recv.Δmom_mv
                    else:
                        mv_recv_list = component_recv.mom_mv
        else:
            abort(
                f'Currently only "pos" and "mom" are implemented '
                f'as variables in sendrecv_component()'
            )
        for dim in range(3):
            mv_send = mv_send_list[dim][:component_send.N_local]
            # In communication mode we only need to send the particular
            # particles within the specified tiles. Here we copy the
            # variable of these specific particles to a buffer.
            with unswitch:
                if 𝔹[operation == '=']:
                    n = 0
                    for i in range(tile_indices_send.shape[0]):
                        tile_index = tile_indices_send[i]
                        tile    = tiles        [tile_index]
                        rungs_N = tiles_rungs_N[tile_index]
                        for rung_index in range(
                            ℤ[component_send.lowest_populated_rung],
                            ℤ[component_send.highest_populated_rung + 1],
                        ):
                            rung = tile[rung_index]
                            for j in range(rungs_N[rung_index]):
                                mv_send_buf[n] = mv_send[rung[j]]
                                n += 1
                    mv_send = mv_send_buf[:n]
            # Communicate the particle data
            with unswitch:
                if 𝔹[operation == '=']:
                    mv_recv = mv_recv_list[dim][:component_recv.N_local]
                    Sendrecv(mv_send, recvbuf=mv_recv, dest=dest, source=source)
                else:  # operation == '+='
                    Sendrecv(mv_send, recvbuf=mv_recv_buf, dest=dest, source=source)
                    # Update particles in the specified tiles.
                    # Though only particles on active rungs will have
                    # non-zero updates, we have to loop over all
                    # populated rungs, as otherwise the indexing into
                    # mv_recv_buf will not be correct.
                    mv_recv = mv_recv_list[dim][:component_recv.N_local]
                    n = 0
                    for i in range(tile_indices_send.shape[0]):
                        tile_index = tile_indices_send[i]
                        tile    = tiles_recv        [tile_index]
                        rungs_N = tiles_rungs_N_recv[tile_index]
                        for rung_index in range(
                            ℤ[component_recv.lowest_populated_rung],
                            ℤ[component_recv.highest_populated_rung + 1],
                        ):
                            rung = tile[rung_index]
                            for j in range(rungs_N[rung_index]):
                                mv_recv[rung[j]] += mv_recv_buf[n]
                                n += 1
    # When in communication mode, we additionally need to communicate
    # the rung indices and rung jumps of the communicated particles.
    # If not using rungs, we skip this.
    if 𝔹[operation == '=' and component_send.use_rungs]:
        # Create contiguous memory view over rung indices.
        # We must only include the rung indices for
        # particles within the specified tiles.
        if rung_indices_arr.shape[0] < N_particles:
            rung_indices_arr.resize(N_particles, refcheck=False)
        rung_indices_buf = rung_indices_arr
        n = 0
        for i in range(tile_indices_send.shape[0]):
            tile_index = tile_indices_send[i]
            rungs_N = tiles_rungs_N[tile_index]
            for rung_index in range(
                ℤ[component_send.lowest_populated_rung],
                ℤ[component_send.highest_populated_rung + 1],
            ):
                for j in range(rungs_N[rung_index]):
                    rung_indices_buf[n] = rung_index
                    n += 1
        # Communicate rung indices
        Sendrecv(rung_indices_buf[:n],
            recvbuf=component_recv.rung_indices_mv, dest=dest, source=source)
        # Fill buffer with jumped rung indices
        # and communicate these as well.
        rung_indices_jumped = component_send.rung_indices_jumped
        rung_indices_jumped_buf = rung_indices_buf  # reuse buffer
        n = 0
        for i in range(tile_indices_send.shape[0]):
            tile_index = tile_indices_send[i]
            tile    = tiles        [tile_index]
            rungs_N = tiles_rungs_N[tile_index]
            for rung_index in range(
                ℤ[component_send.lowest_populated_rung],
                ℤ[component_send.highest_populated_rung + 1],
            ):
                rung = tile[rung_index]
                for j in range(rungs_N[rung_index]):
                    rung_indices_jumped_buf[n] = rung_indices_jumped[rung[j]]
                    n += 1
        Sendrecv(rung_indices_jumped_buf[:n],
            recvbuf=component_recv.rung_indices_jumped_mv, dest=dest, source=source)
        # Count up how many particles occupy each rung
        rung_indices = component_recv.rung_indices
        rungs_N = component_recv.rungs_N
        for rung_index in range(N_rungs):
            rungs_N[rung_index] = 0
        for i in range(component_recv.N_local):
            rungs_N[rung_indices[i]] += 1
        # Find and set lowest and highest populated rung
        component_recv.set_lowest_highest_populated_rung()
        # Communicate the active rung
        component_recv.lowest_active_rung = sendrecv(
            component_send.lowest_active_rung, dest=dest, source=source,
        )
        if component_recv.lowest_active_rung < component_recv.lowest_populated_rung:
            # There is no need to have the lowest active rung
            # be below the lowest populated rung.
            component_recv.lowest_active_rung = component_recv.lowest_populated_rung
    # When in communication mode the buffer (recv) component
    # needs to know its own tiling.
    if 𝔹[operation == '=']:
        # Ensure that the required tiling (and subtiling)
        # is instantiated on the buffer component.
        tiling_recv = component_recv.tilings.get(tiling_name)
        if tiling_recv is None:
            component_recv.init_tiling(tiling_name, initial_rung_size=0)
            tiling_recv = component_recv.tilings[tiling_name]
            if 𝔹[tiling_name != 'trivial']:
                subtiling_name = f'{interaction_name} (subtiles)'
                component_recv.init_tiling(subtiling_name, initial_rung_size=0)
        # Place the tiling over the domain of the process
        # with a rank given by 'source'.
        if 𝔹[tiling_name != 'trivial']:
            domain_layout_source = asarray(np.unravel_index(source, domain_subdivisions),
                dtype=C2np['int'])
            tiling_recv.relocate(asarray(
                (
                    domain_layout_source[0]*domain_size_x,
                    domain_layout_source[1]*domain_size_y,
                    domain_layout_source[2]*domain_size_z,
                ),
                dtype=C2np['double'],
            ))
        # Perform tile sorting, but do not sort into subtiles
        component_recv.tile_sort(tiling_name)
    return component_recv
# Declare buffer component and buffer for rung_indices
# used by sendrecv_component(). The rung_indices_arr array is also
# used by the exchange() function.
cython.declare(
    component_buffer='Component',
    rung_indices_arr=object,
)
component_buffer = None
rung_indices_arr = empty(1, dtype=C2np['signed char'])

# Very general function for different MPI communications
@cython.pheader(
    # Arguments
    block_send=object,  # Memoryview of dimension 1, 2 or 3
    block_recv=object,  # Memoryview of dimension 1, 2 or 3, or int
    dest='int',
    source='int',
    root='int',
    reverse='bint',
    mpifun=str,
    operation=str,
    # Local
    arr_recv=object,  # NumPy aray
    arr_send=object,  # NumPy aray
    block_recv_passed_as_scalar='bint',
    contiguous_recv='bint',
    contiguous_send='bint',
    data_recv=object,  # NumPy aray
    data_send=object,  # NumPy aray
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    recving='bint',
    sending='bint',
    shape_send=tuple,
    size_recv='Py_ssize_t',
    size_send='Py_ssize_t',
    sizes_recv='Py_ssize_t[::1]',
    recvbuf_mv='double[::1]',
    recvbuf_name=object,  # int or str
    reverse_mpifun_mapping=dict,
    sendbuf_mv='double[::1]',
    using_recvbuf='bint',
    returns=object,  # NumPy array or mpi4py.MPI.Request
)
def smart_mpi(
    block_send=(), block_recv=(), dest=-1, source=-1, root=master_rank,
    reverse=False, mpifun='', operation='=',
):
    """This function will do MPI communication. It will send the data in
    the array/memoryview block_send to the process of rank dest
    and receive data into array/memoryview block_recv from rank source.
    The arrays can be of any shape (currently bigger than 0 and less
    than 4) and size and may be different for the two.
    If block_recv is larger than the received data, the extra elements
    in the end will be filled with junk if the dimension of block_recv
    is larger than 1. Though for the sake of performance, always pass
    a fitting block_recv.
    The MPI function to use is specified in the mpifun argument
    (e.g. mpifun='sendrecv' or mpifun='send'). Uppercase communication
    (array communication) is always used, regardless of the case of the
    value of mpifun.
    For some MPI communications a root process should be specified.
    This can be set by the root argument.
    All arguments are optional, so that it is not needed to specify e.g.
    block_recv when doing a Send. For Cython to be able to compile this,
    a cython.pheader decorator is used (corresponding to cython.ccall
    or cpdef). Also, if a call to smart_mpi results in a receive but not
    a send, block_recv can be passed as the first argument
    instead of block_send (which is not used in this case).
    It is allowed not to pass in a block_recv, even when a message
    should be received. In that case, the recvbuf buffer will be used
    and returned.
    The received data can either be copied into block_recv (overwriting
    existing data) or it can be added to the existing data. Change
    this behaviour through the operation argument (operation='=' or
    operation='+=').
    If the passed blocks are contiguous, they will be used directly
    in the communication (though in the case of block_recv, only when
    operation='='). If not, contiguous buffers will be used. The
    buffers used are the variables sendbuf/sendbuf_mv and
    recvbuf/recvbuf_mv. These will be enlarged if necessary.
    Since the buffers contain doubles, the passed arrays must also
    contain doubles if the buffers are to be used. If communication can
    take place directly without the use of the buffers, the passed
    arrays may contain any type (though the type of the send and recv
    block should always be identical).
    What is returned depends on the choice of mpifun. Whenever a message
    should be received, the passed block_recv is returned (as block_recv
    is populated with values in-place, this is rarely used). When a
    non-blocking send-only is used, the MPI request is returned. When a
    blocking send-only is used, None is returned.
    If reverse is True, the communication is reversed, meaning that
    sending block_send to dist and receiving into block_recv from source
    turns into sending block_recv to source and receiving into
    block_send from dist.
    """
    # Sanity check on operation argument
    if master and operation not in {'=', '+='}:
        abort(f'smart_mpi() got operation = "{operation}" ∉ {{"=", "+="}}')
    # Determine whether we are sending and/or receiving
    mpifun = mpifun.lower()
    sending = False
    recving = False
    if 'all' in mpifun:
        sending = True
        recving = True
    else:
        if 'send' in mpifun:
            sending = True
            if dest == -1:
                abort('Cannot send when no destination is given')
        if 'recv' in mpifun:
            recving = True
            if source == -1:
                abort('Cannot receive when no source is given')
        if 'bcast' in mpifun:
            sending = (rank == root)
            recving = not sending
        if 'gather' in mpifun:
            sending = True
            recving = (rank == root)
    if not sending and not recving:
        if mpifun:
            abort(f'MPI function "{mpifun}" not understood')
        else:
            abort('Which MPI function to use is not specified')
    # If requested, reverse the communication direction
    if reverse:
        # Swap the send and receive blocks
        block_send, block_recv = block_recv, block_send
        # Swap the source and destination
        dest, source = source, dest
        # Reverse the MPI function
        reverse_mpifun_mapping = {'recv'    : 'send',
                                  'send'    : 'recv',
                                  'sendrecv': 'sendrecv',
                                   }
        if mpifun not in reverse_mpifun_mapping:
            abort(f'MPI function "{mpifun}" cannot be reversed')
        mpifun = reverse_mpifun_mapping[mpifun]
    # If only receiving, block_recv should be
    # accessible as the first argument.
    if (
            not sending
        and     recving
        and not (isinstance(block_send, tuple) and len(block_send) == 0)  # block_send != ()
        and     (isinstance(block_recv, tuple) and len(block_recv) == 0)  # block_recv == ()
    ):
        block_send, block_recv = block_recv, block_send
    # If block_recv is an int or str,
    # this designates a specific buffer to use as recvbuf.
    recvbuf_name = 'recv'
    if isinstance(block_recv, (int, str)):
        recvbuf_name = block_recv
    # NumPy arrays over the data
    arr_send = asarray(block_send)
    arr_recv = asarray(block_recv)
    # If the input blocks contain different types (and one of them
    # contain doubles), convert them both to doubles.
    # This is not done in-place, meaning that the passed recv_block will
    # not be changed! The returned block should be used instead.
    if sending and recving:
        if   (    arr_send.dtype == np.dtype(C2np['double'])
              and arr_recv.dtype != np.dtype(C2np['double'])):
            arr_recv = arr_recv.astype(C2np['double'])
        elif (    arr_send.dtype != np.dtype(C2np['double'])
              and arr_recv.dtype == np.dtype(C2np['double'])):
            arr_send = arr_send.astype(C2np['double'])
    # Are the passed arrays contiguous?
    contiguous_send = arr_send.flags.c_contiguous
    contiguous_recv = arr_recv.flags.c_contiguous
    # The send and recv blocks cannot be scalar NumPy arrays.
    # Do an in-place reshape to 1D-arrays of size 1.
    if arr_send.ndim == 0:
        arr_send.resize(1, refcheck=False)
    block_recv_passed_as_scalar = False
    if arr_recv.ndim == 0:
        block_recv_passed_as_scalar = True
        arr_recv.resize(1, refcheck=False)
    size_send = arr_send.size
    shape_send = arr_send.shape
    # Figure out the size of the data to be received
    size_recv = 0
    if mpifun == 'bcast':
        # Broadcast the shape of the data to be broadcasted
        shape_recv = bcast(arr_send.shape, root=root)
        size_recv = np.prod(shape_recv)
        if rank == root:
            size_recv = 0
    elif mpifun == 'gather':
        # The root process will receive a block of size_send
        # from all processes.
        if rank == root:
            size_recv = nprocs*size_send
    elif mpifun == 'gatherv':
        # The root process will receive blocks of possibly different
        # sizes from all processes. Communicate these sizes.
        if rank == root:
            sizes_recv = empty(nprocs, dtype=C2np['Py_ssize_t'])
        Gather(asarray(size_send, dtype=C2np['Py_ssize_t']),
            sizes_recv if rank == root else None)
    elif sending and recving:
        if mpifun == 'allgather':
            # A block of size_send is to be received from each process
            size_recv = nprocs*size_send
        elif mpifun == 'allgatherv':
            # The blocks to be received from each process may have
            # different sizes. Communicate these sizes.
            sizes_recv = empty(nprocs, dtype=C2np['Py_ssize_t'])
            Allgather(asarray(size_send, dtype=C2np['Py_ssize_t']), sizes_recv)
        else:
            # Communicate the size of the data to be exchanged
            size_recv = sendrecv(size_send, dest=dest, source=source)
    elif recving:
        # The exact size of the data to receive is not known,
        # but it cannot be larger than the size of the receiver block.
        size_recv = arr_recv.size
    # Based on the contiguousness of the input arrays, assign the names
    # data_send and data_recv to the contiguous blocks of data,
    # which are to be passed into the MPI functions.
    if contiguous_send:
        data_send = arr_send
    else:
        sendbuf_mv = get_buffer(size_send, 'send')
        data_send = sendbuf_mv
    # When no block_recv is passed, use the recvbuf buffer
    using_recvbuf = False
    if arr_recv.size == 0 or block_recv_passed_as_scalar:
        using_recvbuf = True
        recvbuf_mv = get_buffer(size_recv, recvbuf_name)
        data_recv = recvbuf_mv
        arr_recv = asarray(data_recv)
    elif contiguous_recv and operation == '=':
        # Only if operation == '=' can we receive
        # directly into the input array.
        data_recv = arr_recv
    else:
        using_recvbuf = True
        recvbuf_mv = get_buffer(size_recv, recvbuf_name)
        data_recv = recvbuf_mv
    # Fill send buffer if this is to be used
    if sending and not contiguous_send:
        copy_to_contiguous(arr_send, sendbuf_mv)
    # Do the communication
    if mpifun == 'allgather':
        Allgather(data_send, data_recv)
    elif mpifun == 'allgatherv':
        Allgatherv(data_send, (data_recv, sizes_recv))
    elif mpifun == 'bcast':
        if rank == root:
            Bcast(data_send, root=root)
        else:
            Bcast(data_recv, root=root)
    elif mpifun == 'gather':
        Gather(data_send, data_recv, root=root)
    elif mpifun == 'gatherv':
        Gatherv(data_send, (data_recv, sizes_recv) if rank == root else None, root=root)
    elif mpifun == 'isend':
        return Isend(data_send, dest=dest)
    elif mpifun == 'recv':
        Recv(data_recv, source=source)
    elif mpifun == 'send':
        Send(data_send, dest=dest)
    elif mpifun == 'sendrecv':
        Sendrecv(data_send, recvbuf=data_recv, dest=dest, source=source)
    else:
        abort('MPI function "{}" is not implemented'.format(mpifun))
    # If only sending, return now
    if not recving:
        return data_send
    # If nothing was received, return an empty slice of arr_recv
    if size_recv == 0:
        return arr_recv[:0]
    # Copy or add the received data from the buffer
    # to the passed block_recv (arr_recv), if needed.
    if (operation == '=' and not contiguous_recv) or operation == '+=':
        copy_to_noncontiguous(recvbuf_mv, arr_recv, operation)
    # If both sending and receiving, the two blocks of data
    # should (probably) have the same shape. If no block_recv was
    # supplied, arr_recv will always be 1D.
    # In this case, do a reshaping.
    if sending and recving and using_recvbuf and size_send == size_recv:
        arr_recv = arr_recv.reshape(shape_send)
    # When broadcasting, the received data should be of the same size
    # as that which was send.
    if mpifun == 'bcast' and using_recvbuf:
        arr_recv = arr_recv.reshape(shape_recv)
    # Return the now populated arr_recv
    return arr_recv

# Function for copying a multi-dimensional non-contiguous
# array into a 1D contiguous buffer.
@cython.header(
    # Arguments
    arr=object,  # np.ndarray
    buf='double[::1]',
    # Locals
    bufptr='double*',
    bufview1D='double[::1]',
    bufview2D='double[:, ::1]',
    bufview3D='double[:, :, ::1]',
    i='Py_ssize_t',
    index='Py_ssize_t',
    index_i='Py_ssize_t',
    index_ij='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    ndim='int',
    size='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    view1D='double[:]',
    view2D='double[:, :]',
    view3D='double[:, :, :]',
    viewcontig='double[::1]',
    returns='void',
)
def copy_to_contiguous(arr, buf):
    """It is assumed that the contiguous buf is at least
    as large as the arr, but it may be larger.
    """
    arr = asarray(arr, dtype=C2np['double'])
    if arr.flags.c_contiguous:
        size = arr.size
        viewcontig = arr.reshape(size)
        buf[:size] = viewcontig
        return
    ndim = arr.ndim
    bufptr = cython.address(buf[:])
    if ndim == 1:
        view1D = arr
        bufview1D = cast(bufptr, 'double[:view1D.shape[0]]')
        bufview1D[:] = view1D
    elif ndim == 2:
        view2D = arr
        bufview2D = cast(bufptr, 'double[:view2D.shape[0], :view2D.shape[1]]')
        bufview2D[...] = view2D
    elif ndim == 3:
        view3D = arr
        bufview3D = cast(bufptr, 'double[:view3D.shape[0], :view3D.shape[1], :view3D.shape[2]]')
        bufview3D[...] = view3D
    elif ndim == 0:
        pass
    else:
        abort(f'copy_to_contiguous() got array with {ndim} dimensions')

# Function for copying a 1D contiguous buffer into a
# multi-dimensional non-contiguous array.
@cython.header(
    # Arguments
    buf='double[::1]',
    arr=object,  # np.ndarray
    operation=str,
    # Locals
    bufptr='double*',
    bufview1D='double[::1]',
    bufview2D='double[:, ::1]',
    bufview3D='double[:, :, ::1]',
    i='Py_ssize_t',
    index='Py_ssize_t',
    index_i='Py_ssize_t',
    index_ij='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    ndim='int',
    size='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    view1D='double[:]',
    view2D='double[:, :]',
    view3D='double[:, :, :]',
    viewcontig='double[::1]',
    returns='void',
)
def copy_to_noncontiguous(buf, arr, operation='='):
    """It is assumed that the contiguous buf is at least
    as large as the arr, but it may be larger.
    """
    arr = asarray(arr, dtype=C2np['double'])
    bufptr = cython.address(buf[:])
    if arr.flags.c_contiguous:
        size = arr.size
        viewcontig = arr.reshape(size)
        if operation == '=':
            viewcontig[:] = buf[:size]
        else:  # operation == '+='
            for index in range(size):
                viewcontig[index] += bufptr[index]
        return
    ndim = arr.ndim
    if ndim == 1:
        view1D = arr
        size = view1D.shape[0]
        if operation == '=':
            bufview1D = cast(bufptr, 'double[:size]')
            view1D[:] = bufview1D
        else:  # operation == '+='
            for index in range(size):
                view1D[index] += bufptr[index]
    elif ndim == 2:
        view2D = arr
        size_i, size_j = view2D.shape[0], view2D.shape[1]
        if operation == '=':
            bufview2D = cast(bufptr, 'double[:size_i, :size_j]')
            view2D[...] = bufview2D
        else:  # operation == '+='
            index_i = -size_j
            for i in range(size_i):
                index_i += size_j
                for j in range(size_j):
                    index = index_i + j
                    view2D[i, j] += bufptr[index]
    elif ndim == 3:
        view3D = arr
        size_i, size_j, size_k = view3D.shape[0], view3D.shape[1], view3D.shape[2]
        if operation == '=':
            bufview3D = cast(bufptr, 'double[:size_i, :size_j, :size_k]')
            view3D[...] = bufview3D
        else:  # operation == '+='
            index_i = -size_j
            for i in range(size_i):
                index_i += size_j
                index_ij = (index_i - 1)*size_k
                for j in range(size_j):
                    index_ij += size_k
                    for k in range(size_k):
                        index = index_ij + k
                        view3D[i, j, k] += bufptr[index]
    elif ndim == 0:
        pass
    else:
        abort(f'copy_to_noncontiguous() got array with {ndim} dimensions')

# Function which manages buffers used by other functions
@cython.pheader(
    # Arguments
    size_or_shape=object,  # Py_ssize_t or tuple
    buffer_name=object,  # Any hashable object
    nullify='bint',
    # Local
    N_buffers='Py_ssize_t',
    buffer='double*',
    buffer_mv='double[::1]',
    i='Py_ssize_t',
    index='Py_ssize_t',
    shape=tuple,
    size='Py_ssize_t',
    size_given='bint',
    returns=object,  # multi-dimensional array of doubles
)
def get_buffer(size_or_shape=-1, buffer_name=0, nullify=False):
    """This function returns a contiguous buffer containing doubles.
    The buffer will be exactly of size 'size_or_shape' when this is an
    integer, or exactly the shape of 'size_or_shape' when this is
    a tuple. If no size or shape is given, the buffer will be returned
    as a 1D array with whatever size it happens to have.
    When multiple buffers are in use, a specific buffer can be
    requested by passing a buffer_name, which can be any hashable type.
    A buffer with the given name does not have to exist beforehand.
    A given buffer will be reallocated (enlarged) if necessary.
    If nullify is True, all elements of the buffer will be set to 0.
    """
    global buffers
    # Get shape and size from argument
    if size_or_shape == -1:
        size_given = False
        size_or_shape = 1
    else:
        size_given = True
    shape = size_or_shape if isinstance(size_or_shape, tuple) else (size_or_shape, )
    size = np.prod(shape)
    # The smallest possible buffer size is 1
    if size == 0:
        size = 1
        shape = (1, )
    # Fetch or create the buffer
    if buffer_name in buffers_mv:
        # This buffer already exists
        index = 0
        for key in buffers_mv:
            if key == buffer_name:
                break
            index += 1
        buffer = buffers[index]
        buffer_mv = buffers_mv[buffer_name]
        if size > buffer_mv.shape[0]:
            # Enlarge this buffer
            resize_buffer(size, buffer_name)
            buffer = buffers[index]
            buffer_mv = buffers_mv[buffer_name]
        elif not size_given:
            # No size was given. Use the entire array.
            size = buffer_mv.shape[0]
            shape = (size, )
    else:
        # This buffer does not exist yet. Create it.
        buffer = malloc(size*sizeof('double'))
        N_buffers = len(buffers_mv) + 1
        buffers = realloc(buffers, N_buffers*sizeof('double*'))
        buffers[N_buffers - 1] = buffer
        buffer_mv = cast(buffer, 'double[:size]')
        buffers_mv[buffer_name] = buffer_mv
    # Nullify the buffer, if required
    if nullify:
        for i in range(size):
            buffer[i] = 0
    # Return the buffer in the requsted shape
    return np.reshape(buffer_mv[:size], shape)
# Function which resizes one of the global buffers
@cython.header(# Arguments
               buffer_name=object,  # Any hashable object
               size='Py_ssize_t',
               # Local
               buffer='double*',
               buffer_mv='double[::1]',
               index='Py_ssize_t',
               )
def resize_buffer(size, buffer_name):
    if buffer_name not in buffers_mv:
        abort('Cannot resize buffer "{}" as it does not exist'.format(buffer_name))
    index = 0
    for key in buffers_mv:
        if key == buffer_name:
            break
        index += 1
    buffer = buffers[index]
    buffer = realloc(buffer, size*sizeof('double'))
    buffers[index] = buffer
    buffer_mv = cast(buffer, 'double[:size]')
    buffers_mv[buffer_name] = buffer_mv
# Initialize buffers
cython.declare(buffers='double**',
               buffer='double*',
               buffer_mv='double[::1]',
               buffers_mv=object,  # OrderedDict
               )
buffers = malloc(1*sizeof('double*'))
buffer = malloc(1*sizeof('double'))
buffers[0] = buffer
buffer_mv = cast(buffer, 'double[:1]')
buffers_mv = collections.OrderedDict()
buffers_mv[0] = buffer_mv

# Cutout domains at import time
cython.declare(
    domain_subdivisions='int[::1]',
    domain_layout='int[:, :, ::1]',
    domain_layout_local_indices='int[::1]',
    domain_size_x='double',
    domain_size_y='double',
    domain_size_z='double',
    domain_volume='double',
    domain_start_x='double',
    domain_start_y='double',
    domain_start_z='double',
    domain_end_x='double',
    domain_end_y='double',
    domain_end_z='double',
)
# Number of subdivisions (domains) of the box
# in each of the three dimensions.
domain_subdivisions = cutout_domains(nprocs)
# The global 3D layout of the division of the box
domain_layout = arange(nprocs, dtype=C2np['int']).reshape(domain_subdivisions)
# The indices in domain_layout of the local domain
domain_layout_local_indices = asarray(np.unravel_index(rank, domain_subdivisions), dtype=C2np['int'])
# The size of the domain, which is the same for all of them
domain_size_x = boxsize/domain_subdivisions[0]
domain_size_y = boxsize/domain_subdivisions[1]
domain_size_z = boxsize/domain_subdivisions[2]
domain_volume = domain_size_x*domain_size_y*domain_size_z
# The start and end coordinates of the local domain
domain_start_x = domain_layout_local_indices[0]*domain_size_x
domain_start_y = domain_layout_local_indices[1]*domain_size_y
domain_start_z = domain_layout_local_indices[2]*domain_size_z
domain_end_x = domain_start_x + domain_size_x
domain_end_y = domain_start_x + domain_size_x
domain_end_z = domain_start_x + domain_size_x

# Initialize variables used in the exchange function
cython.declare(
    N_send='Py_ssize_t[::1]',
    indices_send='Py_ssize_t**',
    indices_send_sizes='Py_ssize_t[::1]',
    j='int',
)
# This variable stores the number of particles to send to each prcess
N_send = empty(nprocs, dtype=C2np['Py_ssize_t'])
# This Py_ssize_t** variable stores the indices of particles to be send
# to other processes. That is, indices_send[other_rank][i] is the local
# index of some particle which should be send to other_rank.
indices_send = malloc(nprocs*sizeof('Py_ssize_t*'))
for j in range(nprocs):
    indices_send[j] = malloc(1*sizeof('Py_ssize_t'))
# The size of the allocated indices_send[:] memory
indices_send_sizes = ones(nprocs, dtype=C2np['Py_ssize_t'])
