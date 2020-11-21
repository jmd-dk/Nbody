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
    'from communication import        '
    '    communicate_ghosts,          '
    '    domain_layout_local_indices, '
    '    domain_size_x,               '
    '    domain_size_y,               '
    '    domain_size_z,               '
    '    domain_start_x,              '
    '    domain_start_y,              '
    '    domain_start_z,              '
    '    domain_subdivisions,         '
    '    get_buffer,                  '
    '    partition,                   '
    '    rank_neighbouring_domain,    '
    '    smart_mpi,                   '
)

# Function pointer types used in this module
pxd('ctypedef double* (*func_dstar_ddd)(double, double, double)')

# Import declarations from fft.c
pxd("""
# FFT functionality via FFTW from fft.c
cdef extern from "fft.c":
    # The fftw_plan type
    ctypedef struct fftw_plan_struct:
        pass
    ctypedef fftw_plan_struct *fftw_plan
    # The returned struct of fftw_setup
    struct fftw_return_struct:
        ptrdiff_t gridsize_local_i
        ptrdiff_t gridsize_local_j
        ptrdiff_t gridstart_local_i
        ptrdiff_t gridstart_local_j
        double* grid
        fftw_plan plan_forward
        fftw_plan plan_backward
    # Functions
    fftw_return_struct fftw_setup(ptrdiff_t gridsize_i,
                                  ptrdiff_t gridsize_j,
                                  ptrdiff_t gridsize_k,
                                  char*     rigor,
                                  bint      fftw_wisdom_reuse,
                                  char*     wisdom_filename,
                                  )
    void fftw_execute(fftw_plan plan)
    void fftw_clean(double* grid, fftw_plan plan_forward,
                                  fftw_plan plan_backward)
""")



# Function for initializing and tabulating a cubic grid
# with vector values.
@cython.header(# Arguments
               gridsize='Py_ssize_t',
               func=func_dstar_ddd,
               factor='double',
               filename=str,
               # Locals
               dim='Py_ssize_t',
               grid='double[:, :, :, ::1]',
               grid_local='double[::1]',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               n_vectors='Py_ssize_t',
               n_vectors_local='Py_ssize_t',
               size='Py_ssize_t',
               size_edge='Py_ssize_t',
               size_face='Py_ssize_t',
               size_local='Py_ssize_t',
               start_local='Py_ssize_t',
               vector_value='double*',
               ℓ_local='Py_ssize_t',
               ℓ='Py_ssize_t',
               returns='double[:, :, :, ::1]',
               )
def tabulate_vectorgrid(gridsize, func, factor, filename=''):
    """This function tabulates a cubic grid of size
    gridsize*gridsize*gridsize with vector values computed by
    the function func, as
    grid[i, j, k] = func(i*factor, j*factor, k*factor).
    If filename is set, the tabulated grid is saved to a hdf5 file
    with this name.
    The grid is not distributed. Each process will end up with its own
    copy of the entire grid.
    The first point grid[0, 0, 0] corresponds to the origin of the box,
    while the last point grid[gridsize - 1, gridsize - 1, gridsize - 1]
    corresponds to the physical point ((gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize,
                                       (gridsize - 1)/gridsize*boxsize).
    This means that grid[gridsize, gridsize, gridsize] is the corner
    of the box the furthers away from the origin. This point is not
    represented in the grid because of the periodicity of the box,
    which means that physically, this point is the same as the origin.
    The mapping from grid indices to physical coordinates is thus
    (i, j, k) --> (i/gridsize*boxsize,
                   j/gridsize*boxsize,
                   k/gridsize*boxsize).
    Therefore, if you wish to tabulate a global (and periodic) field,
    extending over the entire box, the factor argument should be
    proportional to boxsize/gridsize. If you wish to only tabulate say
    one octant of the box, this volume is no more periodic, and so
    the factor should be proportional to 0.5*boxsize/(gridsize - 1).
    """
    # The number of elements in each point, edge, face and bulk
    # (the total), when the grid is viewed as a 3D box of vectors.
    size_edge = gridsize*3
    size_face = gridsize*size_edge
    size      = gridsize*size_face
    # Initialize the global grid to be
    # of shape gridsize*gridsize*gridsize*3.
    # That is, grid is not really cubic, but rather four-dimensional.
    grid = empty([gridsize, gridsize, gridsize, 3], dtype=C2np['double'])
    # Partition the grid fairly among the processes.
    # Each part is defined by a linear size and starting index.
    n_vectors = gridsize**3
    start_local, n_vectors_local = partition(n_vectors)
    # Each point really consists of three elements
    start_local *= 3
    size_local = 3*n_vectors_local
    # Make a linear array for storing the local part of the grid
    grid_local = empty(size_local, dtype=C2np['double'])
    # Tabulate the local grid
    for ℓ_local in range(0, size_local, 3):
        ℓ = start_local + ℓ_local
        # The global 3D indices
        i = ℓ//size_face
        ℓ -= i*size_face
        j = ℓ//size_edge
        ℓ -= j*size_edge
        k = ℓ//3
        # Fill grid_local
        vector_value = func(i*factor, j*factor, k*factor)
        for dim in range(3):
            grid_local[ℓ_local + dim] = vector_value[dim]
    # Gather the tabulated local grid parts into the common, global grid
    smart_mpi(grid_local, grid, mpifun='allgatherv')
    # Save grid to disk using parallel HDF5
    if filename:
        if master:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
        Barrier()
        with open_hdf5(filename, mode='w', driver='mpio', comm=comm) as hdf5_file:
            dset = hdf5_file.create_dataset('data', (size, ), dtype=C2np['double'])
            dset[start_local:(start_local + size_local)] = grid_local
    return grid

# Function for doing lookup in a grid with vector values
@cython.header(
    # Argument
    grid='double[:, :, :, :]',
    x='double',
    y='double',
    z='double',
    order='int',
    # Locals
    dim='int',
    i='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    index_i='Py_ssize_t',
    index_j='Py_ssize_t',
    index_k='Py_ssize_t',
    returns='double*',
)
def interpolate_in_vectorgrid(grid, x, y, z, order):
    """This function looks up tabulated vectors in a grid and
    interpolates to (x, y, z).
    Input arguments must be normalized so that
      0 <= x < grid.shape[0] - 1,
      0 <= y < grid.shape[1] - 1,
      0 <= z < grid.shape[2] - 1.
    It is assumed that the grid is nonperiodic (that is, the first and
    the last gridpoint in any dimension are physically distinct). The
    grid is not (necessarily) a domain grid and has no ghost points.
    This means that the interpolation will fail if using a high order
    and the coordinates are close to the boundary of the grid
    (grid.shape[dim] - 1). For coordinates within the legal range,
    first-order (NGP) and second-order (CIC) interpolation is
    always safe.
    """
    # Set interpolation weights and get grid indices
    if order == 1:  # NGP interpolation
        index_i = set_weights_NGP(x, weights_x)
        index_j = set_weights_NGP(y, weights_y)
        index_k = set_weights_NGP(z, weights_z)
    elif order == 2:  # CIC interpolation
        index_i = set_weights_CIC(x, weights_x)
        index_j = set_weights_CIC(y, weights_y)
        index_k = set_weights_CIC(z, weights_z)
    elif order == 3:  # TSC interpolation
        index_i = set_weights_TSC(x, weights_x)
        index_j = set_weights_TSC(y, weights_y)
        index_k = set_weights_TSC(z, weights_z)
    elif order == 4:  # PCS interpolation
        index_i = set_weights_PCS(x, weights_x)
        index_j = set_weights_PCS(y, weights_y)
        index_k = set_weights_PCS(z, weights_z)
    else:
        abort(
            f'interpolate_in_vectorgrid() called with order = {order} '
            f'∉ {{1 (NGP), 2 (CIC), 3 (TSC), 4 (PCS)}}'
        )
        index_i = index_j = index_k = 0  # To satisfy the compiler
    # Assign the weighted grid values to the vector components
    for dim in range(3):
        vector[dim] = 0
    for         i in range(order):
        for     j in range(order):
            for k in range(order):
                for dim in range(3):
                    vector[dim] += grid[
                        ℤ[index_i + i],
                        ℤ[index_j + j],
                        ℤ[index_k + k],
                        dim,
                    ]*ℝ[ℝ[ℝ[weights_x[i]]*weights_y[j]]*weights_z[k]]
    return vector
# Vector used as the return value of the
# interpolate_in_vectorgrid() function.
cython.declare(vector='double*')
vector = malloc(3*sizeof('double'))

# Function for doing lookup in a grid with scalar values and
# interpolating to specified coordinates.
@cython.header(
    # Argument
    grid='double[:, :, ::1]',
    component='Component',
    variable=str,
    dim='int',
    order='int',
    factor='double',
    # Locals
    cellsize='double',
    i='Py_ssize_t',
    index='Py_ssize_t',
    index_i='Py_ssize_t',
    index_j='Py_ssize_t',
    index_k='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    offset_x='double',
    offset_y='double',
    offset_z='double',
    posx='double*',
    posy='double*',
    posz='double*',
    ptr_dim='double*',
    value='double',
    x='double',
    y='double',
    z='double',
    returns='void',
)
def interpolate_domaingrid_to_particles(grid, component, variable, dim, order, factor=1):
    """This function updates the dim'th dimension of variable ('pos',
    'mom' or 'Δmom') of the component, through interpolation in the grid
    of a given order. If the grid values should be multiplied by a
    factor prior to adding them to the variable, this may be specified.
    """
    if variable == 'pos':
        ptr_dim = component.pos[dim]
    elif variable == 'mom':
        ptr_dim = component.mom[dim]
    elif variable == 'Δmom':
        ptr_dim = component.Δmom[dim]
    else:
        abort(
            f'interpolate_domaingrid_to_particles() called with variable = "{variable}" '
            f'∉ {{"pos", "mom"}}'
        )
    # Offsets and scalings needed for the interpolation
    cellsize = domain_size_x/(grid.shape[0] - ℤ[2*nghosts])  # We have cubic grid cells
    offset_x = domain_start_x - ℝ[(1 + machine_ϵ)*(nghosts - 0.5)*cellsize]
    offset_y = domain_start_y - ℝ[(1 + machine_ϵ)*(nghosts - 0.5)*cellsize]
    offset_z = domain_start_z - ℝ[(1 + machine_ϵ)*(nghosts - 0.5)*cellsize]
    # Interpolate onto each particle
    posx = component.posx
    posy = component.posy
    posz = component.posz
    for index in range(component.N_local):
        # Get, translate and scale the coordinates so that
        # nghosts - ½ < r < shape[r] - nghosts - ½ for r ∈ {x, y, z}.
        x = (posx[index] - offset_x)*ℝ[1/cellsize]
        y = (posy[index] - offset_y)*ℝ[1/cellsize]
        z = (posz[index] - offset_z)*ℝ[1/cellsize]
        # Set interpolation weights and get grid indices
        with unswitch:
            if order == 1:  # NGP interpolation
                index_i = set_weights_NGP(x, weights_x)
                index_j = set_weights_NGP(y, weights_y)
                index_k = set_weights_NGP(z, weights_z)
            elif order == 2:  # CIC interpolation
                index_i = set_weights_CIC(x, weights_x)
                index_j = set_weights_CIC(y, weights_y)
                index_k = set_weights_CIC(z, weights_z)
            elif order == 3:  # TSC interpolation
                index_i = set_weights_TSC(x, weights_x)
                index_j = set_weights_TSC(y, weights_y)
                index_k = set_weights_TSC(z, weights_z)
            elif order == 4:  # PCS interpolation
                index_i = set_weights_PCS(x, weights_x)
                index_j = set_weights_PCS(y, weights_y)
                index_k = set_weights_PCS(z, weights_z)
            else:
                abort(
                    f'interpolate_domaingrid_to_particles() called with order = {order} '
                    f'∉ {{1 (NGP), 2 (CIC), 3 (TSC), 4 (PCS)}}'
                )
                index_i = index_j = index_k = 0  # To satisfy the compiler
        # Apply the update
        value = 0
        for         i in range(order):
            for     j in range(order):
                for k in range(order):
                    value += grid[
                        ℤ[index_i + i],
                        ℤ[index_j + j],
                        ℤ[index_k + k],
                    ]*ℝ[ℝ[weights_x[i]]*weights_y[j]]*weights_z[k]
        with unswitch:
            if factor != 1:
                value *= factor
        ptr_dim[index] += value

# Function for interpolating a certain quantity from components
# (particles and fluids) onto global domain grids using intermediate
# upstream grids and pixel mixing.
@cython.pheader(
    # Arguments
    components=list,
    gridsizes_upstream=list,
    gridsize_global='Py_ssize_t',
    quantity=str,
    order='int',
    include_shifted_particles='bint',
    ᔑdt=dict,
    do_ghost_communication='bint',
    # Locals
    component='Component',
    grid_global='double[:, :, ::1]',
    grid_upstream='double[:, :, ::1]',
    grids_global=dict,
    grids_upstream=dict,
    gridshape_local=tuple,
    gridsize_upstream='Py_ssize_t',
    group=list,
    groups=object,  # collections.defaultdict(list)
    representation=str,
    returns=dict,
)
def interpolate_upstream(
    components, gridsizes_upstream, gridsize_global, quantity, order,
    include_shifted_particles=False, ᔑdt=None, do_ghost_communication=True,
):
    """Given a list of components, a list of corresponding upstream grid
    sizes and a single global grid size, this function interpolates the
    components onto global grids by first interpolating directly onto
    the upstream grids, which are then pixel mixed onto the global
    grids.
    See the interpolate_components() function for the remaining
    arguments and the return value.
    """
    # Group components according to their upstream grid size
    grids_global = {}
    gridshape_local = get_gridshape_local(gridsize_global)
    groups = collections.defaultdict(list)
    for component, gridsize_upstream in zip(components, gridsizes_upstream):
        groups[gridsize_upstream].append(component)
        # Instantiate the needed global grids
        if component.representation not in grids_global:
            grids_global[component.representation] = get_buffer(
                gridshape_local, f'global_grid_{component.representation}',
                nullify=True,
            )
            if include_shifted_particles and component.representation == 'particles':
                grids_global['particles_shifted'] = get_buffer(
                    gridshape_local, 'global_grid_particles_shifted',
                    nullify=True,
                )
    # Interpolate components onto global grids
    for gridsize_upstream, group in groups.items():
        if gridsize_upstream == gridsize_global:
            # The upstream grid size is equal to the global grid size,
            # and so we can interpolate the group of components
            # directly onto the global grids.
            interpolate_components(
                group, quantity, gridsize_upstream, order,
                include_shifted_particles, ᔑdt, grids_global,
                do_ghost_communication=False,
            )
        else:
            # Interpolate the group of components onto upstream grids
            grids_upstream = interpolate_components(
                group, quantity, gridsize_upstream, order,
                include_shifted_particles, ᔑdt,
                do_ghost_communication=False,
            )
            # Pixel mix upstream grids onto global grids
            for representation, grid_upstream in grids_upstream.items():
                # Particle grids need to have their ghost values
                # added to their proper grid positions.
                if representation.startswith('particles'):
                    communicate_ghosts(grid_upstream, '+=')
                # Now do the pixel mixing
                interpolate_grid_to_grid(grid_upstream, grids_global[representation])
    # Communicate ghost points
    for representation, grid_global in grids_global.items():
        # We only need to add the current ghost values in the particles
        # slabs to their actual locations ('+='). No copying of values
        # to the ghosts ('=') is needed, for particles or fluid slabs.
        if representation.startswith('particles'):
            communicate_ghosts(grid_global, '+=')
        # Further populate ghost points if requested
        with unswitch:
            if do_ghost_communication:
                communicate_ghosts(grid_global, '=')
    # Return the global grids only
    return grids_global

# Function for interpolating a certain quantity from components
# (particles and fluids) onto domain grids.
@cython.pheader(
    # Arguments
    component_or_components=object,  # Component or list of Components
    quantity=str,
    gridsize='Py_ssize_t',
    order='int',
    include_shifted_particles='bint',
    ᔑdt=dict,
    grids=dict,
    do_ghost_communication='bint',
    # Locals
    any_fluid='bint',
    any_particles='bint',
    any_particles_shifted='bint',
    components=list,
    component='Component',
    fluid_components=list,
    grid='double[:, :, ::1]',
    gridshape_local=tuple,
    particle_components=list,
    representation=str,
    returns=dict,
)
def interpolate_components(
    component_or_components, quantity, gridsize, order,
    include_shifted_particles=False, ᔑdt=None, grids=None, do_ghost_communication=True,
):
    """This function interpolates a specified quantity of components to
    a domain grid. A dict of the form
    {
        'particles'        : double[:, :, ::1],
        'fluid'            : double[:, :, ::1],
        'particles_shifted': double[:, :, ::1],
    }
    is returned, storing separate grids for particles and fluid
    components, as well as for "shifted" particle components (more on
    that later). If any of these three are not present, the returned
    dict will not contain them either.

    The gridsize determines the global size of the grids. Reardless of
    gridsize, the same persistent chunk of memory will be used, meaning
    that you should never call this function before you are done with
    the grids returned in the previous call.
    If at the same time a dict of domain grids is supplied using the
    grids argument (in the format shown for the return value above), the
    components will be interpolated (and added) onto these grids instead
    of using the persisent buffers.

    For particle components, the interpolation scheme is determined by
    the order argument (1 for NGP, 2 for CIC, 3 for TSC, 4 for PCS).
    For fluid components, the interpolation is carried out using the
    "pixel mixing" scheme.

    For the 'particles_shifted' grid to appear in the output dict, you
    must set include_shifted_particles to True. This grid contains the
    interpolated particle quantity, but with grid points shifted by half
    a grid cell (0.5*boxsize/gridsize) for all three dimensions.

    If do_ghost_communication is True, the returned grids will have
    correctly set and populated boundary and ghost cells. If not, the
    boundary cells of the particle grids will contain data that needs to
    be communicated, and none of the grids will have correctly populated
    ghost cells.

    The quantity argument determines what should be interpolated onto
    the grid(s). Valid values are:
    - 'ρ': The returned grid(s) will hold physical densities. Note that
      ρ = a**(-3(1 + w_eff))*ϱ. Note that this physical 'ρ' is always
      preferable to the conserved 'ϱ' when multiple components are to be
      interpolated together, as only 'ρ' is additive across
      components/species. Each particle will contribute with
      a**(-3*w_eff)*mass/V_cell, a**(-3*w_eff)*mass being the current
      mass of the particle (a**(-3*w_eff) taking decay into account) and
      V_cell = (a*boxsize/gridsize)**3 being the physical grid cell
      volume. In total, each particle contribute with
      a**(-3*(1 + w_eff))*(gridsize/boxsize)**3*mass. Each fluid cell
      will contribute with a**(-3*(1 + w_eff))*ϱᵢⱼₖ*V_cell_fluid/V_cell,
      where a**(-3*(1 + w_eff))*ϱᵢⱼₖ = ρᵢⱼₖ is the physical density of
      fluid cell [i, j, k] and
      V_cell_fluid = (a*boxsize/gridsize_fluid)**3 is the physical cell
      volume of the fluid grid. In total, each fluid cell contribute
      with a**(-3*(1 + w_eff))*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ
    - 'a²ρ': The returned grid(s) will hold physical densities times the
      square of the scale factor. From the 'ρ' entry above, we then have
      that each particle will contribute with
      a**(-3*w_eff - 1)*(gridsize/boxsize)**3*mass and that each fluid
      cell will contribute with
      a**(-3*w_eff - 1)*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    - 'ϱ': The returned grid(s) will hold the conserved densities. From
      the 'ρ' entry above, we then have that each particle will
      contribute with ϱ = (gridsize/boxsize)**3*mass and that each fluid
      cell will contribute with (gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    - 'Jx': The returned grid(s) will hold the conserved momentum
      density Jₓ = a**4*(ρ + c⁻²P)*uₓ. As this is built from physical
      quantities, this is additive across components/species. For
      particles we set P = 0, leaving Jₓ = a**4*ρ*uₓ. The canonical
      momentum momₓ and peculiar velocity uₓ is related by
      momₓ = a*(a**(-3*w_eff)*mass)*uₓ, and so from the particle
      construction of ρ above we get that each particle contribute with
      (gridsize/boxsize)**3*momₓ. As each fluid cell already stores Jₓ,
      they contribute by
      V_cell_fluid/V_cell*Jₓ = (gridsize/gridsize_fluid)**3*Jₓ.
    - 'Jy': Similar to Jx.
    - 'Jz': Similar to Jx.
    In all of the above, expressions involving a = a(t) will be
    evaluated at the current universal time, unless a dict ᔑdt of time
    step integrals is passed, in which case the expressions will be
    integrated over the time step.
    """
    if isinstance(component_or_components, list):
        components = component_or_components
    else:
        components = [component_or_components]
    # Separate out the components and make a flag
    # for each of the three grid types.
    particle_components = [
        component for component in components if component.representation == 'particles'
    ]
    fluid_components = [
        component for component in components if component.representation == 'fluid'
    ]
    any_particles = bool(particle_components)
    any_fluid = bool(fluid_components)
    any_particles_shifted = (any_particles and include_shifted_particles)
    # Fetch grids as needed
    if grids is None:
        grids = {}
        gridshape_local = get_gridshape_local(gridsize)
        if any_particles:
            grids['particles'] = get_buffer(
                gridshape_local, 'grid_particles', nullify=True,
            )
        if any_fluid:
            grids['fluid'] = get_buffer(
                gridshape_local, 'grid_fluid', nullify=True,
            )
        if any_particles_shifted:
            grids['particles_shifted'] = get_buffer(
                gridshape_local, 'grid_particles_shifted', nullify=True,
            )
    # Interpolate particle components
    for component in particle_components:
        interpolate_particles(
            component, gridsize, grids['particles'], quantity, order, ᔑdt,
            do_ghost_communication=False,
        )
        with unswitch:
            if any_particles_shifted:
                interpolate_particles(
                    component, gridsize, grids['particles_shifted'], quantity, order, ᔑdt,
                    shift=0.5, do_ghost_communication=False,
                )
    # Interpolate fluid components
    for component in fluid_components:
        interpolate_fluid(component, grids['fluid'], quantity, ᔑdt)
    # Add ghost point contributions from the above interpolations
    if do_ghost_communication:
        for representation, grid in grids.items():
            if representation.startswith('particles'):
                communicate_ghosts(grid, '+=')
            communicate_ghosts(grid, '=')
    return grids

# Function for interpolating a certain quantity from a particle
# component onto a supplied domain grid.
@cython.pheader(
    # Arguments
    component='Component',
    gridsize='Py_ssize_t',
    grid='double[:, :, ::1]',
    quantity=str,
    order='int',
    ᔑdt=dict,
    shift='double',
    do_ghost_communication='bint',
    # Locals
    a='double',
    cellsize='double',
    constant_contribution='bint',
    contribution='double',
    contribution_factor='double',
    contribution_ptr='double*',
    dim='int',
    i='Py_ssize_t',
    index='Py_ssize_t',
    index_i='Py_ssize_t',
    index_j='Py_ssize_t',
    index_k='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    offset_x='double',
    offset_y='double',
    offset_z='double',
    posx='double*',
    posy='double*',
    posz='double*',
    w_eff='double',
    x='double',
    y='double',
    z='double',
    returns='void',
)
def interpolate_particles(component, gridsize, grid, quantity, order, ᔑdt,
    shift=0, do_ghost_communication=True):
    """The given quantity of the component will be added to current
    content of the local grid with global grid size given by gridsize.
    For info about the quantity argument, see the
    interpolate_components() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    The supplied grid should contain ghost layers, as the interpolation
    will populate these. To communicate and add the resulting values in
    the ghost cells to their physical cells, set do_ghost_communication
    to True. Note that even with do_ghost_communication set to True, the
    ghost cells will not end up with copies of the boundary values.
    """
    # Always use the current time
    a = universals.a
    w_eff = component.w_eff(a=a)
    # Determine the contribution of each particle based on the quantity
    contribution = 1
    if quantity == 'ρ':
        constant_contribution = True
        if ᔑdt:
            contribution = ᔑdt['a**(-3*(1+w_eff))', component.name]/ᔑdt['1']
        else:
            contribution = a**(-3*(1 + w_eff))
        contribution *= component.mass
    elif quantity == 'a²ρ':
        constant_contribution = True
        if ᔑdt:
            contribution = ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
        else:
            contribution = a**(-3*w_eff - 1)
        contribution *= component.mass
    elif quantity == 'ϱ':
        constant_contribution = True
        contribution = component.mass
    elif quantity in {'Jx', 'Jy', 'Jz'}:
        constant_contribution = False
        dim = 'xyz'.index(quantity[1])
        contribution_ptr = component.mom[dim]
    else:
        abort(
            f'interpolate_particles() called with '
            f'quantity = "{quantity}" ∉ {{"ρ", "a²ρ", "ϱ", "Jx", "Jy", "Jz"}}'
        )
    contribution_factor = (gridsize/boxsize)**3
    contribution *= contribution_factor
    # Offsets and scalings needed for the interpolation
    cellsize = boxsize/gridsize
    offset_x = domain_start_x - ℝ[(1 + machine_ϵ)*(nghosts - 0.5 + shift)*cellsize]
    offset_y = domain_start_y - ℝ[(1 + machine_ϵ)*(nghosts - 0.5 + shift)*cellsize]
    offset_z = domain_start_z - ℝ[(1 + machine_ϵ)*(nghosts - 0.5 + shift)*cellsize]
    # Interpolate each particle
    posx = component.posx
    posy = component.posy
    posz = component.posz
    for index in range(component.N_local):
        # Get the total contribution from this particle
        with unswitch:
            if not constant_contribution:
                contribution = contribution_factor*contribution_ptr[index]
        # Get, translate and scale the coordinates so that
        # nghosts - ½ < r < shape[r] - nghosts - ½ for r ∈ {x, y, z}.
        x = (posx[index] - offset_x)*ℝ[1/cellsize]
        y = (posy[index] - offset_y)*ℝ[1/cellsize]
        z = (posz[index] - offset_z)*ℝ[1/cellsize]
        # Set interpolation weights and get grid indices
        with unswitch:
            if order == 1:  # NGP interpolation
                index_i = set_weights_NGP(x, weights_x)
                index_j = set_weights_NGP(y, weights_y)
                index_k = set_weights_NGP(z, weights_z)
            elif order == 2:  # CIC interpolation
                index_i = set_weights_CIC(x, weights_x)
                index_j = set_weights_CIC(y, weights_y)
                index_k = set_weights_CIC(z, weights_z)
            elif order == 3:  # TSC interpolation
                index_i = set_weights_TSC(x, weights_x)
                index_j = set_weights_TSC(y, weights_y)
                index_k = set_weights_TSC(z, weights_z)
            elif order == 4:  # PCS interpolation
                index_i = set_weights_PCS(x, weights_x)
                index_j = set_weights_PCS(y, weights_y)
                index_k = set_weights_PCS(z, weights_z)
            else:
                abort(
                    f'interpolate_particles() called with order = {order} '
                    f'∉ {{1 (NGP), 2 (CIC), 3 (TSC), 4 (PCS)}}'
                )
                index_i = index_j = index_k = 0  # To satisfy the compiler
        # Assign weighted contributions to grid points
        for         i in range(order):
            for     j in range(order):
                for k in range(order):
                    grid[
                        ℤ[index_i + i],
                        ℤ[index_j + j],
                        ℤ[index_k + k],
                    ] += ℝ[ℝ[contribution*weights_x[i]]*weights_y[j]]*weights_z[k]
    # All particles interpolated. Some may have gotten interpolated
    # partly onto ghost points, which then need to be communicated.
    if do_ghost_communication:
        communicate_ghosts(grid, '+=')

# Function for interpolating a certain quantity from a fluid component
# onto a supplied domain grid.
@cython.header(
    # Arguments
    component='Component',
    grid='double[:, :, ::1]',
    quantity=str,
    ᔑdt=dict,
    # Locals
    a='double',
    contribution_factor='double',
    dim='int',
    fluidscalar='FluidScalar',
    w_eff='double',
    returns='void',
)
def interpolate_fluid(component, grid, quantity, ᔑdt):
    """The component has to be a fluid component, and the passed grid is
    interpreted as having the same physical extent as that of the fluid
    (domain) grids. The grid passed should be a full local domain grid,
    including ghost points.
    The given quantity of the component will be added to current
    content of the local grid. For info about the quantity argument,
    see the interpolate_components() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    Though ghost cells are required to exist, these are not actually
    touched by this function, and will not be properly set either.
    """
    # Always use the current time
    a = universals.a
    w_eff = component.w_eff(a=a)
    # Determine the contribution factor of each fluid cell
    # based on the quantity.
    contribution_factor = 1
    if quantity == 'ρ':
        if ᔑdt:
            contribution_factor = ᔑdt['a**(-3*(1+w_eff))', component.name]/ᔑdt['1']
        else:
            contribution_factor = a**(-3*(1 + w_eff))
        fluidscalar = component.ϱ
    elif quantity == 'a²ρ':
        if ᔑdt:
            contribution_factor = ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
        else:
            contribution_factor = a**(-3*w_eff - 1)
        fluidscalar = component.ϱ
    elif quantity == 'ϱ':
        fluidscalar = component.ϱ
    elif quantity in {'Jx', 'Jy', 'Jz'}:
        dim = 'xyz'.index(quantity[1])
        fluidscalar = component.J[dim]
    else:
        abort(
            f'interpolate_fluid() called with '
            f'quantity = "{quantity}" ∉ {{"ρ", "a²ρ", "ϱ", "Jx", "Jy", "Jz"}}'
        )
    # Add values of the FluidScalar grid (multiplied by the
    # contribution_factor) to the current values of the grid,
    # through interpolation.
    interpolate_grid_to_grid(fluidscalar.grid_mv, grid, factor=contribution_factor)

# Function for interpolating one grid onto another
@cython.pheader(
    # Arguments
    grid='double[:, :, ::1]',
    buffer_or_buffer_name=object,  # double[:, :, ::1] or int or str
    gridsize_buffer='Py_ssize_t',
    factor='double',
    # Locals
    buffer='double[:, :, ::1]',
    buffer_ptr='double*',
    buffer_supplied='bint',
    factor_x='double',
    factor_y='double',
    factor_z='double',
    grid_ptr='double*',
    gridshape_buffer_local=tuple,
    gridsize_grid='Py_ssize_t',
    i='Py_ssize_t',
    i_box='Py_ssize_t',
    index_and_size='Py_ssize_t*',
    index_i='Py_ssize_t',
    index_j='Py_ssize_t',
    index_k='Py_ssize_t',
    j='Py_ssize_t',
    j_box='Py_ssize_t',
    k='Py_ssize_t',
    k_box='Py_ssize_t',
    scaling='double',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    value='double',
    weights_x='double*',
    weights_y='double*',
    weights_z='double*',
    returns='double[:, :, ::1]',
)
def interpolate_grid_to_grid(grid, buffer_or_buffer_name=0, gridsize_buffer=-1, factor=1):
    """The grid will be interpolated onto the buffer, using the
    "pixel mixing" scheme (or "voxel mixing" as we are in 3D). Note that
    the physical quantity on the grid should be some kind of density
    (e.g. ρ, ϱ, J), as opposed to say mass. Otherwise, the rescaling of
    the grid will not conserve the corresponding integrated quantity.
    Both the grid and the buffer should contain ghost cells, though
    ghost cells will not be touched or populated by this function. The
    buffer may be explicitly supplied as a grid, or alternatively as a
    buffer name/number. In the latter case, the global grid size of the
    buffer should be passed as well. If a buffer is supplied, the
    interpolated values will be added to existing values in the buffer.
    If not, the retrieved buffer will be nullified before the
    interpolated values are added. The interpolated values will be
    multiplied by factor before they are added. Note that if a buffer is
    not passed and the grid sizes of the grid and the buffer are
    identical, the supplied grid will be mutated according to factor
    and returned.
    """
    # If the gridsize of the grid and the buffer are identical, no
    # interpolation is needed and we may return the grid itself, but
    # multiplied by the factor. If a buffer is explicitly passed
    # however, we copy the values of the grid onto the buffer and return
    # the buffer.
    buffer_supplied = (not isinstance(buffer_or_buffer_name, (int, str)))
    if buffer_supplied:
        buffer = buffer_or_buffer_name
        gridsize_buffer = (buffer.shape[0] - ℤ[2*nghosts])*domain_subdivisions[0]
    gridsize_grid = (grid.shape[0] - ℤ[2*nghosts])*domain_subdivisions[0]
    if gridsize_grid == gridsize_buffer:
        grid_ptr = cython.address(grid[:, :, :])
        if buffer_supplied:
            buffer_ptr = cython.address(buffer[:, :, :])
            for i in range(grid.shape[0]*grid.shape[1]*grid.shape[2]):
                with unswitch:
                    if factor != 1:
                        buffer_ptr[i] += factor*grid_ptr[i]
                    else:
                        buffer_ptr[i] += grid_ptr[i]
            return buffer
        else:
            if factor != 1:
                for i in range(grid.shape[0]*grid.shape[1]*grid.shape[2]):
                    grid_ptr[i] *= factor
            return grid
    # If no buffer is supplied, fetch the buffer with the name
    # given by buffer_or_buffer_name and a global grid size given by
    # gridsize_buffer.
    if not buffer_supplied:
        gridshape_buffer_local = get_gridshape_local(gridsize_buffer)
        buffer = get_buffer(gridshape_buffer_local, buffer_or_buffer_name, nullify=True)
    # The scaling factor between the old and new grid size. Note that
    # this is the same in all dimensions.
    scaling = float(gridsize_buffer)/gridsize_grid
    # Allocate arrays for storing cell weights. Up to int(scaling)
    # cells in the new grid (the buffer) may reside completely within a
    # given single cell of the original grid. In addition, it may happen
    # that a fraction of a new cell is partly covered by the same old
    # cell, for both directions. For each dimension then, an old cell
    # may be distributed among at most 2 + int(scaling) new cells.
    weights_x = malloc(ℤ[2 + int(scaling + machine_ϵ)]*sizeof('double'))
    weights_y = malloc(ℤ[2 + int(scaling + machine_ϵ)]*sizeof('double'))
    weights_z = malloc(ℤ[2 + int(scaling + machine_ϵ)]*sizeof('double'))
    # For each cell in the original grid, we need to know the number of
    # elements of the weights arrays in use (i.e. how many new cells the
    # old cell covers), as well as the index of the first covered new
    # grid cell. Allocate common arrays for both of these.
    index_and_size = malloc(2*sizeof('Py_ssize_t'))
    # Loop over the bulk (i.e. not ghosts) or the original grid
    # and distribute the value of each cell amongst the overlapping
    # cells in the buffer.
    for i in range(ℤ[grid.shape[0] - ℤ[2*nghosts]]):
        set_weights_pixelmixing(i, scaling, weights_x, index_and_size)
        index_i = index_and_size[0]
        size_i  = index_and_size[1]
        for j in range(ℤ[grid.shape[1] - ℤ[2*nghosts]]):
            set_weights_pixelmixing(j, scaling, weights_y, index_and_size)
            index_j = index_and_size[0]
            size_j  = index_and_size[1]
            for k in range(ℤ[grid.shape[2] - ℤ[2*nghosts]]):
                set_weights_pixelmixing(k, scaling, weights_z, index_and_size)
                index_k = index_and_size[0]
                size_k  = index_and_size[1]
                value = grid[ℤ[nghosts + i], ℤ[nghosts + j], ℤ[nghosts + k]]
                with unswitch:
                    if factor != 1:
                        value *= factor
                # Loop over box in new grid corresponding to old cell
                for i_box in range(size_i):
                    factor_x = weights_x[i_box]
                    for j_box in range(size_j):
                        factor_y = weights_y[j_box]
                        for k_box in range(size_k):
                            factor_z = weights_z[k_box]
                            buffer[
                                ℤ[ℤ[nghosts + index_i] + i_box],
                                ℤ[ℤ[nghosts + index_j] + j_box],
                                ℤ[ℤ[nghosts + index_k] + k_box],
                            ] += ℝ[ℝ[value*factor_x]*factor_y]*factor_z
    # Cleanup and return result
    free(weights_x)
    free(weights_y)
    free(weights_z)
    free(index_and_size)
    return buffer

# Function for computing weights and indices for
# "pixel mixing" interpolation.
@cython.header(
    # Arguments
    i='Py_ssize_t',
    scaling='double',
    weights='double*',
    index_and_size='Py_ssize_t*',
    # Locals
    i_bgn='Py_ssize_t',
    i_end='Py_ssize_t',
    i_new='Py_ssize_t',
    i_new_float='double',
    size='Py_ssize_t',
    x='double',
    returns='void',
)
def set_weights_pixelmixing(i, scaling, weights, index_and_size):
    # Compute range of indices into the new array corresponding to
    # index i of the old array.
    x = (i + machine_ϵ)*scaling
    i_bgn = int(x)
    i_end = int(x + scaling)
    # Set cell weights
    size = 0
    for i_new in range(i_bgn, i_end + 1):
        i_new_float = float(i_new)
        if i_new_float >= ℝ[(i + 1)*scaling]:
            # This new cell is completely outside of the old cell,
            # and any further new cells will be outside as well.
            break
        if i_new_float >= ℝ[i*scaling]:
            if ℝ[i_new_float + 1] <= ℝ[(i + 1)*scaling]:
                # This new cell is completely inside the old cell.
                weights[size] = 1
            else:
                # This new cell sticks out of the old cell at the end
                weights[size] = ℝ[(i + 1)*scaling] - i_new_float
        elif ℝ[i_new_float + 1] <= ℝ[(i + 1)*scaling]:
            # This new cell sticks out of the old cell at the beginning
            weights[size] = 1 - (ℝ[i*scaling] - i_new_float)
        else:
            # This new cell contains the entire old cell
            weights[size] = scaling
        size += 1
    # Store first index into the new grid and number of new cells
    # affected by this old ell.
    index_and_size[0] = i_bgn
    index_and_size[1] = size

# Function for converting particles of a particle component to fluid
# grids, effectively changing the representation of the component.
@cython.header(
    # Arguments
    component='Component',
    order='int',
    # Locals
    J_dim='FluidScalar',
    N_vacuum='Py_ssize_t',
    N_vacuum_originally='Py_ssize_t',
    dim='int',
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    original_representation=str,
    shape=tuple,
    vacuum_sweep='Py_ssize_t',
    Δϱ_each='double',
    ϱ='double[:, :, ::1]',
    ᔑdt=dict,
    returns='Py_ssize_t',
)
def convert_particles_to_fluid(component, order):
    """This function interpolates particle positions and momenta onto
    fluid grids, effectively converting from a 'particles'
    representation to a 'fluid' representation. The mass attribute of
    the passed component should be the particle mass, not the average
    fluid element mass. The value of the representation attribute does
    not matter and will not be altered. The size of the fluid grids are
    determined by component.gridsize. To save memory, the particle data
    will be freed (resized to a minimum size) during the process.
    """
    # Backup of original representation
    original_representation = component.representation
    # Instantiate fluid grids spanning the local domains.
    # The newly allocated grids will be nullified.
    component.representation = 'fluid'
    shape = tuple([component.gridsize//ds for ds in domain_subdivisions])
    if any([component.gridsize != domain_subdivisions[dim]*shape[dim] for dim in range(3)]):
        abort(
            f'The grid size of {component.name} is {component.gridsize} '
            f'which cannot be equally shared among {nprocs} processes'
        )
    component.resize(shape)  # This also nullifies all fluid grids
    # Do the particle -> fluid interpolation
    ᔑdt = {}
    gridsize = component.gridsize
    ϱ = component.ϱ.grid_mv
    interpolate_particles(component, gridsize, ϱ, 'ϱ', order, ᔑdt)
    for dim in range(3):
        J_dim = component.J[dim]
        interpolate_particles(component, gridsize, J_dim.grid_mv, 'J' + 'xyz'[dim], order, ᔑdt)
    # The interpolation may have left some cells empty. Count up the
    # number of such vacuum cells and add to each a density of
    # ρ_vacuum, while leaving the momentum at zero. This will increase
    # the total mass, which then has to be lowered again, which we do
    # by subtracting a constant amount from each cell. This subtraction
    # may itself produce vacuum cells, and so we need to repeat until
    # no vacuum is detected.
    for vacuum_sweep in range(gridsize):
        # Count up and assign to vacuum cells
        N_vacuum = 0
        for         i in range(nghosts, ℤ[ϱ.shape[0] - nghosts]):
            for     j in range(nghosts, ℤ[ϱ.shape[1] - nghosts]):
                for k in range(nghosts, ℤ[ϱ.shape[2] - nghosts]):
                    if ϱ[i, j, k] < ρ_vacuum:
                        N_vacuum += 1
                        ϱ[i, j, k] += ρ_vacuum
        N_vacuum = allreduce(N_vacuum, op=MPI.SUM)
        # Remember the original number of vacuum cells
        if vacuum_sweep == 0:
            N_vacuum_originally = N_vacuum
        # We are done when no vacuum is left
        if N_vacuum == 0:
            break
        # Ensure mass conservation
        Δϱ_each = N_vacuum*ℝ[ρ_vacuum/gridsize**3]
        for         i in range(nghosts, ℤ[ϱ.shape[0] - nghosts]):
            for     j in range(nghosts, ℤ[ϱ.shape[1] - nghosts]):
                for k in range(nghosts, ℤ[ϱ.shape[2] - nghosts]):
                    ϱ[i, j, k] -= Δϱ_each
    else:
        # Failed to remove vacuum
        masterwarn(
            'The convert_particles_to_fluid() function was unable to '
            'get rid of vacuum cells in the fluid after interpolation'
        )
    # Populate ghost points of all fluid grids
    component.communicate_fluid_grids('=')
    # The particle data is no longer needed. Free it to save memory.
    component.representation = 'particles'
    component.resize(1)
    # Re-insert the original representation and return
    # the original number of vacuum cells.
    component.representation = original_representation
    return N_vacuum_originally

# Function for getting the shape of a local grid, which is part of a
# global, cubic grid with some gridsize.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    dim='int',
    gridshape_local=tuple,
    returns=tuple,
)
def get_gridshape_local(gridsize):
    # Cache lookup
    gridshape_local = gridshape_local_cache.get(gridsize)
    if gridshape_local is not None:
        return gridshape_local
    # The global grid will be cut into domains according to the
    # domain_subdivisions. The cut along each dimension has to leave
    # the local grids with integer gridsize.
    for dim in range(3):
        if gridsize%domain_subdivisions[dim] != 0:
            abort(
                f'A grid of global gridsize {gridsize} is to be distributed '
                f'across the processes, but {gridsize}×{gridsize}×{gridsize} '
                f'cannot be divided according to the domain decomposition '
                f'{domain_subdivisions[0]}×{domain_subdivisions[1]}×{domain_subdivisions[2]}.'
            )
    # We have nghosts ghost points on both sides of the local grid,
    # for all dimensions.
    gridshape_local = tuple([
        gridsize//domain_subdivisions[dim] + ℤ[2*nghosts] for dim in range(3)
    ])
    for dim in range(3):
        if gridshape_local[dim] < ℤ[4*nghosts]:
            abort(
                f'A grid of shape {gridshape_local} (or '
                f'{asarray(gridshape_local) - 2*nghosts} without ghosts) was encountered '
                f'in get_gridshape_local(), but all domain grids must have at least twice '
                f'as many grid points across each dimension as the number of ghost layers '
                f'nghosts = {nghosts}.'
            )
    # Store result in cache and return
    gridshape_local_cache[gridsize] = gridshape_local
    return gridshape_local
# Cache used by the get_local_local function
cython.declare(gridshape_local_cache=dict)
gridshape_local_cache = {}

# Function that compute a lot of information needed by the
# slab_decompose and domain_decompose functions.
@cython.header(
    # Arguments
    domain_grid='double[:, :, ::1]',
    slab='double[:, :, ::1]',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    domain_end_i='Py_ssize_t',
    domain_end_j='Py_ssize_t',
    domain_end_k='Py_ssize_t',
    domain_grid_noghosts='double[:, :, :]',
    domain_grid_shape=tuple,
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain_size_i='Py_ssize_t',
    domain_size_j='Py_ssize_t',
    domain_size_k='Py_ssize_t',
    domain_start_i='Py_ssize_t',
    domain_start_j='Py_ssize_t',
    domain_start_k='Py_ssize_t',
    domain2slabs_recvsend_ranks='int[::1]',
    index='Py_ssize_t',
    info=tuple,
    rank_recv='int',
    rank_send='int',
    recvtuple=tuple,
    sendtuple=tuple,
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slab_shape=tuple,
    slab_size_i='Py_ssize_t',
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns=tuple,
)
def prepare_decomposition(domain_grid, slab):
    # Simply look up and return the needed information
    # if previously computed.
    domain_grid_shape = asarray(domain_grid).shape
    slab_shape = asarray(slab).shape
    info = decomposition_info.get((domain_grid_shape, slab_shape))
    if info:
        return info
    # Memoryview of the domain grid without the ghost layers
    domain_grid_noghosts = domain_grid[
        nghosts:(domain_grid.shape[0] - nghosts),
        nghosts:(domain_grid.shape[1] - nghosts),
        nghosts:(domain_grid.shape[2] - nghosts),
    ]
    # The size (number of grid points) of the truly local part of the
    # domain grid, excluding ghost layers, for each dimension.
    domain_size_i = domain_grid_noghosts.shape[0]
    domain_size_j = domain_grid_noghosts.shape[1]
    domain_size_k = domain_grid_noghosts.shape[2]
    # The global start and end indices of the local domain
    # in the global grid.
    domain_start_i = domain_layout_local_indices[0]*domain_size_i
    domain_start_j = domain_layout_local_indices[1]*domain_size_j
    domain_start_k = domain_layout_local_indices[2]*domain_size_k
    domain_end_i = domain_start_i + domain_size_i
    domain_end_j = domain_start_j + domain_size_j
    domain_end_k = domain_start_k + domain_size_k
    # When in real space, the slabs are distributed over the first
    # dimension. Give the size of the slab in this dimension a name.
    slab_size_i = slab.shape[0]
    # Find local i-indices to send and to which process by
    # shifting a piece of the number line in order to match
    # the communication pattern used.
    domain_sendrecv_i_start = np.roll(asarray([ℓ - domain_start_i
                                               for ℓ in range(domain_start_i,
                                                               domain_end_i,
                                                               slab_size_i)
                                               ], dtype=C2np['int']),
                                      -rank)
    domain_sendrecv_i_end = np.roll(asarray([ℓ - domain_start_i + slab_size_i
                                             for ℓ in range(domain_start_i,
                                                             domain_end_i,
                                                             slab_size_i)
                                             ], dtype=C2np['int']),
                                    -rank)
    slabs2domain_sendrecv_ranks = np.roll(asarray([ℓ//slab_size_i
                                                  for ℓ in range(domain_start_i,
                                                                  domain_end_i,
                                                                  slab_size_i)],
                                                  dtype=C2np['int']),
                                     -rank)
    # Communicate the start and end (j, k)-indices of the slab,
    # where parts of the local domains should be received into.
    slab_sendrecv_j_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_start       = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_j_end         = empty(nprocs, dtype=C2np['int'])
    slab_sendrecv_k_end         = empty(nprocs, dtype=C2np['int'])
    domain2slabs_recvsend_ranks = empty(nprocs, dtype=C2np['int'])
    index = 0
    for ℓ in range(nprocs):
        # Process ranks to send/receive to/from
        rank_send = mod(rank + ℓ, nprocs)
        rank_recv = mod(rank - ℓ, nprocs)
        # Send the global y and z start and end indices of the domain
        # to be send, if anything should be send to process rank_send.
        # Otherwise send None.
        sendtuple = ((domain_start_j, domain_start_k, domain_end_j, domain_end_k)
                     if rank_send in asarray(slabs2domain_sendrecv_ranks) else None)
        recvtuple = sendrecv(sendtuple, dest=rank_send, source=rank_recv)
        if recvtuple is not None:
            slab_sendrecv_j_start[index]       = recvtuple[0]
            slab_sendrecv_k_start[index]       = recvtuple[1]
            slab_sendrecv_j_end[index]         = recvtuple[2]
            slab_sendrecv_k_end[index]         = recvtuple[3]
            domain2slabs_recvsend_ranks[index] = rank_recv
            index += 1
    # Cut off the tails
    slab_sendrecv_j_start = slab_sendrecv_j_start[:index]
    slab_sendrecv_k_start = slab_sendrecv_k_start[:index]
    slab_sendrecv_j_end = slab_sendrecv_j_end[:index]
    slab_sendrecv_k_end = slab_sendrecv_k_end[:index]
    domain2slabs_recvsend_ranks = domain2slabs_recvsend_ranks[:index]
    # The maximum number of communications it takes to communicate
    # the domain grid to the slabs (or vice versa).
    N_domain2slabs_communications = np.max([slabs2domain_sendrecv_ranks.shape[0],
                                            domain2slabs_recvsend_ranks.shape[0]])
    # Store and return all the resultant information,
    # needed for communicating between the domain and the slab.
    info = (N_domain2slabs_communications,
            domain2slabs_recvsend_ranks,
            slabs2domain_sendrecv_ranks,
            domain_sendrecv_i_start,
            domain_sendrecv_i_end,
            slab_sendrecv_j_start,
            slab_sendrecv_j_end,
            slab_sendrecv_k_start,
            slab_sendrecv_k_end,
            )
    decomposition_info[domain_grid_shape, slab_shape] = info
    return info
# Cache storing results of the prepare_decomposition function.
# The keys have the format (domain_grid_shape, slab_shape).
cython.declare(decomposition_info=dict)
decomposition_info = {}

# Function for transfering data from slabs to domain grids
@cython.pheader(
    # Arguments
    slab_or_slabs=object,  # double[:, :, ::1] or dict
    grid_or_grids_or_buffer_name=object,  # double[:, :, ::1], dict, int or str
    do_ghost_communication='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
    grid='double[:, :, ::1]',
    grid_noghosts='double[:, :, :]',
    grid_or_buffer_name=object,  # double[:, :, ::1], int or str
    grids=dict,
    gridsize='Py_ssize_t',
    key=str,
    request=object,  # mpi4py.MPI.Request
    shape=tuple,
    slab='double[:, :, ::1]',
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slabs=dict,
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns=object,  # double[:, :, ::1] or dict
)
def domain_decompose(slab_or_slabs, grid_or_grids_or_buffer_name=0, do_ghost_communication=True):
    """If the first argument is a dict (of slabs), it is assumed that
    the second argument is a dict of grids, in which case these grids
    will be mutated in-place (and the dict of grids is returned back).
    If the first argument is a slab, the second arument may be a grid
    or a buffer name.
    """
    if isinstance(slab_or_slabs, dict):
        slabs = slab_or_slabs
        grids = grid_or_grids_or_buffer_name
        if set(slabs.keys()) != set(grids.keys()):
            abort(
                f'The slabs and grids dicts passed to domain_decompose() do not '
                f'share the same keys: {tuple(slabs.keys())}, {tuple(grids.keys())}.'
            )
        for key, slab in slabs.items():
            grid = grids[key]
            domain_decompose(slab, grid, do_ghost_communication)
        return grids
    slab = slab_or_slabs
    if slab is None:
        return None
    grid_or_buffer_name = grid_or_grids_or_buffer_name
    if slab.shape[0] > slab.shape[1]:
        masterwarn(
            'domain_decompose() was called with a slab that appears to be transposed, '
            'i.e. in Fourier space.'
        )
    # Determine the correct shape of the domain grid corresponding to
    # the passed slab.
    gridsize = slab.shape[1]
    shape = tuple([gridsize//domain_subdivisions[dim] + 2*nghosts for dim in range(3)])
    # If no domain grid is passed, fetch a buffer of the right shape
    if isinstance(grid_or_buffer_name, (int, str)):
        buffer_name = grid_or_buffer_name
        grid = get_buffer(shape, buffer_name)
    else:
        grid = grid_or_buffer_name
        if asarray(grid).shape != shape:
            abort(
                f'domain_decompose(): The slab and domain grid have '
                f'incompatible shapes: {asarray(slab).shape}, {asarray(grid).shape}.'
            )
    grid_noghosts = grid[
        nghosts:(grid.shape[0] - nghosts),
        nghosts:(grid.shape[1] - nghosts),
        nghosts:(grid.shape[2] - nghosts),
    ]
    # Compute needed communication variables
    (
        N_domain2slabs_communications,
        domain2slabs_recvsend_ranks,
        slabs2domain_sendrecv_ranks,
        domain_sendrecv_i_start,
        domain_sendrecv_i_end,
        slab_sendrecv_j_start,
        slab_sendrecv_j_end,
        slab_sendrecv_k_start,
        slab_sendrecv_k_end,
    ) = prepare_decomposition(grid, slab)
    # Communicate the slabs to the domain grid
    for ℓ in range(N_domain2slabs_communications):
        # The lower ranks storing the slabs sends part of their slab
        if ℓ < domain2slabs_recvsend_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # In the x-dimension, the slabs are always thinner than (or
            # at least as thin as) the domain.
            request = smart_mpi(slab[:,
                                     slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                                     slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ],
                                     ],
                                dest=domain2slabs_recvsend_ranks[ℓ],
                                mpifun='Isend')
        # The corresponding process receives the message.
        # Since the slabs extend throughout the entire yz-plane,
        # we receive into the entire yz-part of the domain grid
        # (excluding ghost points).
        if ℓ < slabs2domain_sendrecv_ranks.shape[0]:
            smart_mpi(
                grid_noghosts[
                    domain_sendrecv_i_start[ℓ]:domain_sendrecv_i_end[ℓ],
                    :grid_noghosts.shape[1],
                    :grid_noghosts.shape[2],
                ],
                source=slabs2domain_sendrecv_ranks[ℓ],
                mpifun='Recv',
            )
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    # Populate ghost layers
    if do_ghost_communication:
        communicate_ghosts(grid, '=')
    return grid

# Function for transfering data from domain grids to slabs
@cython.pheader(
    # Arguments
    grid_or_grids=object,  # double[:, :, ::1] or dict
    slab_or_buffer_name=object,  # double[:, :, ::1], int or str
    prepare_fft='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
    grid='double[:, :, ::1]',
    grid_noghosts='double[:, :, :]',
    grids=dict,
    gridsize='Py_ssize_t',
    representation=str,
    request=object,  # mpi4py.MPI.Request object
    shape=tuple,
    slab='double[:, :, ::1]',
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slabs=dict,
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns=object,  # double[:, :, ::1] or dict
)
def slab_decompose(grid_or_grids, slab_or_buffer_name='slab_particles', prepare_fft=False):
    """This function communicates a global domain decomposed grid into
    a global slab decomposed grid. If an existing slab grid should be
    used it can be passed as the second argument.
    Alternatively, if a slab grid should be fetched from elsewhere,
    its name should be specified as the second argument.

    If the first argument is not a single grid but a dict of grids,
    a corresponding dict of slabs will be returned. Here the slab buffer
    names will be constructed from the keys of the passed dict,
    rendering the second argument superfluous.

    If FFT's are to be carried out on a slab fetched by name,
    you must specify prepare_fft=True, in which case the slab will be
    created via FFTW.

    Note that ghost points will not be read by this function,
    and so the passed domain grid need not have these set properly.

    By default, the slab called 'slab_particles' is used. Often we only
    hold a single slab in memory at a time, and so this one chunk of
    memory is reused for lots of different purposes. We need multiple
    simultaneous slabs for doing FFT's of pairs of grids containing
    interpolated particle and fluid data (gravity, power spectra).
    Here, the slabs 'slab_particles' and 'slab_fluid' are used,
    which is why one of these is used as the default.
    """
    if isinstance(grid_or_grids, dict):
        grids = grid_or_grids
        slabs = {
            representation: slab_decompose(grid, f'slab_{representation}', prepare_fft)
            for representation, grid in grids.items()
        }
        return slabs
    grid = grid_or_grids
    if grid is None:
        return None
    # Determine the correct shape of the slab grid corresponding to
    # the passed domain grid.
    grid_noghosts = grid[
        nghosts:(grid.shape[0] - nghosts),
        nghosts:(grid.shape[1] - nghosts),
        nghosts:(grid.shape[2] - nghosts),
    ]
    gridsize = grid_noghosts.shape[0]*domain_subdivisions[0]
    if gridsize%nprocs != 0:
        abort(
            f'A domain decomposed grid of size {gridsize} was passed to the slab_decompose() '
            f'function. This grid size is not evenly divisible by {nprocs} processes.'
        )
    shape = (
        gridsize//nprocs,  # Distributed dimension
        gridsize,
        2*(gridsize//2 + 1), # Padded dimension
    )
    # If no slab grid is passed, fetch a buffer of the right shape
    if isinstance(slab_or_buffer_name, (int, str)):
        buffer_name = slab_or_buffer_name
        if prepare_fft:
            slab = get_fftw_slab(gridsize, buffer_name)
        else:
            slab = get_buffer(shape, buffer_name)
    else:
        slab = slab_or_buffer_name
        if asarray(slab).shape != shape:
            abort(
                f'slab_decompose(): The slab and domain grid have '
                f'incompatible shapes: {asarray(slab).shape}, {asarray(grid).shape}.'
            )
    # Nullify the additional elements in the padded dimension, which may
    # contain junk. This is not needed if an FFT is to be done.
    # However, some places the code skips the FFT because it knows that
    # the slab consists purely of zeros. Any junk in the additional
    # elements ruin this.
    # Due to a bug in cython, we need the below operation to be done by
    # NumPy, not directly on the memory view.
    # See https://github.com/cython/cython/issues/2941.
    slab_arr = asarray(slab)
    slab_arr[:, :, gridsize:] = 0
    # Compute needed communication variables
    (
        N_domain2slabs_communications,
        domain2slabs_recvsend_ranks,
        slabs2domain_sendrecv_ranks,
        domain_sendrecv_i_start,
        domain_sendrecv_i_end,
        slab_sendrecv_j_start,
        slab_sendrecv_j_end,
        slab_sendrecv_k_start,
        slab_sendrecv_k_end,
    ) = prepare_decomposition(grid, slab)
    # Communicate the domain grid to the slabs
    for ℓ in range(N_domain2slabs_communications):
        # Send part of the local domain
        # grid to the corresponding process.
        if ℓ < slabs2domain_sendrecv_ranks.shape[0]:
            # A non-blocking send is used, because the communication
            # is not pairwise.
            # Since the slabs extend throughout the entire yz-plane,
            # we should send the entire yz-part of domain
            # (excluding ghost points).
            request = smart_mpi(
                grid_noghosts[
                    domain_sendrecv_i_start[ℓ]:domain_sendrecv_i_end[ℓ],
                    :grid_noghosts.shape[1],
                    :grid_noghosts.shape[2],
                ],
                dest=slabs2domain_sendrecv_ranks[ℓ],
                mpifun='Isend',
            )
        # The lower ranks storing the slabs receives the message.
        # In the x-dimension, the slabs are always thinner than (or at
        # least as thin as) the domain.
        if ℓ < domain2slabs_recvsend_ranks.shape[0]:
            smart_mpi(
                slab[
                    :,
                    slab_sendrecv_j_start[ℓ]:slab_sendrecv_j_end[ℓ],
                    slab_sendrecv_k_start[ℓ]:slab_sendrecv_k_end[ℓ],
                ],
                source=domain2slabs_recvsend_ranks[ℓ],
                mpifun='Recv',
            )
        # Wait for the non-blockind send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    return slab

# Iterator implementing looping over Fourier space slabs.
# The yielded values are the linear index into the slab, the physical
# ki, kj, kk (in grid units) and the deconvolution factor.
# Nyquist planes are excluded from the iteration. If sparse is True,
# only unique points in the z DC plane will be visited. If the
# deconvolution factor is to be used, specify the deconvolution
# (interpolation) order as deconv_order. If both a 'particles' and a
# 'particles_shifted' slab is present, these will be interlaced together
# (in-place into the 'particles' slab) if do_interlacing is True.
@cython.iterator(
    depends=[
        # Functions used by slab_fourier_loop()
        'get_deconvolution_factor',
    ]
)
def slab_fourier_loop(
    slabs, sparse=False, deconv_order=0, do_interlacing=False,
):
    # Cython declarations for variables used for the iteration,
    # not including positional arguments and variables to yield,
    # but including keyword arguments.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Keyword arguments
        sparse='bint',
        deconv_order='int',
        do_interlacing='bint',
        # Locals
        _gridsize='Py_ssize_t',
        _nyquist='Py_ssize_t',
        _slab='double[:, :, ::1]',
        _slab_particles='double[:, :, ::1]',
        _slab_particles_ptr='double*',
        _slab_particles_shifted='double[:, :, ::1]',
        _slab_particles_shifted_ptr='double*',
        deconv_ij='double',
        deconv_j='double',
        i='Py_ssize_t',
        im='double',
        j='Py_ssize_t',
        j_global='Py_ssize_t',
        ki_plus_kj='Py_ssize_t',
        re='double',
        slab_size_i='Py_ssize_t',
        slab_size_j='Py_ssize_t',
        slab_size_k='Py_ssize_t',
        θ='double',
    )
    # Default values, mostly to satisfy the compiler
    deconv = deconv_j = deconv_ij = 1
    ki_plus_kj = 0
    # The index into a slab
    index = 0
    # The DC element (origin) is always skipped in the below iteration.
    # This element is always located at the beginning of the slab
    # belonging to the master process.
    if master:
        index += 2
    # As we increment the index (by 2) prior to using it,
    # we decrement it now.
    index -= 2
    # Prepare for the interlacing
    if do_interlacing:
        _slab_particles         = slabs.get('particles')
        _slab_particles_shifted = slabs.get('particles_shifted')
        if _slab_particles is None or _slab_particles_shifted is None:
            do_interlacing = False
        else:
            _slab_particles_ptr         = cython.address(_slab_particles        [:, :, :])
            _slab_particles_shifted_ptr = cython.address(_slab_particles_shifted[:, :, :])
    # Get slab shape (must be equal between slabs of different keys).
    # Note that in Fourier space, the x (i) and y (j) dimensions
    # are transposed.
    for _slab in slabs.values():
        slab_size_j, slab_size_i, slab_size_k = asarray(_slab).shape
        break
    _gridsize = slab_size_i
    _nyquist = _gridsize//2
    # Begin iterating over slab. As the first and second dimensions
    # are transposed due to the FFT, the j-dimension is first.
    for j in range(slab_size_j):
        # The j-component of the wave vector (grid units).
        # Since the slabs are distributed along the j-dimension,
        # an offset must be used.
        j_global = ℤ[slab_size_j*rank] + j
        kj = j_global - _gridsize*(j_global >= _nyquist)
        # The j-component of the deconvolution
        with unswitch(1):
            if deconv_order:
                deconv_j = get_deconvolution_factor(kj*ℝ[π/_gridsize])
        # Loop through the complete i-dimension
        for i in range(_gridsize):
            # The i-component of the wave vector (grid units)
            ki = i - _gridsize*(i >= _nyquist)
            # The product of the i- and the j-component
            # of the deconvolution.
            with unswitch(2):
                if deconv_order:
                    deconv_ij = deconv_j*get_deconvolution_factor(ki*ℝ[π/_gridsize])
            # The sum of wave vector elements, used for interlacing
            with unswitch(2):
                if do_interlacing:
                    ki_plus_kj = ki + kj
            # Only the non-negative part of the k-dimension exists.
            # Loop through this half one complex number at a time,
            # looping directly over kk instead of k == 2*kk.
            # To avoid |k|² = 0 we start at kk = 1 if kj == 0 == ki.
            for kk in range(𝔹[kj == 0] and ki == 0, _nyquist + 1):
                # Jump to the next complex number
                index += 2
                # Skip Nyquist points
                if 𝔹[𝔹[ki == ℤ[-_nyquist]] or 𝔹[kj == ℤ[-_nyquist]]] or kk == _nyquist:
                    continue
                # The z DC plane consists of complex conjugate pairs of
                # points. When looping sparsely we only want to hit one
                # point from each pair. To do this, we choose to skip
                # points with positive ki and also points with positive
                # kj and ki == 0.
                with unswitch(3):
                    if sparse:
                        if 𝔹[ki > 0 or (𝔹[ki == 0] and 𝔹[kj > 0])] and kk == 0:
                            continue
                # Interlace the two relatively shifted particle slabs
                with unswitch(3):
                    if do_interlacing:
                        # Rotate the phase of the complex number of the
                        # shifted particles slab at this [j, i, k] by θ,
                        # which according to harmonic averaging is
                        #   θ = (kx + ky + kz)*(gridsize/boxsize)/2
                        #     = π/gridsize*(ki + kj + kk)
                        re = _slab_particles_shifted_ptr[index    ]
                        im = _slab_particles_shifted_ptr[index + 1]
                        θ = ℝ[π/_gridsize]*(ki_plus_kj + kk)
                        re, im = re*ℝ[cos(θ)] - im*ℝ[sin(θ)], re*ℝ[sin(θ)] + im*ℝ[cos(θ)]
                        # The interlaced result overwrites the current
                        # values in the particles slab.
                        _slab_particles_ptr[index    ] = 0.5*(_slab_particles_ptr[index    ] + re)
                        _slab_particles_ptr[index + 1] = 0.5*(_slab_particles_ptr[index + 1] + im)
                # The full deconvolution factor
                with unswitch(3):
                    if deconv_order:
                        # The total (NGP) deconvolution factor
                        deconv = deconv_ij*get_deconvolution_factor(kk*ℝ[π/_gridsize])
                        # The full deconvolution factor
                        deconv **= deconv_order
                # Yield the needed variables
                yield index, ki, kj, kk, deconv

# Function for deconvolving and/or interlacing particle grids
@cython.header(
    # Arguments
    grids=dict,
    order='int',
    interlacing='bint',
    do_ghost_communication='bint',
    # Locals
    any_particles='bint',
    any_particles_shifted='bint',
    deconv='double',
    fft_factor='double',
    grid='double[:, :, ::1]',
    grids_extra=dict,
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    representation=str,
    slab='double[:, :, ::1]',
    slab_particles='double[:, :, ::1]',
    slab_particles_ptr='double*',
    slabs=dict,
    returns=dict,
)
def deconvolve_interlace(grids, order=0, interlacing=True, do_ghost_communication=True):
    """The passed grids dict should be of the form returned by
    interpolate_components(). If a fluid grid is present, this will be
    ignored. The particle grid will be deconvolved if order > 0.
    Interlacing is performed if interlacing is True and both a standard
    and a shifted particle grid is present. Note that this updates the
    values of the standard particle grid in-place, and that the shifted
    grid will be popped from the grids dict. Also, if deconvolution is
    to be applied but interlacing is not and a shifted particle grid is
    present, this shifted grid will not be touched at all.
    Ghosts will be populated by setting do_ghost_communication to True.
    Note that this will not take place if neither deconvolution nor
    interlacing is to be carried out.
    """
    # Bail out if nothing to do
    any_particles         = ('particles'         in grids)
    any_particles_shifted = ('particles_shifted' in grids)
    if not any_particles:
        if any_particles_shifted:
            abort(
                'deconvolve_interlace() got a "particles_shifted" grid '
                'but no matching "particles" grid'
            )
        return grids
    if order == 0 and (not interlacing or not any_particles_shifted):
        return grids
    # Stow away unneeded grids
    grids_extra = {}
    if 'fluid' in grids:
        grids_extra['fluid'] = grids['fluid']
        grids['fluid'] = None
    if not interlacing and 'particles_shifted' in grids:
        grids_extra['particles_shifted'] = grids['particles_shifted']
        grids['particles_shifted'] = None
    # Slab decompose the particle grids
    slabs = slab_decompose(grids, prepare_fft=True)
    # Do a forward in-place Fourier transform of the slabs
    fft(slabs, 'forward')
    # Get the grid size from the slab
    for slab in slabs.values():
        slab_size_i = slab.shape[1]
        break
    gridsize = slab_size_i
    # Multiplicative factor needed after a
    # forward and a backward Fourier transformation.
    fft_factor = float(gridsize)**(-3)
    # Perform the deconvolution and interlacing
    slab_particles = slabs['particles']
    slab_particles_ptr = cython.address(slab_particles[:, :, :])
    for index, ki, kj, kk, deconv in slab_fourier_loop(
        slabs,
        deconv_order=order,
        do_interlacing=interlacing,
    ):
        # Compute the total power at this index resulting
        # from both particles and fluid components,
        # with the particles slab values deconvolved.
        slab_particles_ptr[index    ] *= ℝ[fft_factor*deconv]  # real part
        slab_particles_ptr[index + 1] *= ℝ[fft_factor*deconv]  # imag part
    # The fft_factor should be applied to the DC element as well
    if master:
        slab_particles_ptr[0] *= fft_factor  # real part
        slab_particles_ptr[1] *= fft_factor  # imag part
    # Fourier transform the slabs back to coordinate space
    fft(slabs, 'backward')
    # Domain-decompose the slabs
    domain_decompose(slabs, grids, do_ghost_communication)
    # Put the unneeded grids back in and remove the
    # shifted particle grid if interlacing has been performed.
    for representation, grid in grids_extra.items():
        grids[representation] = grid
    if interlacing:
        grids.pop('particles_shifted', None)
    return grids

# Function returning the Fourier-space deconvulution factor needed for
# NGP interpolation in one dimension. The full deconvulution factor is
# achieved through exponentiation (**2 -> CIC, **3 -> TSC, **4 -> PCS)
# and multiplication with one-dimensional factors for other dimensions.
# The value to pass should be kᵢ*π/gridsize, with kᵢ the i'th component
# of the wave vector in grid units.
@cython.header(value='double', returns='double')
def get_deconvolution_factor(value):
    # The NGP deconvolution factor is given by
    #   1/sinc(value)
    # with
    #   sinc(value) ≡ sin(value)/value
    # and
    #   sinc(0) = 1.
    # Exploit the short-circuit behavior of 'or' to avoid division by
    # zero. Cast to double to ensure that we never get a boolean back
    # in pure Python mode.
    return cast((value == 0) or value/sin(value), 'double')

# Function for nullifying sets of modes of Fourier space slabs
@cython.header(
    # Arguments
    slab_or_slabs=object,  # double[:, :, ::1] or dict
    nullifications=object,  # str or list of str's
    # Locals
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    nullification=str,
    nyquist='Py_ssize_t',
    slab='double[:, :, ::1]',
    slab_jik='double*',
    slab_size_j='Py_ssize_t',
    slab_size_i='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    slabs=dict,
    returns='void',
)
def nullify_modes(slab_or_slabs, nullifications):
    """The nullifications argument can be a str of comma-separated
    words, or alternatively a list of str's, each being a single word.
    The words specify which types of modes to nullify:
    - "DC": Nullify the origin ki = kj = kk = 0. Other points which may
      be considered as "DC" along only one or two dimensions
      (e.g. ki = kj = 0, kk = 7) will not be nullified.
    - "Nyquist": Nullify the three Nyquist planes:
        ki = -nyquist, -nyquist ≤ kj < nyquist, 0 ≤ kk ≤ nyquist.
        kj = -nyquist, -nyquist ≤ ki < nyquist, 0 ≤ kk ≤ nyquist.
        kk = +nyquist, -nyquist ≤ ki < nyquist, -nyquist ≤ kj < nyquist.
    """
    if isinstance(slab_or_slabs, dict):
        slabs = slab_or_slabs
        for slab in slabs.values():
            nullify_modes(slab, nullifications)
        return
    else:
        slab = slab_or_slabs
    # Parse nullifications
    if isinstance(nullifications, str):
        nullifications = nullifications.split(',')
    nullifications = [nullification.strip().lower() for nullification in nullifications]
    # Get slab dimensions
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    nyquist = gridsize//2
    # Perform nullifications
    for nullification in nullifications:
        if nullification == 'dc':
            # Nullify the DC point ki == kj == kk == 0. This is always
            # located as the first element on the master process.
            if master:
                slab_jik = cython.address(slab[0, 0, 0:])
                slab_jik[0] = 0  # real part
                slab_jik[1] = 0  # imag part
        elif nullification == 'nyquist':
            # Nullify the three Nyquist planes ki == -nyquist,
            # kj = -nyquist and kk = +nyquist. These planes overlap
            # pairwise at the edges and so a little effort can be spared
            # by not nullifying these edges twice. We take this into
            # account for the two edges in the kk = +nyquist plane but
            # not for the remaining ki = kj = -nyquist edge, as skipping
            # the Nyquist point along the i or j direction (unlike along
            # the k direction) requires logic.
            ki = -nyquist
            i = ki + gridsize*(ki < 0)
            for j in range(slab_size_j):
                for k in range(0, gridsize, 2):  # exclude k = gridsize (kk = nyquist)
                    slab_jik = cython.address(slab[j, i, k:])
                    slab_jik[0] = 0  # real part
                    slab_jik[1] = 0  # imag part
            kj = -nyquist
            j_global = kj + gridsize*(kj < 0)
            j = j_global - ℤ[slab_size_j*rank]
            if 0 <= j < slab_size_j:
                for i in range(gridsize):
                    for k in range(0, gridsize, 2):  # exclude k = gridsize (kk = nyquist)
                        slab_jik = cython.address(slab[j, i, k:])
                        slab_jik[0] = 0  # real part
                        slab_jik[1] = 0  # imag part
            kk = +nyquist
            k = 2*kk
            for j in range(slab_size_j):
                for i in range(gridsize):
                    slab_jik = cython.address(slab[j, i, k:])
                    slab_jik[0] = 0  # real part
                    slab_jik[1] = 0  # imag part
        else:
            abort(f'nullify_modes(): nullification "{nullification}" not understood')

# Function that returns a slab decomposed grid,
# allocated by FFTW.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    buffer_name=object,  # int or str
    nullify='bint',
    # Locals
    acquire='bint',
    as_expected='bint',
    fftw_plans_index='Py_ssize_t',
    fftw_struct=fftw_return_struct,
    plan_backward=fftw_plan,
    plan_forward=fftw_plan,
    shape=tuple,
    slab='double[:, :, ::1]',
    slab_address='Py_ssize_t',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_start_i='Py_ssize_t',
    slab_start_j='Py_ssize_t',
    wisdom_filename=str,
    returns='double[:, :, ::1]',
)
def get_fftw_slab(gridsize, buffer_name='slab_particles', nullify=False):
    """By default, the slab called 'slab_particles' is used. Often we
    only hold a single slab in memory at a time, and so this one chunk
    of memory is reused for lots of different purposes. The one thing
    that we need multiple simultaneous slabs for is for doing FFT's of
    pairs of grids containing interpolated particle and fluid data
    (gravity, power spectra). Here, the slabs 'slab_particles' and
    'slab_fluid' are used, which is why one of these is used
    as the default.
    """
    global fftw_plans_size, fftw_plans_forward, fftw_plans_backward
    # If this slab has already been constructed, fetch it
    slab = slabs.get((gridsize, buffer_name))
    if slab is not None:
        if nullify:
            slab[...] = 0
        return slab
    # Checks on the passed gridsize
    if gridsize%nprocs:
        abort(
            f'A grid size of {gridsize} was passed to the get_fftw_slab() function. '
            f'This grid size is not evenly divisible by {nprocs} processes.'
        )
    if gridsize%2:
        masterwarn(
            f'An odd grid size ({gridsize}) was passed to the get_fftw_slab() function. '
            f'Some operations may not function correctly.'
    )
    shape = (
        (gridsize//nprocs), # Distributed dimension
        (gridsize),
        # Explicit int cast necessary for some reason
        int(2*(gridsize//2 + 1)), # Padded dimension
    )
    # In pure Python mode we use NumPy, which really means that there
    # is no needed preparations. In compiled mode we use FFTW,
    # which means that the grid and its plans must be prepared.
    if not cython.compiled:
        slab = empty(shape, dtype=C2np['double'])
    else:
        # Get path to FFTW wisdom file
        wisdom_filename = get_wisdom_filename(gridsize)
        # Initialize fftw_mpi, allocate the grid, initialize the
        # local grid sizes and start indices and do FFTW planning.
        acquire = False
        if master:
            os.makedirs(os.path.dirname(wisdom_filename), exist_ok=True)
            if gridsize not in wisdom_acquired and not os.path.isfile(wisdom_filename):
                acquire = True
                masterprint(
                    f'Acquiring FFTW wisdom ({fftw_wisdom_rigor}) for grid size {gridsize} ...'
                )
        fftw_struct = fftw_setup(
            gridsize, gridsize, gridsize,
            bytes(fftw_wisdom_rigor, encoding='ascii'),
            fftw_wisdom_reuse,
            bytes(wisdom_filename, encoding='ascii'),
        )
        if acquire:
            masterprint('done')
        wisdom_acquired[gridsize] = True
        # Unpack every variable from fftw_struct
        # and compare to expected values.
        slab_size_i   = int(fftw_struct.gridsize_local_i)
        slab_size_j   = int(fftw_struct.gridsize_local_j)
        slab_start_i  = int(fftw_struct.gridstart_local_i)
        slab_start_j  = int(fftw_struct.gridstart_local_j)
        plan_forward  = fftw_struct.plan_forward
        plan_backward = fftw_struct.plan_backward
        slab_ptr      = fftw_struct.grid
        as_expected = True
        if (
               slab_size_i  != ℤ[shape[0]]
            or slab_size_j  != ℤ[shape[0]]
            or slab_start_i != ℤ[shape[0]*rank]
            or slab_start_j != ℤ[shape[0]*rank]
        ):
            as_expected = False
            warn(
                f'FFTW has distributed a slab of grid size {gridsize} differently '
                f'from what was expected on rank {rank}:\n'
                f'    slab_size_i  = {slab_size_i}, expected {shape[0]},\n'
                f'    slab_size_j  = {slab_size_j}, expected {shape[0]},\n'
                f'    slab_start_i = {slab_start_i}, expected {shape[0]*rank},\n'
                f'    slab_start_j = {slab_start_j}, expected {shape[0]*rank},\n'
            )
        as_expected = allreduce(as_expected, op=MPI.LOR)
        if not as_expected:
            abort('Refusing to carry on with this non-expected decomposition')
        # Wrap the slab pointer in a memory view. Looping over this
        # memory view should be done as noted in fft.c, but use
        # slab[i, j, k] when in real space and slab[j, i, k]
        # when in Fourier space.
        slab = cast(slab_ptr, 'double[:shape[0], :shape[1], :shape[2]]')
        # Store the plans for this slab in the global
        # fftw_plans_forward and fftw_plans_backward arrays.
        fftw_plans_index = fftw_plans_size
        fftw_plans_size += 1
        fftw_plans_forward  = realloc(fftw_plans_forward , fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_backward = realloc(fftw_plans_backward, fftw_plans_size*sizeof('fftw_plan'))
        fftw_plans_forward [fftw_plans_index] = plan_forward
        fftw_plans_backward[fftw_plans_index] = plan_backward
        # Insert mapping from the slab to the index of its plans
        # in the global fftw_plans_forward and fftw_plans_backward
        # arrays, into the global fftw_plans_mapping dict.
        slab_address = cast(cython.address(slab[:, :, :]), 'Py_ssize_t')
        fftw_plans_mapping[slab_address] = fftw_plans_index
    # Store and return this slab
    slabs[gridsize, buffer_name] = slab
    if nullify:
        slab[...] = 0
    return slab
# Cache storing slabs. The keys have the format (gridsize, buffer_name).
cython.declare(slabs=dict)
slabs = {}
# Arrays of FFTW plans
cython.declare(
    fftw_plans_size='Py_ssize_t',
    fftw_plans_forward ='fftw_plan*',
    fftw_plans_backward='fftw_plan*',
)
fftw_plans_size = 0
fftw_plans_forward  = malloc(fftw_plans_size*sizeof('fftw_plan'))
fftw_plans_backward = malloc(fftw_plans_size*sizeof('fftw_plan'))
# Mapping from memory addreses of slabs to indices in
# fftw_plans_forward and fftw_plans_backward.
cython.declare(fftw_plans_mapping=dict)
fftw_plans_mapping = {}
# Dict keeping track of what FFTW wisdom has already been acquired
cython.declare(wisdom_acquired=dict)
wisdom_acquired = {}

# Helper function for the get_fftw_slab() function,
# which construct the absolute path to the wisdome file to use.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    content=str,
    fftw_pkgconfig_filename=str,
    index='Py_ssize_t',
    match=object,  # re.Match
    node_process_count=object,  # collections.Counter
    other_node='int',
    other_node_name=str,
    primary_nodes=list,
    process_count='Py_ssize_t',
    process_count_max='Py_ssize_t',
    sha_length='int',
    wisdom_filename=str,
    wisdom_hash=str,
    returns=str,
)
def get_wisdom_filename(gridsize):
    """The FFTW wisdom file name is built as a hash of several things:
    - The passed grid size.
    - The total number of processes.
    - The global FFTW wisdom rigor.
    - The FFTW version.
    - The name of the node "owning" the wisdom in the case of
      fftw_wisdom_share being True. Here a node is said to own the
      wisdom if i hosts the majority of the processes. A more elaborate
      key like the complete MPI layout is of no use, as FFTW wisdom is
      really generated on each process, after which the wisdom of one is
      chosen arbitrarily as the wisdom to stick with.
      When fftw_wisdom_share is False, this part of the key is constant.
    """
    global fftw_version, wisdom_owner
    # The master process constructs the file name
    # and then broadcasts it.
    if not master:
        return bcast()
    # Get the version of FFTW in use
    if not fftw_version:
        fftw_version = '<unknown>'
        fftw_pkgconfig_filename = paths['fftw_dir'] + f'/lib/pkgconfig/fftw3.pc'
        if os.path.exists(fftw_pkgconfig_filename):
            with open(fftw_pkgconfig_filename, 'r') as fftw_pkgconfig_file:
                content = fftw_pkgconfig_file.read()
            match = re.search('Version.*?([0-9].*)', content)
            if match:
                fftw_version = match.group(1)
            else:
                masterwarn('Failed to determine FFTW version from fftw3.pc')
        else:
            masterwarn('Could not find the fftw3.pc file needed to determine the FFTW version')
    # Get the name of the node owning the wisdom
    if not wisdom_owner:
        if fftw_wisdom_share:
            wisdom_owner = '<shared>'
        else:
            node_process_count = collections.Counter()
            for other_node in range(nnodes):
                other_node_name = node_numbers2names[other_node]
                node_process_count[other_node_name] += np.sum(asarray(nodes) == other_node)
            primary_nodes = node_process_count.most_common(len(node_process_count))
            process_count_max = primary_nodes[0][1]
            for index, (other_node_name, process_count) in enumerate(primary_nodes):
                if process_count < process_count_max:
                    primary_nodes = primary_nodes[:index]
                    break
            wisdom_owner = sorted([  # guarantees deterministic outcome in case of ties
                other_node_name
                for other_node_name, process_count in primary_nodes
            ])[0]
    # Construct hash
    sha_length = 10  # 10 -> 50% chance of 1 hash collision after ~10⁶ hashes
    wisdom_hash = hashlib.sha1(str((
        gridsize,
        nprocs,
        fftw_wisdom_rigor,
        fftw_version,
        wisdom_owner,
    )).encode()).hexdigest()[:sha_length]
    # The full path to the wisdom file
    wisdom_filename = paths['reusables_dir'] + f'/fftw/{wisdom_hash}.wisdom'
    # Broadcast and return result
    return bcast(wisdom_filename)
# Constant strings set and used by the get_wisdom_filename function
cython.declare(fftw_version=str, wisdom_owner=str)
fftw_version = ''
wisdom_owner = ''

# Function performing Fourier transformations of slab decomposed grids
@cython.header(
    # Arguments
    slab_or_slabs=object,  # double[:, :, ::1] or dict
    direction=str,
    # Locals
    slab='double[:, :, ::1]',
    slabs=dict,
    fftw_plans_index='Py_ssize_t',
    slab_address='Py_ssize_t',
    returns='void',
)
def fft(slab_or_slabs, direction):
    """Fourier transform the given slab decomposed grid.
    For a forwards transformation from real to Fourier space, supply
    direction='forward'. Note that this is an unnormalized transform,
    as defined by FFTW. To do the normalization, divide all elements of
    the slab by gridsize**3, where gridsize is the linear grid size
    of the cubic grid.
    For a backwards transformation from Fourier to real space, supply
    direction='backward'. Here, no further normalization is needed,
    as defined by FFTW.

    In pure Python, NumPy is used to carry out the Fourier transforms.
    To emulate the effects of FFTW perfectly, a lot of extra steps
    are needed.
    """
    if isinstance(slab_or_slabs, dict):
        slabs = slab_or_slabs
        for slab in slabs.values():
            fft(slab, direction)
        return
    slab = slab_or_slabs
    if slab is None:
        return
    if not direction in ('forward', 'backward'):
        abort(
            f'fft() was called with the direction "{direction}", '
            f'which is neither "forward" nor "backward".'
        )
    if not cython.compiled:
        # The pure Python FFT implementation is serial.
        # Every process computes the entire FFT of the temporary
        # varaible grid_global_pure_python.
        slab_size_i = slab_size_j = slab.shape[0]
        slab_start_i = slab_size_i*rank
        slab_start_j = slab_size_j*rank
        gridsize = slab.shape[1]
        gridsize_padding = slab.shape[2]
        grid_global_pure_python = empty((gridsize, gridsize, gridsize_padding))
        Allgatherv(slab, grid_global_pure_python)
        if direction == 'forward':
            # Delete the padding on last dimension
            for i in range(gridsize_padding - gridsize):
                grid_global_pure_python = np.delete(grid_global_pure_python, -1, axis=2)
            # Do real transform via NumPy
            grid_global_pure_python = np.fft.rfftn(grid_global_pure_python)
            # FFTW transposes the first two dimensions
            grid_global_pure_python = grid_global_pure_python.transpose([1, 0, 2])
            # FFTW represents the complex array by doubles only
            tmp = empty((gridsize, gridsize, gridsize_padding))
            for i in range(gridsize_padding):
                if i % 2:
                    tmp[:, :, i] = grid_global_pure_python.imag[:, :, i//2]
                else:
                    tmp[:, :, i] = grid_global_pure_python.real[:, :, i//2]
            grid_global_pure_python = tmp
            # As in FFTW, distribute the slabs along the y-dimension
            # (which is the first dimension now, due to transposing).
            slab[...] = grid_global_pure_python[slab_start_j:(slab_start_j + slab_size_j), :, :]
        elif direction == 'backward':
            # FFTW represents the complex array by doubles only.
            # Go back to using complex entries.
            tmp = zeros((gridsize, gridsize, gridsize_padding//2), dtype='complex128')
            for i in range(gridsize_padding):
                if i % 2:
                    tmp[:, :, i//2] += 1j*grid_global_pure_python[:, :, i]
                else:
                    tmp[:, :, i//2] += grid_global_pure_python[:, :, i]
            grid_global_pure_python = tmp
            # FFTW transposes the first
            # two dimensions back to normal.
            grid_global_pure_python = grid_global_pure_python.transpose([1, 0, 2])
            # Do real inverse transform via NumPy
            grid_global_pure_python = np.fft.irfftn(grid_global_pure_python, s=[gridsize]*3)
            # Remove the autoscaling provided by NumPy
            grid_global_pure_python *= gridsize**3
            # Add padding on last dimension, as in FFTW
            padding = empty((gridsize,
                             gridsize,
                             gridsize_padding - gridsize,
                             ))
            grid_global_pure_python = np.concatenate((grid_global_pure_python, padding), axis=2)
            # As in FFTW, distribute the slabs along the x-dimension
            slab[...] = grid_global_pure_python[slab_start_i:(slab_start_i + slab_size_i), :, :]
    else:  # Compiled mode
        # Look up the index of the FFTW plans for the passed slab.
        slab_address = cast(cython.address(slab[:, :, :]), 'Py_ssize_t')
        fftw_plans_index = fftw_plans_mapping[slab_address]
        # Look up the plan and let FFTW do the Fourier transformation
        if direction == 'forward':
            fftw_execute(fftw_plans_forward[fftw_plans_index])
        elif direction == 'backward':
            fftw_execute(fftw_plans_backward[fftw_plans_index])

# Function for deallocating a slab and its plans, allocated by FFTW
@cython.header(# Arguments
               gridsize='Py_ssize_t',
               buffer_name=object,  # int or str
               # Locals
               fftw_plans_index='Py_ssize_t',
               plan_forward=fftw_plan,
               plan_backward=fftw_plan,
               slab='double[:, :, ::1]',
               slab_ptr='double*',
               )
def free_fftw_slab(gridsize, buffer_name):
    # Fetch the slab from the slab cache and remove it
    slab = slabs.pop((gridsize, buffer_name))
    # Grab pointer to the slab
    slab_ptr = cython.address(slab[:, :, :])
    # Look up the index of the FFTW plans for the passed slab
    # and use this to look up the plans.
    slab_address = cast(slab_ptr, 'Py_ssize_t')
    fftw_plans_index = fftw_plans_mapping[slab_address]
    plan_forward  = fftw_plans_forward[fftw_plans_index]
    plan_backward = fftw_plans_backward[fftw_plans_index]
    # Let FFTW do the cleanup
    fftw_clean(slab_ptr, plan_forward, plan_backward)
    # Note that the arrays fftw_plans_forward and fftw_plans_backward
    # as well as the dict fftw_plans_mapping have not been altered.
    # Thus, accessing the pointers in fftw_plans_forward or
    # fftw_plans_backward for the newly freed plans will cause a
    # segmentation fault. As this should not ever happen, we leave
    # these as is.

# Function for checking that the slabs satisfy the required symmetry
# of a Fourier transformed real field.
@cython.remove
def slabs_check_symmetry(
    slab, nullified_nyquist=False,
    gridsize=-1, allow_zeros=False, pure_embedding=True, count_information=True,
    rel_tol=1e-12, abs_tol=machine_ϵ,
):
    """This function checks and reports on the symmetries that a Fourier
    transformed 3D grid (passed as FFTW slabs) of real data should obey.
    Two distinct symmetries exist:
    - Bulk inversion through the origin, i.e. grid point
      [ki, kj, kk] should be the complex conjugate of [-ki, -kj, -kk].
    - Plane inversion through the center of each Nyquist plane,
      e.g. the grid point [-nyquist, kj, kk] should be the complex
      conjugate of [-nyquist, -kj, -kk].
    In the above, {ki, kj, kk} are the components of the physical k
    vector (in grid units in the code below), not indices i, j, k.
    Also, nyquist = gridsize//2.
    A standard 3D grid is tabulated at
      ki ∈ [-nyquist, -nyquist + 1, ..., nyquist - 1],
      kj ∈ [-nyquist, -nyquist + 1, ..., nyquist - 1],
      kk ∈ [0, 1, ..., nyquist],
    i.e. the z direction is cut roughly in half and the positive x and
    y Nyquist planes are not part of the tabulation. The surviving
    Nyquist planes are then the negative x and y and the positive z.

    From the bulk and Nyqist plane inversion symmetries it follows that
    (see the docstring of get_purely_reals() below) the grid point at
    the origin, grid points at centers of Nyquist planes, at centers of
    Nyquist edges and at corners must all be real. This is also checked.

    An example of a Nyquist edge could be [-nyquist, kj, nyquist]. Such
    edges are subject to three of the four symmetries. Applying all
    three, we have
        grid[-nyquist, +kj, +nyquist]
      → grid[+nyquist, -kj, -nyquist]*  (bulk inversion)
      → grid[+nyquist, +kj, +nyquist]   (x Nyquist plane inversion)
      → grid[-nyquist, -kj, +nyquist]*  (z Nyquist plane inversion)
    i.e. we effectively have a new symmetry; edge inversion, again under
    complex conjugation. Though this follows logically from the basic
    bulk and plane inversion symmetries, it is possible to satisfy these
    individually but not combined. Thus, a separate check for this edge
    symmetry is carried out. Note that all relevant edges lie in the
    positive z Nyquist plane, and that only two (one at ki = -nyquist
    and one at kj = -nyquist) are within the tabulated region.

    If a gridsize is further supplied to this function, it signals that
    the passed grid contains within it a smaller, embedded grid of the
    given size. This brings the positive x and y Nyquist planes within
    the tabulated region as well, and so the symmetry conditions will
    also be checked here. Note that the symmetry of plane inversion for
    the positive x and y Nyquist planes are not distinct symmetries, but
    follows from bulk inversion and the negative Nyquist plane
    symmetries. Thus, even in the case of an embedded grid, checking the
    bulk symmetry (now including ki, kj = +nyquist) and the three
    original Nyquist plane symmetries suffice. Similarly, combining the
    bulk symmetry with one of the Nyquist plane symmetries, a "new"
    symmetry is produced, akin to the aforementioned edge symmetries:
        grid[-nyquist, +kj, +kk]
      → grid[+nyquist, -kj, -kk]*  (bulk inversion)
      → grid[+nyquist, +kj, +kk]   (x Nyquist plane inversion)
    i.e. pairs of parallel Nyquist planes are identical copies of one
    another, without doing plane inversion or conjugation. Again, since
    this symmetry is implied by the basic bulk and three Nyquist plane
    symmetries (and automatically enforced once each of these are
    satisfied — unlike the edge symmetry), this will not be checked
    explicitly. Finally, note that the edge symmetries at kk = +nyquist
    and ki = +nyquist or kj = +nyquist are also guaranteed to be
    satistied if the bulk symmetry and the corresponding ki = -nyquist
    and kj = -nyquist symmetries are satisfied, and so these two extra
    edge symmetries will also not be explicitly checked.

    For further explanation of symmetries and data layout,
    see the comments and especially docstrings below.

    An easy way to falsely satisfy the symmetry conditions is by having
    grid cells being equal to zero. When allow_zeros is False, any zeros
    found within the embedded grid will be interpreted as failures.

    If an embedded grid is passed and pure_embedding is True,
    it is further checked that all modes outside of the small grid
    are exactly zero.

    Too much symmetry — as in wrongly copied data points — is also bad.
    When count_information is True, all (real and imaginary) numbers in
    the grid are compared against one another in order to find the total
    information content of the grid. This is then compared to what it
    should be for a Fourier transformed grid which had no symmetries at
    all in real space, i.e. each Fourier mode is independent. If the
    input grid is the Fourier transform of some tabulated analytical
    function, you should probably set count_information to zero.

    If nullified_nyquist is True, it signals that the grid is supposed
    to have nullified Nyquist planes. This will be taken into account
    when searching for (non-)zeros and when counting the information
    content. Also, a check that the Nyquist planes really are nullified
    will be added.

    Note that this function is not written with performance in mind
    and should not be called during actual simualtion, and never with
    large grids.
    """
    # Get grid size
    gridsize_large = slab.shape[1]
    if gridsize == -1:
        gridsize = gridsize_large
        pure_embedding = False
    if gridsize%2:
        abort(f'Cannot check symmetry of grid of odd grid size {gridsize}')
    if gridsize_large%2:
        abort(
            f'Cannot check symmetry of grid embedded within a larger grid '
            f'of odd grid size {gridsize_large}'
        )
    if gridsize_large < gridsize:
        abort(
            f'The passed gridsize ({gridsize}) should be less than the '
            f'grid size of the grid ({gridsize_large})'
        )
    nyquist = gridsize//2
    nyquist_large = gridsize_large//2
    masterprint('Checking slab symmetries ...')
    # Gather all slabs into global grid on the master process
    if master:
        grid = empty((gridsize, gridsize, slab.shape[2]), dtype=C2np['double'])
        grid[:slab.shape[0], :, :] = slab[...]
        for slave in range(1, nprocs):
            j_bgn = slab.shape[0]*slave
            j_end = j_bgn + slab.shape[0]
            smart_mpi(grid[j_bgn:j_end, :, :], source=slave, mpifun='recv')
    else:
        smart_mpi(slab, dest=master_rank, mpifun='send')
        return
    # Create set of [ki, kj, kk] where grid points
    # ought to be purely real.
    def get_purely_reals(nyquist):
        """Purely real grid points occur when an odd number of
        applications of some of the basic symmetry inversions results in
        the identity transformation (as each inversion comes with a
        complex conjugation, having an odd number of these bringing a
        grid point back to itself implies that it should equal its own
        complex conjugate and hence be real).
        As an example, consider bulk inversion followed by x Nyquist
        plane inversion followed by y Nyquist plane inversion,
        on the grid point [ki = ±nyquist, kj = ±nyquist, kk = m]:
            grid[±nyquist, ±nyquist, +m]
          → grid[∓nyquist, ∓nyquist, -m]*  (bulk inversion)
          → grid[∓nyquist, ±nyquist, +m]   (x Nyquist plane inversion)
          → grid[±nyquist, ±nyquist, -m]*  (y Nyquist plane inversion)
        from which we see that [ki = ±nyquist, kj = ±nyquist, kk = m]
        should be real for m = 0.
        Similarly, [ki = ±nyquist, kj = ±nyquist, kk = ±nyquist] can be
        found to be real by applying all three Nyquist plane inversions.
        More trivially, applying just bulk inversions leaves [0, 0, 0]
        invariant, and applying just e.g. the x Nyquist plane inversions
        leaves [±nyquist, 0, 0] invariant. In total, all possible
        triplets [ki, kj, kk] with ki, kj, kk ∈  {0, ±nyquist} should
        be purely real. This corresponds to the center or the bulk
        (the origin), the center of each of the 6 Nyquist faces,
        the center of each of the 12 Nyquist edges
        and the 8 Nyquist corners.
        """
        return set(itertools.product(*[(0, +nyquist, -nyquist)]*3))
    purely_reals = get_purely_reals(nyquist)
    # Generator for looping over the grid
    def visit(nyquist=nyquist, sparse=False):
        """When sparse is False, all points within |nyquist| will be
        visited except kk < 0. This includes points with ki or kj equal
        to +nyquist. When sparse is True, we only visit points that are
        tabulated (i.e. we exclude ki and kj equal to +nyquist), and we
        also only visits one grid point from each conjugate grid point
        pair. To perform this sparse iteration, note that all conjugate
        pairs with both members in the tabulated region are situated
        either in the positive z Nyquist plane (kk == nyquist) or on the
        z DC plane (kk == 0). Both of these planes are symmetric with
        respect to inversion through their center (and complex
        conjugation), with the DC plane inheriting this symmetry from
        the bulk inversion symmetry. To only hit half of the points in
        these planes, we skip points with ki > 0. Furthermore, when
        ki == 0, we skip points with kj > 0, which reduces the remaining
        four symmetric lines to half lines. To better understand these
        four lines, consider:
          z DC plane:
            - grid[       0, kj, 0] == grid[       0, -kj, 0]* (bulk inversion)
            - grid[-nyquist, kj, 0] == grid[-nyquist, -kj, 0]* (x Nyquist plane inversion)
          z Nyquist plane:
            - grid[       0, kj, nyquist] == grid[       0, -kj, nyquist]* (z Nyquist plane inversion)
            - grid[-nyquist, kj, nyquist] == grid[-nyquist, -kj, nyquist]* (all three above inversions together)
        As a bonus, this sparse sampling allows us to calculate the
        number of independent grid points. Defining
          n_bulk      = (gridsize//2 + 1)*gridsize*gridsize
          n_halfplane = (gridsize//2 - 1)*gridsize
          n_halfline   = (gridsize//2 - 1)
        as the number of grid points in the bulk, in one of the half
        planes that we skip, in one of the half lines that we skip,
        respectively, the total number comes out to be
          n_independent = n_bulk - 2*(n_halfplane + 2*n_halfline)
                        = gridsize**3//2 + 4
        """
        if not sparse:
            for         ki in range(-nyquist, nyquist + 1):
                for     kj in range(-nyquist, nyquist + 1):
                    for kk in range(       0, nyquist + 1):
                        yield ki, kj, kk
        else:
            for         ki in range(-nyquist, nyquist    ):
                for     kj in range(-nyquist, nyquist    ):
                    for kk in range(       0, nyquist + 1):
                        if kk == 0 or kk == nyquist:
                            if ki > 0:
                                continue
                            if (ki == 0 or ki == -nyquist) and kj > 0:
                                continue
                        yield ki, kj, kk
    # Function for checking that [ki, kj, kk] is within
    # the tabulated region.
    def is_within_tabulation(ki, kj, kk):
        if not (-nyquist_large <= ki < nyquist_large):
            return False
        if not (-nyquist_large <= kj < nyquist_large):
            return False
        if not (0 <= kk <= nyquist_large):
            return False
        return True
    # Function for looking up a complex grid point
    def lookup(ki, kj, kk):
        i = ki + gridsize_large*(ki < 0)
        j = kj + gridsize_large*(kj < 0)
        k = 2*kk
        return complex(*grid[i, j, k:k+2])
    # Function checking the reality condition
    # which applies for certain grid points.
    def check_reality(ki, kj, kk):
        """A return value of True signals that
        a non-zero imaginary part was found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip if this grid point is not one of those
        # that should be purely real.
        if (ki, kj, kk) not in purely_reals:
            return False
        # Check for non-zero imaginary part
        c = lookup(ki, kj, kk)
        if np.abs(c.imag) > abs_tol:
            name = {
                0: 'center of bulk (origin)',
                1: 'center of face',
                2: 'center of edge',
                3: 'corner',
            }[(np.abs(ki) == nyquist) + (np.abs(kj) == nyquist) + (np.abs(kk) == nyquist)]
            fancyprint(
                f'Should be real ({name}): [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function checking for existence of zeros inside embedded grid
    def check_zero_within_embedding(ki, kj, kk):
        """A return value of True signals that
        a zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        if nullified_nyquist and (nyquist in np.abs((ki, kj, kk))):
            return False
        # Check for zero
        c = lookup(ki, kj, kk)
        found_zero = False
        should_be_real = ((ki, kj, kk) in purely_reals)
        if should_be_real:
            if c.real == 0:
                found_zero = True
                fancyprint(
                    f'Found zero (should be real): [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                    wrap=False,
                    file=sys.stderr,
                )
        else:
            if c.real == 0 or c.imag == 0:
                found_zero = True
                fancyprint(
                    f'Found zero: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                    wrap=False,
                    file=sys.stderr,
                )
        return found_zero
    # Function checking for existence of non-zeros outside embedded grid
    def check_nonzero_beyond_embedding(ki, kj, kk):
        """A return value of True signals that
        a non-zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip if [ki, kj, kk] is within the embedding
        if np.abs(ki) <= nyquist or np.abs(kj) <= nyquist or np.abs(kk) <= nyquist:
            return False
        # Check for non-zero (exact)
        c = lookup(ki, kj, kk)
        if c != 0:
            fancyprint(
                f'Found non-zero outside embedding: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function checking for existence of non-zeros at Nyquist planes
    def check_nonzero_at_nyquist(ki, kj, kk):
        """A return value of True signals that
        a non-zero has been found.
        """
        if not is_within_tabulation(ki, kj, kk):
            return False
        # Skip non-Nyquist points
        if nyquist not in np.abs((ki, kj, kk)):
            return False
        # Check for non-zero (exact)
        c = lookup(ki, kj, kk)
        if c != 0:
            fancyprint(
                f'Found non-zero at Nyquist plane: [ki = {ki}, kj = {kj}, kk = {kk}] → {c}',
                wrap=False,
                file=sys.stderr,
            )
            return True
        return False
    # Function factory for generating inversion functions
    # capable of checking conjugate symmetries.
    def generate_check(accept, invert, name):
        def check(ki, kj, kk, *, visited=set()):
            """A return value of True signals broken symmetry"""
            # Skip if [ki, kj, kk] not within region of interest
            if not accept(ki, kj, kk):
                return False
            # Skip if [ki, kj, kk] not within tabulated region
            if not is_within_tabulation(ki, kj, kk):
                return False
            # Skip if inverted [ki, kj, kk] not within tabulated region
            ki_inv, kj_inv, kk_inv = invert(ki, kj, kk)
            if not is_within_tabulation(ki_inv, kj_inv, kk_inv):
                return False
            # Skip if [ki, kj, kk] has already been visited.
            # If not, add [ki, kj, kk] and its inversion
            # as visited sites.
            if (ki, kj, kk) in visited:
                return False
            visited.add((ki,     kj,     kk     ))
            visited.add((ki_inv, kj_inv, kk_inv))
            # If [ki, kj, kk] and its inversion are one and the same
            # point, the conjugate symmetry implies that this point
            # should be real. We could also just look up (ki, kj, kk)
            # in the purely_reals set.
            should_be_real = ((ki, kj, kk) == (ki_inv, kj_inv, kk_inv))
            # Check symmetry
            c     = lookup(ki,     kj,     kk    )
            c_inv = lookup(ki_inv, kj_inv, kk_inv)
            symmetry_broken = False
            if not (
                    np.isclose(c.real,  c_inv.real, rel_tol, abs_tol)
                and np.isclose(c.imag, -c_inv.imag, rel_tol, abs_tol)
            ):
                symmetry_broken = True
                if should_be_real:
                    # This grid point has a non-zero imaginary part
                    # while it ought to be purely real. Though this
                    # is counted as an error, we do not print anything,
                    # as a separate reality check will also
                    # be carried out.
                    pass
                else:
                    fancyprint(
                        f'Should be conjugate pair ({name}): '
                        f'[ki = {ki}, kj = {kj}, kk = {kk}] → {c} vs. '
                        f'[ki = {ki_inv}, kj = {kj_inv}, kk = {kk_inv}] → {c_inv}',
                        wrap=False,
                        file=sys.stderr,
                    )
            return symmetry_broken
        return check
    checks = []
    # Symmetry of inverting everything through the origin.
    # Tack on checking of zeros.
    checks.append(
        generate_check(
            lambda ki, kj, kk: True,
            lambda ki, kj, kk: (-ki, -kj, -kk),
            'inversion through origin',
        )
    )
    # Symmetry of inverting the negative x Nyquist plane
    # through its center.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist),
            lambda ki, kj, kk: (+ki, -kj, -kk),
            'inversion through center of ki = -nyquist plane',
        )
    )
    # Symmetry of inverting the negative y Nyquist plane
    # through its center.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist),
            lambda ki, kj, kk: (-ki, +kj, -kk),
            'inversion through center of kj = -nyquist plane',
        )
    )
    # Symmetry of inverting the positive z Nyquist plane
    # through its center.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kk == +nyquist),
            lambda ki, kj, kk: (-ki, -kj, +kk),
            'inversion through center of kk = +nyquist plane',
        )
    )
    # Symmetry of inverting the negative x, positive z Nyquist edge
    # through its center.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (+ki, -kj, +kk),
            'inversion through center of ki = -nyquist, kk = +nyquist edge',
        )
    )
    # Symmetry of inverting the negative y, positive z Nyquist edge
    # through its center.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (-ki, +kj, +kk),
            'inversion through center of kj = -nyquist, kk = +nyquist edge',
        )
    )
    # Reality of special points
    checks.append(check_reality)
    # Check symmetries throughoout the small grid
    symmetry_broken = False
    for check in checks:
        for ki, kj, kk in visit():
            symmetry_broken |= check(ki, kj, kk)
    # Check for zeros in the bulk, if requested
    if not allow_zeros:
        for ki, kj, kk in visit():
            symmetry_broken |= check_zero_within_embedding(ki, kj, kk)
    # Check that non-tabulated modes are nullified, if requested
    if pure_embedding:
        for ki, kj, kk in visit(nyquist_large):
            symmetry_broken |= check_nonzero_beyond_embedding(ki, kj, kk)
    # Check that the Nyquist planes are nullified, if requested
    if nullified_nyquist:
        for ki, kj, kk in visit(nyquist_large):
            symmetry_broken |= check_nonzero_at_nyquist(ki, kj, kk)
    # Tally up the total amount of information and compare
    # against the expected value, if requested.
    if count_information:
        # Count up the number of different real numbers within the grid,
        # including both the real and imaginary part
        # and disregarding signs.
        decimals = int(round(-log10(rel_tol)))
        information_full = len(set(np.round(
              list(np.abs(grid[:, :, 0::2].flatten()))
            + list(np.abs(grid[:, :, 1::2].flatten())),
            decimals,
        )))
        if gridsize_large > gridsize:
            if not pure_embedding:
                # The grid is embedded within a larger non-zero grid,
                # meaning that the above count contains lots of points
                # outside of the small grid. Discard the count.
                information_full = -1
        # We also iterate over the grid in a way that samples each
        # unique complex grid point once, recording the real values
        # (from both the real and imaginary parts).
        information = set()
        for ki, kj, kk in visit(sparse=True):
            c = lookup(ki, kj, kk)
            information.add(np.round(np.abs(c.real), decimals))
            information.add(np.round(np.abs(c.imag), decimals))
        information = len(information)
        # A general real, cubic grid consists of gridsize**3
        # unique real numbers, and so we expect the same number to
        # exist in the Fourier transformed grid. As some complex grid
        # points must be real, the number 0 is guaranteed to exist
        # in the Fourier grid. Disregarding the off change that some
        # complex mode is 0 (as in 0 + 0j), this ups the information
        # count by 1.
        # If we have nullified Nyquist planes, we need to count more
        # careful. Ignoring the Nyquist planes as well as the z DC
        # plane leaves us with points which do not have a conjugate
        # partner within the tabulation. This block of points has
        # gridsize - 1 points along the x and y directions and
        # nyquist - 1 points along the z direction. That this, it has
        #   block = 2*(nyquist - 1)*(gridsize - 1)**2
        # unique real numbers. The remaining z DC plane consists purely
        # of conjugate pairs. That is, it has (gridsize - 1)**2 complex
        # grid points, exactly half of which are redundant. The
        # exception is the origin, which we should count as contributing
        # two real numbers (though one of them is a zero). That is, the
        # z DC plane contains
        #   plane = (gridsize - 1)**2 + 1
        # unique real numbers. In total, we have
        #   block + plane = gridsize**3 + 3*gridsize*(1 - gridsize)
        # unique real numbers.
        if nullified_nyquist:
            information_expected = gridsize**3 + 3*gridsize*(1 - gridsize)
            information_expected_str = (
                f'{gridsize}³ + 3*{gridsize}(1 - {gridsize}) + 1 = {information_expected}'
            )
        else:
            information_expected = gridsize**3 + 1
            information_expected_str = f'{gridsize}³ + 1 = {information_expected}'
        if len({information_full, information, information_expected} - {-1}) != 1:
            symmetry_broken = True
            msg_information_full = ''
            if information_full != -1:
                msg_information_full = f'{information_full} (counted over entire grid), '
            fancyprint(
                f'Dispute about number of unique absolute values '
                f'of real and imaginary parts of grid points: '
                f'{msg_information_full}'
                f'{information} (found through sparse iteration), '
                f'while we expect {information_expected_str} for a random grid.',
                wrap=False,
                file=sys.stderr,
            )
        # As a further check, we want to tally up the number of unique
        # complex number in the grid, with complex conjugate pairs
        # deemed non-unique. Count up number of unique complex number
        # in the entire grid.
        n_unique_full = len(set(np.round(
            np.sqrt((grid[:, :, 0::2]**2 + grid[:, :, 1::2]**2).flatten()),
            decimals,
        )))
        if gridsize_large > gridsize:
            if pure_embedding:
                # The grid is embedded purely, i.e. the grid contains
                # zeros outside of the Nyquist frequencies. These all
                # count as an additional unique complex grid point.
                n_unique_full -= 1
            else:
                # The grid is embedded within a larger non-zero grid,
                # meaning that the above count contains lots of points
                # outside of the small grid. Discard the count.
                n_unique_full = -1
        # We also iterate over the grid in a way that samples each
        # unique complex grid point once, recording the absolute values.
        absolutes = set()
        for ki, kj, kk in visit(sparse=True):
            absolutes.add(np.abs(lookup(ki, kj, kk)))
        n_unique = len(absolutes)
        # The Fourier transform of a 3D random real grid has exactly
        # gridsize**3//2 + 4 complex grid points with unique absolute
        # values (see the docstring of visit() for details).
        # If the Nyquist planes are nullified, the z DC plane contains
        # ((gridsize - 1)**2 - 1)//2 unique complex points, plus the
        # origin. The remaining block contains
        # (nyquist - 1)*(gridsize - 1)**2 unique points.
        # In total we have
        #    (gridsize**3 + 3*gridsize*(1 - gridsize))//2
        # unique absolute values.
        if nullified_nyquist:
            n_unique_full -= 1  # Ignore zeros at Nyquist points
            n_unique      -= 1  # Ignore zeros at Nyquist points
            n_unique_expected = (gridsize**3 + 3*gridsize*(1 - gridsize))//2
            n_unique_expected_str = (
                f'({gridsize}³ + 3*{gridsize}(1 - {gridsize}))//2 = {n_unique_expected}'
            )
        else:
            n_unique_expected = gridsize**3//2 + 4
            n_unique_expected_str = f'{gridsize}³//2 + 4 = {n_unique_expected}'
        if len({n_unique_full, n_unique, n_unique_expected} - {-1}) != 1:
            symmetry_broken = True
            msg_n_unique_full = ''
            if n_unique_full != -1:
                msg_n_unique_full = f'{n_unique_full} (counted over entire grid), '
            fancyprint(
                f'Dispute about number of unique absolute values of complex grid points: '
                f'{msg_n_unique_full}'
                f'{n_unique} (found through sparse iteration), '
                f'while we expect {n_unique_expected} for a random grid.',
                wrap=False,
                file=sys.stderr,
            )
    masterprint('done')
    return not symmetry_broken

# Function for differentiating domain grids
@cython.pheader(
    # Arguments
    grid='double[:, :, ::1]',
    dim='int',
    order='int',
    Δx='double',
    buffer_or_buffer_name=object,  # double[:, :, ::1] or int or str
    direction=str,
    do_ghost_communication='bint',
    # Locals
    i='Py_ssize_t',
    indices_i_m='Py_ssize_t*',
    indices_i_p='Py_ssize_t*',
    indices_j_m='Py_ssize_t*',
    indices_j_p='Py_ssize_t*',
    indices_k_m='Py_ssize_t*',
    indices_k_p='Py_ssize_t*',
    j='Py_ssize_t',
    k='Py_ssize_t',
    step='Py_ssize_t',
    steps_i='Py_ssize_t*',
    steps_j='Py_ssize_t*',
    steps_k='Py_ssize_t*',
    value='double',
    ᐁgrid_dim='double[:, :, ::1]',
    returns='double[:, :, ::1]',
)
def diff_domaingrid(
    grid, dim, order,
    Δx=1, buffer_or_buffer_name=0, direction='forward', do_ghost_communication=True,
):
    """This function differentiates a given domain grid along the dim'th
    dimension once through finite differencing. The passed grid must
    include correctly populated ghost points, and the returned grid will
    contain correctly populated ghost points as well unless
    do_ghost_communication is False.
    The order argument specifies the order of the differentiation,
    meaning the number of neighbouring grid points used to approximate
    the first derivative. For odd orders, the differentiation cannot be
    symmetric, and so the direction should be specified as either
    'forward' or 'backward'.
    To achieve proper units, the physical grid spacing may be specified
    as Δx. If not given, grid units (Δx == 1) are used. The
    buffer_or_buffer_name argument can be a buffer to store the results,
    or alternatively the name of a buffer (retrieved via
    communication.get_buffer()) as an int or str. If a buffer is
    supplied, the result of the differentiations will be added to this
    buffer. If a buffer should be fetched automatically, this will be
    nullify before the differentiation. Note that the buffer has to be
    contiguous.
    Note that a grid cannot be differentiated in-place by passing the
    grid as both the first and fifth argument, as the differentiation
    of each grid point requires information from the original
    (non-differentiated) grid.
    """
    # Sanity checks on input
    if dim not in (0, 1, 2):
        abort(f'diff_domaingrid() called with dim = {dim} ∉ {{0, 1, 2}}')
    if direction not in ('forward', 'backward'):
        abort(
            f'diff_domaingrid() called with direction = {direction} '
            f'∉ {{"forward", "backward"}}'
        )
    # If no buffer is supplied, fetch the buffer with the name
    # given by buffer_or_buffer_name.
    if isinstance(buffer_or_buffer_name, (int, str)):
        ᐁgrid_dim = get_buffer(asarray(grid).shape, buffer_or_buffer_name, nullify=True)
    else:
        ᐁgrid_dim = buffer_or_buffer_name
        if not asarray(ᐁgrid_dim).shape == asarray(grid).shape:
            abort(
                f'diff_domaingrid() called with grid of shape {asarray(grid).shape}'
                f'and buffer of different shape {asarray(ᐁgrid_dim).shape}'
            )
    # Reuse global arrays
    steps_i     = diff_steps_i
    steps_j     = diff_steps_j
    steps_k     = diff_steps_k
    indices_i_m = diff_indices_i_m
    indices_i_p = diff_indices_i_p
    indices_j_m = diff_indices_j_m
    indices_j_p = diff_indices_j_p
    indices_k_m = diff_indices_k_m
    indices_k_p = diff_indices_k_p
    # Set steps along each dimension
    for step in range(1, ℤ[order//2 + 1]):
        steps_i[step] = step*ℤ[dim == 0]
        steps_j[step] = step*ℤ[dim == 1]
        steps_k[step] = step*ℤ[dim == 2]
    # Loop over the local bulk (i.e. not ghosts) of the grid
    for i in range(nghosts, ℤ[grid.shape[0] - nghosts]):
        # Set grid indices along the x direction
        with unswitch:
            if order == 1:
                indices_i_p[0] = i + ℤ[ℤ[dim == 0] and ℤ[direction == 'forward' ]]
                indices_i_m[0] = i - ℤ[ℤ[dim == 0] and ℤ[direction == 'backward']]
            else:
                for step in range(1, ℤ[order//2 + 1]):
                    indices_i_p[step] = i + ℤ[steps_i[step]]
                    indices_i_m[step] = i - ℤ[steps_i[step]]
        for j in range(nghosts, ℤ[grid.shape[1] - nghosts]):
            # Set grid indices along the y direction
            with unswitch:
                if order == 1:
                    indices_j_p[0] = j + ℤ[ℤ[dim == 1] and ℤ[direction == 'forward' ]]
                    indices_j_m[0] = j - ℤ[ℤ[dim == 1] and ℤ[direction == 'backward']]
                else:
                    for step in range(1, ℤ[order//2 + 1]):
                        indices_j_p[step] = j + ℤ[steps_j[step]]
                        indices_j_m[step] = j - ℤ[steps_j[step]]
            for k in range(nghosts, ℤ[grid.shape[2] - nghosts]):
                # Set grid indices along the z direction
                with unswitch:
                    if order == 1:
                        indices_k_p[0] = k + ℤ[ℤ[dim == 2] and ℤ[direction == 'forward' ]]
                        indices_k_m[0] = k - ℤ[ℤ[dim == 2] and ℤ[direction == 'backward']]
                    else:
                        for step in range(1, ℤ[order//2 + 1]):
                            indices_k_p[step] = k + ℤ[steps_k[step]]
                            indices_k_m[step] = k - ℤ[steps_k[step]]
                # Do the finite differencing
                with unswitch:
                    if order == 1:
                        value = ℝ[1/Δx]*(
                            + grid[indices_i_p[0], indices_j_p[0], indices_k_p[0]]
                            - grid[indices_i_m[0], indices_j_m[0], indices_k_m[0]]
                        )
                    elif order == 2:
                        value = ℝ[1/(2*Δx)]*(
                            + grid[indices_i_p[1], indices_j_p[1], indices_k_p[1]]
                            - grid[indices_i_m[1], indices_j_m[1], indices_k_m[1]]
                        )
                    elif order == 4:
                        value = ℝ[1/(12*Δx)]*(
                            + 8*(
                                + grid[indices_i_p[1], indices_j_p[1], indices_k_p[1]]
                                - grid[indices_i_m[1], indices_j_m[1], indices_k_m[1]]
                            )
                            - (
                                + grid[indices_i_p[2], indices_j_p[2], indices_k_p[2]]
                                - grid[indices_i_m[2], indices_j_m[2], indices_k_m[2]]
                            )
                        )
                    elif order == 6:
                        value = ℝ[1/(60*Δx)]*(
                            45*(
                                + grid[indices_i_p[1], indices_j_p[1], indices_k_p[1]]
                                - grid[indices_i_m[1], indices_j_m[1], indices_k_m[1]]
                            )
                            - 9*(
                                + grid[indices_i_p[2], indices_j_p[2], indices_k_p[2]]
                                - grid[indices_i_m[2], indices_j_m[2], indices_k_m[2]]
                            )
                            + (
                                + grid[indices_i_p[3], indices_j_p[3], indices_k_p[3]]
                                - grid[indices_i_m[3], indices_j_m[3], indices_k_m[3]]
                            )
                        )
                    elif order == 8:
                        value = ℝ[1/(280*Δx)]*(
                            224*(
                                + grid[indices_i_p[1], indices_j_p[1], indices_k_p[1]]
                                - grid[indices_i_m[1], indices_j_m[1], indices_k_m[1]]
                            )
                            - 56*(
                                + grid[indices_i_p[2], indices_j_p[2], indices_k_p[2]]
                                - grid[indices_i_m[2], indices_j_m[2], indices_k_m[2]]
                            )
                            + 32./3.*(
                                + grid[indices_i_p[3], indices_j_p[3], indices_k_p[3]]
                                - grid[indices_i_m[3], indices_j_m[3], indices_k_m[3]]
                            )
                            - (
                                + grid[indices_i_p[4], indices_j_p[4], indices_k_p[4]]
                                - grid[indices_i_m[4], indices_j_m[4], indices_k_m[4]]
                            )
                        )
                    else:
                        abort(
                            f'diff_domaingrid() called with '
                            f'order = {order} ∉ {{1, 2, 4, 6, 8}}'
                        )
                        value = 0  # To satisfy the compiler
                # Update the buffer with the result
                # of the differentiation.
                ᐁgrid_dim[i, j, k] += value
    # Populate the ghost points with copies of their
    # corresponding actual points.
    if do_ghost_communication:
        communicate_ghosts(ᐁgrid_dim, '=')
    return ᐁgrid_dim
# Allocate global arrays used by the diff_domaingrid() function
cython.declare(
    highest_differentiation_order_implemented='int',
    diff_steps_i='Py_ssize_t*',
    diff_steps_j='Py_ssize_t*',
    diff_steps_k='Py_ssize_t*',
    diff_indices_i_m='Py_ssize_t*',
    diff_indices_i_p='Py_ssize_t*',
    diff_indices_j_m='Py_ssize_t*',
    diff_indices_j_p='Py_ssize_t*',
    diff_indices_k_m='Py_ssize_t*',
    diff_indices_k_p='Py_ssize_t*',
)
highest_differentiation_order_implemented = 8
diff_steps_i     = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_steps_j     = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_steps_k     = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_i_m = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_i_p = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_j_m = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_j_p = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_k_m = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))
diff_indices_k_p = malloc((highest_differentiation_order_implemented//2 + 1)*sizeof('Py_ssize_t'))

# Below we define weight functions for one-dimensional particle
# interpolation. Each function takes in a coordinate x which must be
# pre-scaled so that it lies in the interval
#   nghosts - ½ < x < shape[dim] - nghosts - ½
# where shape is the full shape (with ghost layers) of the grid taking
# part in the interpolation. The second argument is an array for storing
# the resulting weights.
# The return value is an index into the grid, such that the weight
# associated with that grid point is weights[0]. The full weighted sum
# is then weights[0]*grid[index] + weights[1]*grid[index + 1] + ...
#
# Nearest grid point (NGP) interpolation (order 1)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    index='Py_ssize_t',
    returns='Py_ssize_t',
)
def set_weights_NGP(x, weights):
    index = int(x + 0.5)
    weights[0] = 1
    return index
# Cloud-in-cell (CIC) interpolation (order 2)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    index='Py_ssize_t',
    returns='Py_ssize_t',
)
def set_weights_CIC(x, weights):
    index = int(x)
    dist = x - index  # Distance between leftmost grid point and x; 0 <= dist < 1
    weights[0] = 1 - dist
    weights[1] = dist
    return index
# Triangular-shaped cloud (TSC) interpolation (order 3)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    index='Py_ssize_t',
    returns='Py_ssize_t',
)
def set_weights_TSC(x, weights):
    index = int(x + 0.5)
    dist = x - index  # Distance between center grid point and x; -0.5 <= dist < 0.5
    index -= 1
    weights[0] = 0.5*(0.5 - dist)**2
    weights[1] = 0.75 - dist**2
    weights[2] = 1 - (weights[0] + weights[1])
    return index
# Piecewise cubic spline (PCS) interpolation (order 4)
@cython.header(
    # Arguments
    x='double',
    weights='double*',
    # Locals
    dist='double',
    index='Py_ssize_t',
    tmp='double',
    tmp2='double',
    tmp3='double',
    returns='Py_ssize_t',
)
def set_weights_PCS(x, weights):
    index = int(x)
    index -= 1
    dist = x - index  # Distance between leftmost grid point and x; 1 <= dist < 2
    tmp = 2 - dist
    tmp2 = tmp**2
    tmp3 = tmp*tmp2
    weights[0] = 1./6.*tmp3
    weights[2] = 2./3. - tmp2 + 0.5*tmp3
    weights[3] = 1./6.*(dist - 1)**3
    weights[1] = 1 - (weights[0] + weights[2] + weights[3])
    return index
# Allocate global weights arrays to be used with the above functions
cython.declare(
    highest_implemented_interpolation_order='int',
    weights_x='double*',
    weights_y='double*',
    weights_z='double*',
)
highest_interpolation_order_implemented = 4  # PCS
weights_x = malloc(highest_interpolation_order_implemented*sizeof('double'))
weights_y = malloc(highest_interpolation_order_implemented*sizeof('double'))
weights_z = malloc(highest_interpolation_order_implemented*sizeof('double'))
