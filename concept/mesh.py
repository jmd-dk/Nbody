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
    It is assumed that the grid is non-periodic (that is, the first and
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
    if not (1 <= order <= 4):
        abort(
            f'interpolate_domaingrid_to_particles() called '
            f'with order = {order} ∉ {{1, 2, 3, 4}}'
        )
    # Offsets needed for the interpolation
    cellsize = domain_size_x/(grid.shape[0] - ℤ[2*nghosts])  # We have cubic grid cells
    offset_x = domain_start_x - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered)*cellsize]
    offset_y = domain_start_y - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered)*cellsize]
    offset_z = domain_start_z - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered)*cellsize]
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
            else:  # order == 4  # PCS interpolation
                index_i = set_weights_PCS(x, weights_x)
                index_j = set_weights_PCS(y, weights_y)
                index_k = set_weights_PCS(z, weights_z)
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
# upstream grids.
@cython.pheader(
    # Arguments
    components=list,
    gridsizes_upstream=list,
    gridsize_global='Py_ssize_t',
    quantity=str,
    order='int',
    ᔑdt=dict,
    deconvolve='bint',
    interlace='bint',
    output_space=str,
    do_ghost_communication='bint',
    # Locals
    component='Component',
    fft_factor='double',
    fluid_components=list,
    grid_global='double[:, :, ::1]',
    grid_upstream='double[:, :, ::1]',
    gridshape_upstream_local=tuple,
    gridsize_upstream='Py_ssize_t',
    group=dict,
    groups=dict,
    particle_components=list,
    shift='double',
    shift_index='int',
    slab_global='double[:, :, ::1]',
    returns='double[:, :, ::1]',
)
def interpolate_upstream(
    components, gridsizes_upstream, gridsize_global, quantity, order,
    ᔑdt=None, deconvolve=True, interlace=False,
    output_space='real', do_ghost_communication=True,
):
    """Given a list of components, a list of corresponding upstream grid
    sizes and a single global grid size, this function interpolates the
    components onto a global grid by first interpolating directly onto
    the upstream grids, which are then added together in Fourier space
    to obtain the global grid.

    The (particle) interpolation order is given by the order argument.
    No upstream interpolation takes place for fluid components,
    i.e. the passed υpstream grid sizes are assumed to equal the
    fluid grid size for fluid components.

    Whether to apply particle deconvolution and interlacing can be
    specified by deconvolve and interlace.
    The deconvolution order will be the same as the interpolation order
    (the order argument).

    The returned grid may either be a real space domain grid or a
    Fourier space slab, depending on the value of output_space
    (either 'real' or 'Fourier'). If 'real', the domain grids will have
    properly populated ghost points if do_ghost_communication is True.

    The quantity argument determines what should be interpolated onto
    the grid(s). Valid values are:
    'ρ': The returned grid(s) will hold physical densities. Note that
        ρ = a**(-3(1 + w_eff))*ϱ. Note that this physical 'ρ' is always
        preferable to the conserved 'ϱ' when multiple components are to
        be interpolated together, as only 'ρ' is additive across
        components/species. Each particle will contribute with
        a**(-3*w_eff)*mass/V_cell, a**(-3*w_eff)*mass being the current
        mass of the particle (a**(-3*w_eff) taking decay into account)
        and V_cell = (a*boxsize/gridsize)**3 being the physical grid
        cell volume. In total, each particle contribute with
        a**(-3*(1 + w_eff))*(gridsize/boxsize)**3*mass. Each fluid cell
        will contribute with a**(-3*(1 + w_eff))*ϱᵢⱼₖ*V_cell_fluid/V_cell,
        where a**(-3*(1 + w_eff))*ϱᵢⱼₖ = ρᵢⱼₖ is the physical density of
        fluid cell [i, j, k] and
        V_cell_fluid = (a*boxsize/gridsize_fluid)**3 is the physical
        cell volume of the fluid grid. In total, each fluid cell
        contribute with
        a**(-3*(1 + w_eff))*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ
    'a²ρ': The returned grid(s) will hold physical densities times the
        square of the scale factor. From the 'ρ' entry above, we then
        have that each particle will contribute with
        a**(-3*w_eff - 1)*(gridsize/boxsize)**3*mass and that each fluid
        cell will contribute with
        a**(-3*w_eff - 1)*(gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    'ϱ': The returned grid(s) will hold the conserved densities. From
        the 'ρ' entry above, we then have that each particle will
        contribute with ϱ = (gridsize/boxsize)**3*mass and that each
        fluid cell will contribute with (gridsize/gridsize_fluid)**3*ϱᵢⱼₖ.
    'Jx', 'Jy', 'Jz'; described for Jx: The returned grid(s) will hold
        the conserved momentum density Jₓ = a**4*(ρ + c⁻²P)*uₓ. As this
        is built from physical quantities, this is additive across
        components/species. For particles we set P = 0, leaving
        Jₓ = a**4*ρ*uₓ. The canonical momentum momₓ and peculiar velocity
        uₓ is related by momₓ = a*(a**(-3*w_eff)*mass)*uₓ, and so from
        the particle construction of ρ above we get that each particle
        contribute with (gridsize/boxsize)**3*momₓ. As each fluid cell
        already stores Jₓ, they contribute by
        V_cell_fluid/V_cell*Jₓ = (gridsize/gridsize_fluid)**3*Jₓ.
    In all of the above, expressions involving a = a(t) will be
    evaluated at the current universal time, unless a dict ᔑdt of time
    step integrals is passed, in which case the expressions will be
    integrated over the time step.
    """
    output_space = output_space.lower()
    if output_space not in ('real', 'fourier'):
        abort(
            f'interpolate_upstream() got output_space = "{output_space}" ∉ {{"real", "Fourier"}}'
        )
    # Group components according to their upstream grid size.
    # The order does not matter, except that we want the group with the
    # downstream grid size equal to the global grid size to be
    # the first, if indeed such a group exist.
    groups = group_components(components, gridsizes_upstream, [gridsize_global, ...])
    # The global Fourier slabs will be properly initialized below
    slab_global = None
    # For each group, interpolate the components onto upstream grids,
    # Fourier transform these upstream grids and add them
    # to the global slabs.
    for gridsize_upstream, group in groups.items():
        gridshape_upstream_local = get_gridshape_local(gridsize_upstream)
        particle_components = group.get('particles', [])
        fluid_components    = group.get('fluid',     [])
        # Normalization factor needed after a forward FFT
        fft_factor = float(gridsize_upstream)**(-3)
        # Fluid components
        if fluid_components:
            # Copy fluid components onto upstream grid
            grid_upstream = get_buffer(
                gridshape_upstream_local, 'grid_updownstream',
                nullify=False,
            )
            combine_fluids(fluid_components, grid_upstream, quantity, ᔑdt, fft_factor)
            # Transform the upstream grid to Fourier space
            # and add it to the global Fourier slabs.
            slab_global = add_upstream_to_global_slabs(
                grid_upstream, slab_global, gridsize_upstream, gridsize_global,
            )
        # Particle components
        if particle_components:
            # If interlacing is to be used, the particle interpolation
            # (and subsequent Fourier transformation) needs to be
            # carried out twice; with shift = 0 (standard)
            # and with shift = ±0.5 (+ chosen for cell-centred,
            # - for cell-vertex).
            for shift_index in range(1 + interlace):
                shift = ℝ[cell_centered - 0.5]*shift_index
                # Interpolate particle components onto upstream grid
                grid_upstream = get_buffer(
                    gridshape_upstream_local, 'grid_updownstream',
                    nullify=True,
                )
                for component in particle_components:
                    interpolate_particles(
                        component, gridsize_upstream, grid_upstream, quantity, order, ᔑdt,
                        shift, fft_factor, do_ghost_communication=False,
                    )
                communicate_ghosts(grid_upstream, '+=')
                # Transform the upstream grid to Fourier space,
                # perform deconvolution and interlacing and add
                # the result to the global Fourier slabs.
                slab_global = add_upstream_to_global_slabs(
                    grid_upstream, slab_global, gridsize_upstream, gridsize_global,
                    deconv_order=ℤ[deconvolve*order],
                    interlace_flag=(interlace + shift_index),
                )
    # Global Fourier slabs complete. Note that we do not have to nullify
    # the Nyquist planes of these global slabs as we have nullified the
    # Nyquist planes of all upstream slabs from which the global slabs
    # are built.
    # Return now if Fourier space is what we want.
    if output_space == 'fourier':
        return slab_global
    # Fourier transform the global slabs to real space
    fft(slab_global, 'backward')
    # Domain-decompose and return the global real space grid
    grid_global = get_buffer(get_gridshape_local(gridsize_global), 'grid_global')
    domain_decompose(slab_global, grid_global, do_ghost_communication=do_ghost_communication)
    return grid_global

# Helper function for interpolate_upstream(), taking in a real space
# upstream grid and adding its Fourier modes to the global slabs.
@cython.header(
    # Arguments
    grid_upstream='double[:, :, ::1]',
    slab_global='double[:, :, ::1]',
    gridsize_upstream='Py_ssize_t',
    gridsize_global='Py_ssize_t',
    deconv_order='int',
    interlace_flag='int',
    # Locals
    nullification=str,
    operation=str,
    slab_upstream='double[:, :, ::1]',
    use_upstream_as_global='bint',
    returns='double[:, :, ::1]',
)
def add_upstream_to_global_slabs(
    grid_upstream, slab_global, gridsize_upstream, gridsize_global,
    deconv_order=0, interlace_flag=0,
):
    # If the global slabs have yet to be initialized, we must do so
    # within this function. If at the same time the global and upstream
    # grid sizes are equal, we will use the upstream slabs directly as
    # the global slabs, rather than performing a copy.
    use_upstream_as_global = (slab_global is None and gridsize_global == gridsize_upstream)
    # Transform upstream slabs to Fourier space,
    # using the 'slab_global' or 'slab_updownstream' buffer
    # depending on the passed use_upstream_as_global.
    slab_upstream = slab_decompose(
        grid_upstream,
        'slab_global' if use_upstream_as_global else 'slab_updownstream',
        prepare_fft=True,
    )
    fft(slab_upstream, 'forward')
    # Ensure nullified Nyquist planes
    nullify_modes(slab_upstream, 'nyquist')
    # Perform deconvolution and interlacing on the upstream slabs
    # and assign/update the global slabs to/with the results.
    if use_upstream_as_global:
        # Perform in-place deconvolution and interlacing
        deconvolve_and_interlace(slab_upstream, deconv_order, interlace_flag)
        # The upstream slabs are really the global slabs
        slab_global = slab_upstream
    else:
        # Add upstream slabs to the global slabs,
        # performing deconvolution and interlacing on the way.
        # If the global slabs have yet to be initialized,  we set the
        # values directly, rather than adding to existing ones.
        operation = '+='
        if slab_global is None:
            operation = '='
            # Fetch slab to be used as the global one. Depending on
            # whether this is constructed without scaling or through
            # downscaling or upscaling of the υpstream slab, different
            # sets of elements needs to be nullified. We could of course
            # just always perform a complete nullification.
            if gridsize_global == gridsize_upstream:
                nullification = 'all'
            elif gridsize_global < gridsize_upstream:
                # Global slabs constructed through downscaling of the
                # upstream slabs. All elements of slab_global will be
                # set except the Nyquist planes, which then have
                # to be nullified.
                nullification = 'nyquist'
            else:
                # Global slabs constructed through upscaling of the
                # upstream slabs. Only elements within a cube centred
                # at the origin and of size gridsize_upstream will be
                # set. We then need to nullify all elements beyond
                # this cube.
                nullification = f'beyond cube of |k| < {gridsize_upstream//2}'
            slab_global = get_fftw_slab(gridsize_global, 'slab_global', nullification)
        copy_modes(slab_upstream, slab_global, deconv_order, interlace_flag, operation)
    return slab_global

# Function for grouping components according to
# associated grid sizes and their representation.
def group_components(components, gridsizes, gridsizes_order=()):
    """Given a list of components and a list of corresponding grid
    sizes, this function returns a dict of components, grouped by their
    grid size and representation, in the format
        {
            <gridsize 0>: {
                'particles': [<component 0>],
            },
            <gridsize 2>: {
                'particles': [<component 4>, <component 2>],
                'fluid'    : [<component 1>],
            },
            <gridsize 1>: {
                'fluid': [<component 5>, <component 3>],
            },
        }
    A specific ordering of the grid sizes can be enforced by specifying
    gridsizes_order. The above example can be obtained by one of
        gridsizes_order = [..., <gridsize 2>, <gridsize 1>]
        gridsizes_order = [<gridsize 0>, ..., <gridsize 1>]
        gridsizes_order = [<gridsize 0>, <gridsize 2>, ...]
    The ellipsis (...) signals any grid sizes not explicitly listed.
    The passed gridsizes_order must contain exactly one ellipsis.
    """
    # Look up previous result in cache
    key = (tuple(components), tuple(gridsizes), tuple(gridsizes_order))
    groups = component_groups_cache.get(key)
    if groups is not None:
        return groups
    # Perform the grouping
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for component, gridsize in zip(components, gridsizes):
        groups[gridsize][component.representation].append(component)
    # Create list of ordered grid sizes
    gridsizes_order = any2list(gridsizes_order)
    if not gridsizes_order:
        gridsizes_order = [...]
    if gridsizes_order.count(...) != 1:
        abort(
            f'group_components() got gridsizes_order = {gridsizes_order}, '
            f'but this must contain exactly one ellipsis (...)'
        )
    index = gridsizes_order.index(...)
    gridsizes_first = gridsizes_order[:index]
    gridsizes_last = gridsizes_order[index + 1:]
    gridsizes_ordered = gridsizes_first.copy()
    for gridsize in gridsizes:
        if gridsize in gridsizes_ordered or gridsize in gridsizes_last:
            continue
        gridsizes_ordered.append(gridsize)
    gridsizes_ordered += gridsizes_last
    # Enforce correct ordering of the groups based on their grid sizes
    groups = {
        gridsize: dict(groups[gridsize])
        for gridsize in gridsizes_ordered
        if gridsize in gridsizes
    }
    # Cache and return result
    component_groups_cache[key] = groups
    return groups
# Cache used by the group_components() function
cython.declare(component_groups_cache=dict)
component_groups_cache = {}

# Function for resizing a real space domain grid
# or Fourier space slabs, i.e. change the grid size.
@cython.pheader(
    # Arguments
    grid_or_slab='double[:, :, ::1]',
    gridsize_new='Py_ssize_t',
    input_space=str,
    output_space=str,
    internal_slab_or_buffer_name=object,  # double[:, :, ::1] or int or str
    output_grid_or_buffer_name=object,    # double[:, :, ::1] or int or str
    output_slab_or_buffer_name=object,    # double[:, :, ::1] or int or str
    inplace='bint',
    apply_forward_fft_normalization='bint',
    do_ghost_communication='bint',
    # Locals
    grid='double[:, :, ::1]',
    grid_new='double[:, :, ::1]',
    gridsize='Py_ssize_t',
    nullification=str,
    slab='double[:, :, ::1]',
    slab_new='double[:, :, ::1]',
    returns='double[:, :, ::1]',
)
def resize_grid(
    grid_or_slab, gridsize_new, input_space,
    output_space='same', internal_slab_or_buffer_name='slab_updownstream',
    output_grid_or_buffer_name='grid_global', output_slab_or_buffer_name='slab_global',
    inplace=True, apply_forward_fft_normalization=True, do_ghost_communication=True,
):
    """Given a real space domain grid or Fourier space slabs (specified
    by input_space = 'real' or input_space = 'Fourier') and a new grid
    size (gridsize_new), the grid/slab data will be copied over to a new
    grid/slab with the new grid size. Whether to return a real space
    grid or Fourier space slabs is specified by output_space.
    If the output space is different from the input space, a new
    grid/slab is needed to store the resized result. The grid/slab to
    use for this is specified via output_grid_or_buffer_name (for real
    space output) and output_slab_or_buffer_name (for Fourier space
    output).
    When a resizing is to be performed (the original and new grid size
    are different) and the input is in real space, an additional
    internal slab is needed, as the resizing is carried out in Fourier
    space. The slab to use for this is specified as
    internal_slab_or_buffer_name.
    If the input and output space are the same and the original and new
    grid size are also the same, the passed grid/slab is returned back
    as is. No additional grid/slab will be used. Also, if the input
    space is Fourier, the output space is real and the original and new
    grid size are the same, the input slab will be transformed in-place.
    To guarantee that the passed data is left untouched and that a new
    grid/slab is returned, set inplace = False.
    """
    # Handle the input and output space
    input_space = input_space.lower()
    if input_space == 'real':
        grid = grid_or_slab
        gridsize = (grid.shape[0] - ℤ[2*nghosts])*domain_subdivisions[0]
    elif input_space == 'fourier':
        slab = grid_or_slab
        gridsize = slab.shape[1]
    else:
        abort(f'resize_grid() got input_space = "{input_space}" ∉ {{"real", "Fourier"}}')
    output_space = output_space.lower()
    if output_space == 'same':
        output_space = input_space
    elif output_space not in ('real', 'fourier'):
        abort(f'resize_grid() got output_space = "{output_space}" ∉ {{"real", "Fourier"}}')
    # No resizing is required if the original and new grid size are the
    # same. If the input and output space are also the same, just return
    # the passed grid. Otherwise do the required FFT.
    if gridsize == gridsize_new:
        if input_space == output_space:
            if inplace:
                # In-place real to real or Fourier to Fourier
                return grid_or_slab
            elif input_space == 'real':
                # Out-of-place real to real
                if isinstance(output_grid_or_buffer_name, (int, str)):
                    grid_new = get_buffer(get_gridshape_local(gridsize), output_grid_or_buffer_name)
                else:
                    grid_new = output_grid_or_buffer_name
                grid_new[...] = grid
                return grid_new
            else:  # input_space == 'fourier'
                # Out-of-place Fourier to Fourier
                if isinstance(output_slab_or_buffer_name, (int, str)):
                    slab_new = get_fftw_slab(gridsize, output_slab_or_buffer_name)
                else:
                    slab_new = output_slab_or_buffer_name
                slab_new[...] = slab
                return slab_new
        elif output_space == 'fourier':
            # From real to Fourier
            slab_new = slab_decompose(grid, output_slab_or_buffer_name, prepare_fft=True)
            fft(slab_new, 'forward', apply_forward_fft_normalization)
            return slab_new
        else:  # output_space == 'real'
            # From Fourier to real
            if not inplace:
                if isinstance(internal_slab_or_buffer_name, (int, str)):
                    slab_internal = get_fftw_slab(gridsize, internal_slab_or_buffer_name)
                else:
                    slab_internal = internal_slab_or_buffer_name
                slab_internal[...] = slab
                slab = slab_internal
            fft(slab, 'backward')
            grid_new = domain_decompose(
                slab, output_grid_or_buffer_name, do_ghost_communication,
            )
            return grid_new
    # Resizing should take place. Go to Fourier space.
    if input_space == 'real':
        slab = slab_decompose(grid, internal_slab_or_buffer_name, prepare_fft=True)
        fft(slab, 'forward', apply_forward_fft_normalization)
    # Perform the resizing by copying the slab values into another
    # slab decomposed grid with the new grid size.
    if isinstance(output_slab_or_buffer_name, (int, str)):
        slab_new = get_fftw_slab(gridsize_new, output_slab_or_buffer_name)
    else:
        slab_new = output_slab_or_buffer_name
    if gridsize_new < gridsize:
        # The grid is to be downscaled. All modes except the Nyquist
        # ones will be set and so these need to be explicitly nullified.
        nullification = 'nyquist'
    else:  # gridsize_new > gridsize
        # The grid is to be upscaled. New modes beyond the ones in the
        # original grid will not be set and so these need to be
        # explicitly nullified.
        nullification = f'beyond cube of |k| < {gridsize//2}'
    nullify_modes(slab_new, nullification)
    copy_modes(slab, slab_new, operation='=')
    # Return the slab as is or transform to real space and return the
    # resulting grid, depending on the output space.
    if output_space == 'real':
        fft(slab_new, 'backward')
        grid_new = domain_decompose(slab_new, output_grid_or_buffer_name, do_ghost_communication)
        return grid_new
    else:  # output_space == 'fourier'
        return slab_new

# Function for adding the Fourier modes of one slab to another
@cython.pheader(
    # Arguments
    slab_from='double[:, :, ::1]',
    slab_onto='double[:, :, ::1]',
    deconv_order='int',
    interlace_flag='int',
    operation=str,
    # Locals
    cosθ='double',
    factor='double',
    gridsize_from='Py_ssize_t',
    gridsize_large='Py_ssize_t',
    gridsize_onto='Py_ssize_t',
    gridsize_small='Py_ssize_t',
    half_subslab_recv='double[:, :, ::1]',
    i_from_bgn='Py_ssize_t',
    i_from_end='Py_ssize_t',
    i_onto_bgn='Py_ssize_t',
    i_onto_end='Py_ssize_t',
    i_ranges_from=list,
    i_ranges_large=list,
    i_ranges_onto=list,
    i_ranges_small=list,
    i_small_bgn='Py_ssize_t',
    i_small_end='Py_ssize_t',
    im='double',
    index='Py_ssize_t',
    index_from='Py_ssize_t',
    index_large='Py_ssize_t',
    index_onto='Py_ssize_t',
    index_recv='Py_ssize_t',
    index_small='Py_ssize_t',
    j_from_bgn='Py_ssize_t',
    j_from_end='Py_ssize_t',
    j_global_large='Py_ssize_t',
    j_onto_bgn='Py_ssize_t',
    j_onto_end='Py_ssize_t',
    j_small_bgn='Py_ssize_t',
    j_small_end='Py_ssize_t',
    k_end='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    local='bint',
    nyquist_large='Py_ssize_t',
    nyquist_small='Py_ssize_t',
    rank_from='int',
    rank_onto='int',
    re='double',
    recv='bint',
    request=object,  # mpi4py.MPI.Request
    send='bint',
    sinθ='double',
    slab_from_ptr='double*',
    slab_from_size_i='Py_ssize_t',
    slab_from_size_j='Py_ssize_t',
    slab_from_size_k='Py_ssize_t',
    slab_large='double[:, :, ::1]',
    slab_large_size_i='Py_ssize_t',
    slab_large_size_j='Py_ssize_t',
    slab_large_size_k='Py_ssize_t',
    slab_onto_ptr='double*',
    slab_onto_size_i='Py_ssize_t',
    slab_onto_size_j='Py_ssize_t',
    slab_onto_size_k='Py_ssize_t',
    slab_read_ptr='double*',
    slab_small='double[:, :, ::1]',
    slab_small_size_i='Py_ssize_t',
    slab_small_size_j='Py_ssize_t',
    slab_small_size_k='Py_ssize_t',
    subslabs_dict_from=dict,
    subslabs_dict_large=dict,
    subslabs_dict_onto=dict,
    subslabs_dict_small=dict,
    subslabs_ranges_from=list,
    subslabs_ranges_onto=list,
    upscaling='bint',
    θ='double',
    θ_total='double',
    returns='double[:, :, ::1]',
)
def copy_modes(slab_from, slab_onto, deconv_order=0, interlace_flag=0, operation='='):
    """The Fourier modes of slab_from will be copied into slab_onto.
    Deconvolution will be performed according to deconv_order.
    Set operation = '+=' to add the modes from slab_from to slab_onto,
    rather than overwriting them. By default no interlacing is
    performed. To carry out interlacing you need to call this function
    twice; once with the non-shifted grid (as slab_from) and
    interlace_flag = 1, and once with the shifted grid and
    interlace_flag = 2, with at least the last call using
    operation = '+=' to not overwrite the results from the first call.
    With interlace_flag = 1, the values of slab_from are simply
    multiplied by 0.5 prior to adding them to slab_onto.
    With interlace_flag = 2, the phases of the (complex) values are
    additionally rotated as defined by harmonic averaging.
    """
    # Extract pointers
    slab_onto_ptr = cython.address(slab_onto[:, :, :])
    slab_from_ptr = cython.address(slab_from[:, :, :])
    # Extract grid sizes
    slab_onto_size_j, slab_onto_size_i, slab_onto_size_k = asarray(slab_onto).shape
    slab_from_size_j, slab_from_size_i, slab_from_size_k = asarray(slab_from).shape
    gridsize_from, gridsize_onto = slab_from_size_i, slab_onto_size_i
    # Consider the easy case of equal grid sizes separately
    if gridsize_from == gridsize_onto:
        # As slab_from and slab_onto have identical grid sizes,
        # we can copy the Fourier modes locally,
        # i.e. without any MPI communication.
        if deconv_order == 0 and interlace_flag != 2:
            # Deconvolution should not be performed and no phase shift
            # should be performed due to interlacing.
            # Combine directly.
            for index in range(slab_onto_size_j*slab_onto_size_i*slab_onto_size_k):
                # Set the factor due to interlacing
                factor = 1
                with unswitch:
                    if 𝔹[interlace_flag > 0]:
                        factor = 0.5
                # Write to slab_onto
                with unswitch:
                    if 𝔹[operation == '=']:
                        slab_onto_ptr[index] = factor*slab_from_ptr[index]
                    else:  # operation == '+='
                        slab_onto_ptr[index] += factor*slab_from_ptr[index]
        else:
            # At least one of deconvolution or the non-trivial part of
            # interlacing is to be performed.
            # For this, we need proper iteration over ki, kj, kk.
            for index, ki, kj, kk, factor, θ in fourier_loop(
                gridsize_onto, deconv_order=deconv_order, interlace_flag=interlace_flag,
            ):
                # Extract real and imag part of this
                # Fourier mode of slab_from.
                re = slab_from_ptr[index    ]
                im = slab_from_ptr[index + 1]
                # Rotate the complex phase due to interlacing
                with unswitch:
                    if 𝔹[interlace_flag == 2]:
                        cosθ = cos(θ)
                        sinθ = sin(θ)
                        re, im = (
                            re*cosθ - im*sinθ,
                            re*sinθ + im*cosθ,
                        )
                # Apply factor from deconvolution and interlacing
                re *= factor
                im *= factor
                # Write to slab_onto
                with unswitch:
                    if 𝔹[operation == '=']:
                        slab_onto_ptr[index    ] = re
                        slab_onto_ptr[index + 1] = im
                    else:  # operation == '+='
                        slab_onto_ptr[index    ] += re
                        slab_onto_ptr[index + 1] += im
        # Easy case of equal grid sizes complete
        return slab_onto
    # The two grid sizes are not equal, and so the grid is to be
    # scaled up or down. Assign some variables according to
    # which of the two scenarios we are in.
    upscaling = (gridsize_from < gridsize_onto)
    if upscaling:
        slab_small, slab_large = slab_from, slab_onto
    else:  # downscaling
        slab_small, slab_large = slab_onto, slab_from
    slab_small_size_j, slab_small_size_i, slab_small_size_k = asarray(slab_small).shape
    slab_large_size_j, slab_large_size_i, slab_large_size_k = asarray(slab_large).shape
    gridsize_small, gridsize_large = slab_small_size_i, slab_large_size_i
    nyquist_small, nyquist_large = gridsize_small//2, gridsize_large//2
    # As slab_from and slab_onto have different grid sizes, a given
    # Fourier mode situated within slab_from on process p may be
    # situated outside slab_onto on process p, but instead within
    # slab_onto on some other process q. Generally then, different grid
    # sizes calls for interprocess communication. To do this
    # communication, we subdivide both slab_from and slab_onto into
    # subslabs, i.e. thinner (in the y direction only) slabs which
    # together constitute a full slab. For this subslab decomposition to
    # be of any use, we require that the Fourier modes within a given
    # subslab of slab_small on p is fully contained within just a single
    # subslab of slab_large on q (which may or may not equal p), and in
    # fact that these two corresponding subslabs span the same kj. Some
    # freedom remains regarding the subslab decomposition, but the
    # optimal choice is that which minimizes the number of subslabs.
    # Find this subslab decomposition now (does not mutate any data).
    subslabs_dict_small, subslabs_dict_large = get_subslabs(gridsize_small, gridsize_large)
    if upscaling:
        subslabs_dict_from, subslabs_dict_onto = subslabs_dict_small, subslabs_dict_large
    else:  # downscaling
        subslabs_dict_from, subslabs_dict_onto = subslabs_dict_large, subslabs_dict_small
    # While the j range is set by the subslab in question, they all span
    # the entire slab in the i and k dimension, but we are only
    # interested in the region corresponding to slab_small, as Fourier
    # modes beyond it cannot be copied between the two slabs. Set i and
    # k ranges. Note that each subslab consists of two i ranges, i.e.
    # two half subslabs. This is needed as these two halves are not
    # contiguous (due to the skipped larger Fourier modes in the large
    # slab and the skipped Nyquist modes in the small slab). Also note
    # that we implicitly have k_bgn = 0.
    i_ranges_small = [(0, nyquist_small), (gridsize_small - nyquist_small + 1, gridsize_small)]
    i_ranges_large = [(0, nyquist_small), (gridsize_large - nyquist_small + 1, gridsize_large)]
    if upscaling:
        i_ranges_from, i_ranges_onto = i_ranges_small, i_ranges_large
    else:  # downscaling
        i_ranges_from, i_ranges_onto = i_ranges_large, i_ranges_small
    k_end = slab_small_size_k - 2
    # Loop over processes with which to communicate subslabs
    for (
        (rank_onto, subslabs_ranges_from),
        (rank_from, subslabs_ranges_onto),
    ) in zip(
        subslabs_dict_from.items(),
        subslabs_dict_onto.items(),
    ):
        # Flag specifying whether this is a purely local subslab,
        # meaning that it resides entirely on the local process in
        # both slab_from and slab_onto.
        local = (rank_onto == rank)
        # Loop over subslabs
        for (
            (j_from_bgn, j_from_end),
            (j_onto_bgn, j_onto_end),
        ) in zip(
            subslabs_ranges_from,
            subslabs_ranges_onto,
        ):
            # Flags specifying whether we should
            # send/receive the subslab.
            send = (not local and rank_onto >= 0 and j_from_bgn != -1)
            recv = (not local and rank_from >= 0 and j_onto_bgn != -1)
            # Loop over half subslabs (positive and negative ki)
            for (
                (i_from_bgn, i_from_end),
                (i_onto_bgn, i_onto_end),
            ) in zip(
                i_ranges_from,
                i_ranges_onto,
            ):
                # Send half subslab
                if send:
                    request = smart_mpi(
                        slab_from[j_from_bgn:j_from_end, i_from_bgn:i_from_end, :k_end],
                        dest=rank_onto,
                        mpifun='Isend',
                    )
                # Receive half subslab
                if recv:
                    half_subslab_recv = get_buffer(
                        (j_onto_end - j_onto_bgn, i_onto_end - i_onto_bgn, k_end),
                        'subslab',
                    )
                    smart_mpi(
                        half_subslab_recv,
                        source=rank_from,
                        mpifun='Recv',
                    )
                # Wait for the non-blocking send to complete
                if send:
                    request.wait()
                # Done with subslab communication. Only carry on if the
                # half subslab should be copied into the local slab_onto
                # (i.e. if we are not at a local subslab which should
                # only be send).
                if not (local or recv):
                    continue
                # To copy the (local or received) Fourier modes over to
                # slab_onto, we now need to loop over this slab, or
                # rather some (half) subslab of it. However, we only
                # want to set the modes within the small grid (even in
                # the case of slab_onto = slab_large, as modes beyond it
                # cannot be set from slab_from = slab_small), and so we
                # need to loop only over the j range of the
                # small subslab. Get this j range from
                # j_onto_bgn and j_onto_end.
                if upscaling:
                    # When upscaling we have slab_onto = slab_large
                    # and so the known j range of the onto slab is for
                    # the subslab of the large slab, not the small.
                    # Convert the bgn index.
                    j_global_large = ℤ[slab_large_size_j*rank] + j_onto_bgn
                    kj = j_global_large - gridsize_large*(j_global_large >= nyquist_large)
                    j_small_bgn = kj + gridsize_small*(kj < 0) - ℤ[slab_small_size_j*rank]
                    # Convert the end index, remembering to temporarily
                    # treat it as inclusive rather than exclusive.
                    j_global_large = ℤ[slab_large_size_j*rank] + j_onto_end - 1
                    kj = j_global_large - gridsize_large*(j_global_large >= nyquist_large)
                    j_small_end = kj + gridsize_small*(kj < 0) - ℤ[slab_small_size_j*rank] + 1
                else:  # downscaling
                    # When downscaling we have slab_onto = slab_small
                    # and so the known j range of the onto slab is the
                    # same as the j range of the small slab.
                    j_small_bgn, j_small_end = j_onto_bgn, j_onto_end
                # We further need the i range of the half subslab of the
                # small slab. Here we can always just pick the correct
                # pre-computed such range.
                if upscaling:
                    i_small_bgn, i_small_end = i_from_bgn, i_from_end
                else:  # downscaling
                    i_small_bgn, i_small_end = i_onto_bgn, i_onto_end
                # Pointer to the slab from which
                # the data should be copied.
                if local:
                    slab_read_ptr = slab_from_ptr
                else:
                    slab_read_ptr = cython.address(half_subslab_recv[:, :, :])
                # Loop over the Fourier modes of the half small subslab,
                # evaluating deconvolution and interlacing as if they
                # belong to a grid of size gridsize_from
                # (which they do).
                for index_small, ki, kj, kk, factor, θ in fourier_loop(
                    gridsize_small, gridsize_from,
                    i_small_bgn, i_small_end,
                    j_small_bgn, j_small_end,
                    deconv_order=deconv_order, interlace_flag=interlace_flag,
                ):
                    # Corresponding index into the large subslab
                    index_large = (
                        ℤ[
                            (
                                ℤ[
                                    (
                                        kj + gridsize_large*(kj < 0) - ℤ[slab_large_size_j*rank]  # j_large
                                    )*slab_large_size_i
                                ]
                                + ki + gridsize_large*(ki < 0)  # i_large
                            )*slab_large_size_k
                        ]
                        + 2*kk  # k_large
                    )
                    # Corresponding index into the received half subslab
                    index_recv = (
                        ℤ[
                            (
                                ℤ[
                                    (
                                        (kj + gridsize_small*(kj < 0)) - ℤ[slab_small_size_j*rank + j_small_bgn]  # j_recv
                                    )*ℤ[i_small_end - i_small_bgn]
                                ]
                                + ki + gridsize_small*(ki < 0) - i_small_bgn  # i_recv
                            )*k_end
                        ]
                        + 2*kk  # k_recv
                    )
                    # Set indices into the read and write slab
                    with unswitch(5):
                        if 𝔹[local and upscaling]:
                            # Local upscaling
                            index_from = index_small
                            index_onto = index_large
                        elif 𝔹[local and not upscaling]:
                            # Local downscaling
                            index_from = index_large
                            index_onto = index_small
                        elif upscaling:
                            # Non-local upscaling
                            index_from = index_recv
                            index_onto = index_large
                        else:
                            # Non-local downscaling
                            index_from = index_recv
                            index_onto = index_small
                    # Extract real and imag part of this Fourier mode
                    re = slab_read_ptr[index_from    ]
                    im = slab_read_ptr[index_from + 1]
                    # The total complex phase shift due to both change
                    # of grid size and interlacing.
                    θ_total = ℝ[π*(1/gridsize_onto - 1/gridsize_from)]*ℤ[ℤ[ki + kj] + kk]
                    with unswitch(5):
                        if 𝔹[interlace_flag == 2]:
                            θ_total += θ
                    # Apply factor and phase shift from deconvolution,
                    # interlacing and change of grid size.
                    cosθ = cos(θ_total)
                    sinθ = sin(θ_total)
                    re, im = (
                        factor*(re*cosθ - im*sinθ),
                        factor*(re*sinθ + im*cosθ),
                    )
                    # Write to slab_onto
                    with unswitch(5):
                        if 𝔹[operation == '=']:
                            slab_onto_ptr[index_onto    ] = re
                            slab_onto_ptr[index_onto + 1] = im
                        else:  # operation == '+=':
                            slab_onto_ptr[index_onto    ] += re
                            slab_onto_ptr[index_onto + 1] += im
    return slab_onto

# Function for dividing up a pair of different sized slabs into
# overlapping subslabs. The resulting subslab layout is returned,
# while no slab is mutated.
def get_subslabs(gridsize_small, gridsize_large):
    """Given two grid sizes, one smaller than the other, this function
    returns two corresponding dicts mapping process ranks to lists of
    overlapping subslab ranges between the local process and the process
    rank in question.
    An example will illustrate the format. Name the two returned dicts
    subslabs_dict_j_small and subslabs_dict_j_large and say we see
    things from the perspective of process 1 while running with 4
    processes and having gridsize_small = 12, gridsize_large = 20,
    the result would then be
        subslabs_dict_j_small = {1: [(2, 3)], 0: [(0, 2)]}  # process 1
        subslabs_dict_j_large = {1: [(0, 1)]}               # process 1
    This means that given these two grid sizes, the local (to process 1)
    small slab is divided into two subslabs slab_small[j=0:2, i=:, k=:]
    and slab_small[j=2:3, i=:, k=:], whereas only a single local subslab
    slab_large[j=0:1, i=:, k=:] results from the local large slab. Note
    that the union of all subslabs given may not result in the complete
    slab, as some subslabs may be irrelevant (they contain no overlap
    between the small and large slabs, or contain only Nyquist points
    which are disregarded). The small local subslab
    slab_small[j=0:2, i=:, k=:] overlaps with a large subslab on process
    0 (note that by construction two small or two large subslabs may
    never overlap). The details of this non-local overlapping large
    subslab is not known to process 1, but is contained (among other
    things) in the corresponding result
        subslabs_dict_j_large = {0: [(0, 3)], 1: [(3, 5)]}  # process 0
    on process 0. We see that the other small subslab
    slab_small[j=2:3, i=:, k=:] local to process 1 overlaps with a large
    subslab on process 1 itself. Indeed, looking at
    subslabs_dict_j_large on process 1 we see this large subslab.
    """
    # Look up previous result in cache
    key = (gridsize_small, gridsize_large)
    subslabs_j_both = subslabs_cache.get(key)
    if subslabs_j_both is not None:
        return subslabs_j_both
    # Sanity check on grid sizes
    if gridsize_small > gridsize_large:
        masterwarn(
            f'get_subslabs() called with gridsize_small = {gridsize_small} '
            f'> gridsize_large = {gridsize_large}. These arguments have '
            f'been switched around.'
        )
        gridsize_smal, gridsize_large = gridsize_large, gridsize_small
        return get_subslabs(gridsize_small, gridsize_large)
    for varname, gridsize in {
        'gridsize_small': gridsize_small,
        'gridsize_large': gridsize_large,
    }.items():
        if gridsize%2:
            abort(
                f'get_subslabs() got {varname} = {gridsize}, but this must be even'
            )
        if gridsize%nprocs:
            abort(
                f'get_subslabs() got {varname} = {gridsize} '
                f'which is not divisible by {nprocs} processes'
            )
    # Collect all local kj
    def get_kj_sets(gridsize):
        nyquist = gridsize//2
        slab_size_j = gridsize//nprocs
        kj_sets = []
        for other_rank in range(nprocs):
            kj_set = set()
            for j in range(slab_size_j):
                j_global = slab_size_j*other_rank + j
                kj = j_global - gridsize*(j_global >= nyquist)
                # Ignore Nyquist planes
                if kj == -nyquist:
                    continue
                kj_set.add(kj)
            kj_sets.append(kj_set)
        return kj_sets
    kj_sets_small = get_kj_sets(gridsize_small)
    kj_sets_large = get_kj_sets(gridsize_large)
    # Find groups of consecutive local kj, i.e. subslabs
    def construct_subslabs_kj_dict(kj_set, kj_sets, sign):
        subslabs_dict_kj = {}
        for ℓ in range(nprocs):
            other_rank = (rank + sign*ℓ)%nprocs  # determines rank communication order
            other_kj_set = kj_sets[other_rank]
            overlap = sorted(kj_set & other_kj_set)
            if not overlap:
                continue
            subslabs_kj_ranges = []
            subslabs_dict_kj[other_rank] = subslabs_kj_ranges
            for _, groups in itertools.groupby(enumerate(overlap), lambda el: el[1] - el[0]):
                subrange = [group[1] for group in groups]
                kj_bgn, kj_end = subrange[0], subrange[-1]
                # A subslab may not contain both
                # negative and non-negative kj.
                kj_bgn, kj_end = np.min((kj_bgn, kj_end)), np.max((kj_bgn, kj_end))
                if kj_bgn < 0 and kj_end >= 0:
                    subslabs_kj_ranges.append((kj_bgn, -1))
                    subslabs_kj_ranges.append((0, kj_end))
                else:
                    subslabs_kj_ranges.append((kj_bgn, kj_end))
        return subslabs_dict_kj
    subslabs_dict_kj_small = construct_subslabs_kj_dict(kj_sets_small[rank], kj_sets_large, +1)
    subslabs_dict_kj_large = construct_subslabs_kj_dict(kj_sets_large[rank], kj_sets_small, -1)
    # Convert from kj to j
    def convert_subslabs_kj_to_j(subslabs_kj_dict, gridsize):
        slab_size_j = gridsize//nprocs
        def kj_to_j(kj):
            return kj + gridsize*(kj < 0) - slab_size_j*rank
        subslabs_j_dict = {
            other_rank: [
                (kj_to_j(kj_bgn), kj_to_j(kj_end) + 1)  # use exclusive end point
                for kj_bgn, kj_end in subslabs_kj_ranges
            ]
            for other_rank, subslabs_kj_ranges in subslabs_kj_dict.items()
        }
        return subslabs_j_dict
    subslabs_dict_j_small = convert_subslabs_kj_to_j(subslabs_dict_kj_small, gridsize_small)
    subslabs_dict_j_large = convert_subslabs_kj_to_j(subslabs_dict_kj_large, gridsize_large)
    # Pad the small and large subslab dicts such that
    # they contain the same number of items, with pairs of j ranges
    # which also contain the same number of elements.
    def pad_dict(subslabs_j, length):
        fake_rank = -1
        while len(subslabs_j) < length:
            subslabs_j[fake_rank] = []
            fake_rank -= 1
    pad_dict(subslabs_dict_j_small, len(subslabs_dict_j_large))
    pad_dict(subslabs_dict_j_large, len(subslabs_dict_j_small))
    def pad_ranges(ranges, length):
        while len(ranges) < length:
            ranges.append((-1, -1))
    for subslabs_j_ranges_small, subslabs_j_ranges_large in zip(
        subslabs_dict_j_small.values(), subslabs_dict_j_large.values()
    ):
        length = np.max([len(subslabs_j_ranges_small), len(subslabs_j_ranges_large)])
        pad_ranges(subslabs_j_ranges_small, length)
        pad_ranges(subslabs_j_ranges_large, length)
    # Cache results and return
    subslabs_dict_j_both = (subslabs_dict_j_small, subslabs_dict_j_large)
    subslabs_cache[key] = subslabs_dict_j_both
    return subslabs_dict_j_both
# Cache used by the get_subslabs() function
cython.declare(subslabs_cache=dict)
subslabs_cache = {}

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
    factor='double',
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
    shift=0, factor=1, do_ghost_communication=True):
    """The given quantity of the particle component will be added to
    current content of the local grid with global grid size given by
    gridsize. For info about the quantity argument, see the
    interpolate_upstream() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    By supplying shift != 0, the particle positions will be shifted
    by -shift (grid units) before interpolated to the grid, or
    equivalently the whole grid is shifted +shift prior to the
    interpolation. This is typically used for interlacing two grids
    relatively shifted by 0.5.
    Setting the factor != 1 scales the interpolated quantity.
    The supplied grid should contain ghost layers, as the interpolation
    will populate these. To communicate and add the resulting values in
    the ghost cells to their physical cells, set do_ghost_communication
    to True. Note that even with do_ghost_communication set to True, the
    ghost cells will not end up with copies of the boundary values.
    """
    if not (1 <= order <= 4):
        abort(
            f'interpolate_particles() called with order = {order} '
            f'∉ {{1 (NGP), 2 (CIC), 3 (TSC), 4 (PCS)}}'
        )
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
    contribution_factor = factor*(gridsize/boxsize)**3
    contribution *= contribution_factor
    # Offsets and scalings needed for the interpolation
    cellsize = boxsize/gridsize
    offset_x = domain_start_x - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered + shift)*cellsize]
    offset_y = domain_start_y - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered + shift)*cellsize]
    offset_z = domain_start_z - ℝ[(1 + machine_ϵ)*(nghosts - 0.5*cell_centered + shift)*cellsize]
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
            else:  # order == 4  # PCS interpolation
                index_i = set_weights_PCS(x, weights_x)
                index_j = set_weights_PCS(y, weights_y)
                index_k = set_weights_PCS(z, weights_z)
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

# Function for adding together a certain quantity
# from several fluid components.
@cython.header(
    # Arguments
    components=list,
    grid='double[:, :, ::1]',
    quantity=str,
    ᔑdt=dict,
    factor='double',
    # Locals
    component='Component',
)
def combine_fluids(components, grid, quantity, ᔑdt, factor=1):
    """The given quantity of the components (all of which are assumed to
    be fluid components) will be added together into the grid, with the
    current contents of the grid overwritten.
    """
    if not components:
        return grid
    add_fluid_to_grid(components[0], grid, quantity, ᔑdt, factor, operation='=')
    for component in components[1:]:
        add_fluid_to_grid(component, grid, quantity, ᔑdt, factor, operation='+=')
    return grid

# Function for adding a certain quantity from a fluid
# component onto a supplied domain grid.
@cython.pheader(
    # Arguments
    component='Component',
    grid='double[:, :, ::1]',
    quantity=str,
    ᔑdt=dict,
    factor='double',
    operation=str,
    # Locals
    a='double',
    dim='int',
    fluidscalar='FluidScalar',
    grid_fluid='double[:, :, ::1]',
    grid_fluid_ptr='double*',
    grid_ptr='double*',
    gridshape=tuple,
    gridshape_fluid=tuple,
    index='Py_ssize_t',
    w_eff='double',
    returns='double[:, :, ::1]',
)
def add_fluid_to_grid(component, grid, quantity, ᔑdt, factor=1, operation='+='):
    """The given quantity of the fluid component will be added to
    current content of the local grid with global grid size given by
    gridsize. For info about the quantity argument, see the
    interpolate_upstream() function.
    Time dependent factors in the quantity are evaluated at the current
    time as defined by the universals struct. If ᔑdt is passed as a
    dict containing time step integrals, these factors will be
    integrated over the time step.
    Setting the factor != 1 scales the interpolated quantity.
    The supplied grid should contain ghost layers. If these are properly
    populated when this function is called, they will remain so after
    this function has finished.
    """
    # The component must have identical grid size to the passed grid
    gridshape = asarray(grid).shape
    gridshape_fluid = get_gridshape_local(component.gridsize)
    if gridshape != gridshape_fluid:
        abort(
            f'add_fluid_to_grid() got component with local grid shape {gridshape_fluid} '
            f'(global grid size {component.gridsize}) and non-matching grid '
            f'of local shape {gridshape}.'
        )
    # Always use the current time
    a = universals.a
    w_eff = component.w_eff(a=a)
    # Get the fluid grid and determine the contribution factor of each
    # fluid cell, based on the quantity.
    if quantity == 'ρ':
        fluidscalar = component.ϱ
        if ᔑdt:
            factor *= ᔑdt['a**(-3*(1+w_eff))', component.name]/ᔑdt['1']
        else:
            factor *= a**(-3*(1 + w_eff))
    elif quantity == 'a²ρ':
        fluidscalar = component.ϱ
        if ᔑdt:
            factor *= ᔑdt['a**(-3*w_eff-1)', component.name]/ᔑdt['1']
        else:
            factor *= a**(-3*w_eff - 1)
    elif quantity == 'ϱ':
        fluidscalar = component.ϱ
    elif quantity in {'Jx', 'Jy', 'Jz'}:
        dim = 'xyz'.index(quantity[1])
        fluidscalar = component.J[dim]
    else:
        abort(
            f'add_fluid_to_grid() called with '
            f'quantity = "{quantity}" ∉ {{"ρ", "a²ρ", "ϱ", "Jx", "Jy", "Jz"}}'
        )
    grid_fluid = fluidscalar.grid_mv
    # Copy the values of the fluid grid onto the buffer grid
    grid_fluid_ptr = cython.address(grid_fluid[:, :, :])
    grid_ptr = cython.address(grid[:, :, :])
    for index in range(grid.shape[0]*grid.shape[1]*grid.shape[2]):
        with unswitch:
            if operation == '=':
                with unswitch:
                    if factor == 1:
                        grid_ptr[index] = grid_fluid_ptr[index]
                    else:
                        grid_ptr[index] = grid_fluid_ptr[index]*factor
            else:  # operation == '+=''
                with unswitch:
                    if factor == 1:
                        grid_ptr[index] += grid_fluid_ptr[index]
                    else:
                        grid_ptr[index] += grid_fluid_ptr[index]*factor
    return grid

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

# Function for transferring data from slabs to domain grids
@cython.pheader(
    # Arguments
    slab='double[:, :, ::1]',
    grid_or_buffer_name=object,  # double[:, :, ::1], int or str
    do_ghost_communication='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
    grid='double[:, :, ::1]',
    grid_noghosts='double[:, :, :]',
    gridsize='Py_ssize_t',
    key=str,
    request=object,  # mpi4py.MPI.Request
    shape=tuple,
    slab_sendrecv_j_end='int[::1]',
    slab_sendrecv_j_start='int[::1]',
    slab_sendrecv_k_end='int[::1]',
    slab_sendrecv_k_start='int[::1]',
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def domain_decompose(slab, grid_or_buffer_name=0, do_ghost_communication=True):
    if slab is None:
        return None
    if slab.shape[0] > slab.shape[1]:
        masterwarn(
            'domain_decompose() was called with a slab that appears to be transposed, '
            'i.e. in Fourier space.'
        )
    # Determine the correct shape of the domain grid corresponding to
    # the passed slab.
    gridsize = slab.shape[1]
    shape = get_gridshape_local(gridsize)
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
        # Wait for the non-blocking send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    # Populate ghost layers
    if do_ghost_communication:
        communicate_ghosts(grid, '=')
    return grid

# Function for transferring data from domain grids to slabs
@cython.pheader(
    # Arguments
    grid='double[:, :, ::1]',
    slab_or_buffer_name=object,  # double[:, :, ::1], int or str
    prepare_fft='bint',
    # Locals
    N_domain2slabs_communications='Py_ssize_t',
    buffer_name=object,  # int or str
    domain_sendrecv_i_end='int[::1]',
    domain_sendrecv_i_start='int[::1]',
    domain2slabs_recvsend_ranks='int[::1]',
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
    slabs2domain_sendrecv_ranks='int[::1]',
    ℓ='Py_ssize_t',
    returns='double[:, :, ::1]',
)
def slab_decompose(grid, slab_or_buffer_name='slab_global', prepare_fft=False):
    """This function communicates a global domain decomposed grid into
    a global slab decomposed grid. If an existing slab grid should be
    used it can be passed as the second argument.
    Alternatively, if a slab grid should be fetched from elsewhere,
    its name should be specified as the second argument.

    If FFT's are to be carried out on a slab fetched by name,
    you must specify prepare_fft=True, in which case the slab will be
    created via FFTW.

    Note that ghost points will not be read by this function,
    and so the passed domain grid need not have these set properly.
    """
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
    # NumPy, not directly on the memory view. See
    #   https://github.com/cython/cython/issues/2941
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
        # Wait for the non-blocking send to be complete before
        # continuing. Otherwise, data in the send buffer - which is
        # still in use by the non-blocking send - might get overwritten
        # by the next (non-blocking) send.
        request.wait()
    return slab

# Iterator implementing looping over Fourier space slabs.
# The yielded values are the linear index into the slab, the physical
# ki, kj, kk (in grid units), the combined factor due to deconvolution
# and interlacing, as well as the angle by which to rotate a shifted
# slab for interlacing.
# The iteration is determined from the passed gridsize. By default the
# same grid size is used in the computations of deconvolution and
# interlacing, though this may be specified separately
# through gridsize_corrections.
# If sparse is True, only unique points in the z DC plane will
# be visited.
# If skip_origin is True, the ki = kj = kk = 0 point will be excluded
# from the iteration.
# By default, all points except those on Nyquist planes are included in
# the iteration. If you only want points for which
#   ki**2 + kj**2 + kk**2 < k2_max,
# specify this k2_max (in grid units). For the largest sphere possible
# (i.e. excluding "corner modes"), still excluding points on Nyquist
# planes, set k2_max = (nyquist - 1)**2 = (gridsize//2 - 1)**2.
# If the returned factor is to be used for deconvolution, specify the
# deconvolution (interpolation) order as deconv_order. Similarly, if the
# factor and θ is to be used for interlacing, specify the interlacing
# stage as interlace_flag (1 for the non-shifted grid and 2 for the
# shifted grid). Both deconv_order and interlace_flag may be
# specified simultaneously.
@cython.iterator
def fourier_loop(
    gridsize, gridsize_corrections=-1,
    i_bgn=0, i_end=None,
    j_bgn=0, j_end=None,
    *,
    sparse=False, skip_origin=False, k2_max=-1,
    deconv_order=0, interlace_flag=False,
):
    # Cython declarations for variables used for the iteration,
    # including all arguments and variables to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        # Arguments
        gridsize='Py_ssize_t',
        gridsize_corrections='Py_ssize_t',
        i_bgn='Py_ssize_t',
        i_end=object,
        j_bgn='Py_ssize_t',
        j_end=object,
        sparse='bint',
        skip_origin='bint',
        k2_max='Py_ssize_t',
        deconv_order='int',
        interlace_flag='int',
        # Locals
        _deconv_i='double',
        _deconv_ij='double',
        _deconv_j='double',
        _deconv_k='double',
        _i='Py_ssize_t',
        _i_chunk='int',
        _i_chunk_bgn='Py_ssize_t',
        _i_chunk_end='Py_ssize_t',
        _i_end='Py_ssize_t',
        _index_ij='Py_ssize_t',
        _index_j='Py_ssize_t',
        _j='Py_ssize_t',
        _j_chunk='int',
        _j_chunk_bgn='Py_ssize_t',
        _j_chunk_end='Py_ssize_t',
        _j_end='Py_ssize_t',
        _j_global='Py_ssize_t',
        _j_global_bgn='Py_ssize_t',
        _j_global_chunk_bgn='Py_ssize_t',
        _j_global_chunk_end='Py_ssize_t',
        _j_global_end='Py_ssize_t',
        _kk_bgn='Py_ssize_t',
        _nyquist='Py_ssize_t',
        _offset_j='Py_ssize_t',
        _slab_size_i='Py_ssize_t',
        _slab_size_j='Py_ssize_t',
        _slab_size_k='Py_ssize_t',
        # Yielded
        index='Py_ssize_t',
        ki='Py_ssize_t',
        kj='Py_ssize_t',
        kk='Py_ssize_t',
        θ='double',
    )
    # Set up slab shape
    _nyquist = gridsize//2
    _slab_size_j = gridsize//nprocs
    _slab_size_i = gridsize
    _slab_size_k = gridsize + 2
    if gridsize_corrections == -1:
        gridsize_corrections = gridsize
    # Set default end indices if not given. After this,
    # _j_end and _i_end should be used instead of j_end and i_end.
    if j_end is None:
        # Default to entire local length along this dimension
        _j_end = _slab_size_j
    else:
        _j_end = j_end
    if i_end is None:
        # Default to entire length along this dimension
        _i_end = _slab_size_i
    else:
        _i_end = i_end
    # The slabs are distributed along the j-dimension,
    # introducing a global offset individual to each process.
    _offset_j = _slab_size_j*rank
    _j_global_bgn = _offset_j +  j_bgn
    _j_global_end = _offset_j + _j_end
    # Begin iterating over slab. As the first and second dimensions
    # are transposed due to the FFT, the j-dimension is first.
    # We loop through the j-dimension in two chunks, skipping the
    # Nyquist point (kj = -_nyquist ⇒ _j_global = _nyquist) in between.
    for _j_chunk in range(2):
        if _j_chunk == 0:
            _j_chunk_bgn = j_bgn
            _j_global_chunk_end = pairmin(_j_global_end, _nyquist)
            _j_chunk_end = _j_global_chunk_end - _offset_j
        else:  # _j_chunk == 1
            _j_global_chunk_bgn = pairmax(_j_global_bgn, ℤ[_nyquist + 1])
            _j_chunk_bgn = _j_global_chunk_bgn - _offset_j
            _j_chunk_end = _j_end
        # We similarly loop through the i-dimension in two chunks,
        # skipping the Nyquist point (ki = -_nyquist ⇒ _i = _nyquist)
        # in between.
        for _i_chunk in range(2):
            if _i_chunk == 0:
                _i_chunk_bgn = i_bgn
                _i_chunk_end = pairmin(_i_end, _nyquist)
            else:  # _i_chunk == 1
                _i_chunk_bgn = pairmax(i_bgn, ℤ[_nyquist + 1])
                _i_chunk_end = _i_end
            # Loop over the j chunk
            _index_j = ℤ[_j_chunk_bgn*_slab_size_i - ℤ[_slab_size_i + 1]]
            for _j in range(_j_chunk_bgn, _j_chunk_end):
                _index_j += _slab_size_i
                # The j-component of the wave vector (grid units)
                _j_global = _offset_j + _j
                kj = _j_global - ℤ[gridsize*_j_chunk]
                # Bail out if beyond maximum frequency
                with unswitch(3):
                    if k2_max != -1:
                        if ℤ[kj**2] > k2_max:
                            continue
                # The j-component of the 1D NGP deconvolution factor.
                # This is given by
                #   deconv_1D_NGP = 1/sinc(ky*Δx/2)
                #                 = 1/sinc(kj*π/gridsize)
                # with
                #   sinc(x) ≡ sin(x)/x
                # and
                #   sinc(0) = 1.
                # Here ky = 2*π/boxsize*kj is the y component of the
                # physical wave vector and Δx = boxsize/gridsize is the
                # grid spacing. The 3D deconvolution factor is then
                # given as a product over the three 1D deconvolution
                # factors, and the final higher-order deconvolution
                # factor is obtained through exponentiation with the
                # power given by the deconvolution order (**1 → NGP,
                # **2 → CIC, **3 → TSC, **4 → PCS).
                _deconv_j = kj*ℝ[π/gridsize_corrections] + machine_ϵ
                _deconv_j /= sin(_deconv_j)
                # Loop over the i chunk
                _index_ij = (_i_chunk_bgn + _index_j)*_slab_size_k
                for _i in range(_i_chunk_bgn, _i_chunk_end):
                    _index_ij += _slab_size_k
                    # The i-component of the wave vector (grid units)
                    ki = _i - ℤ[gridsize*_i_chunk]
                    # Bail out if beyond maximum frequency
                    with unswitch(4):
                        if k2_max != -1:
                            if ℤ[ℤ[kj**2] + ki**2] > k2_max:
                                continue
                    # The product of the i- and the j-components
                    # of the 1D NGP deconvolution factor.
                    _deconv_i = ki*ℝ[π/gridsize_corrections] + machine_ϵ
                    _deconv_i /= sin(_deconv_i)
                    _deconv_ij = _deconv_i*_deconv_j
                    # The origin is the first element encountered on the
                    # master process. If the partial index _index_ij has
                    # a value of 0, then this is the first i iteration,
                    # meaning that the origin (with kk = 0) is included
                    # in the upcoming kk iteration. If the origin should
                    # be skipped, we do so by excluding kk = 0 from this
                    # next iteration.
                    _kk_bgn = 0
                    with unswitch(4):
                        if skip_origin:
                            with unswitch(4):
                                if master:
                                    _kk_bgn = int(_index_ij == 0)
                    # The z DC plane consists of complex conjugate pairs
                    # of points. When looping sparsely we only want to
                    # hit one point from each pair. To do this, we
                    # choose to skip points with positive ki and also
                    # points with positive kj and ki == 0. Note that we
                    # could switch the roles of ki and kj here, but this
                    # produces a work load imbalance as skipping all
                    # positive kj points means that half of the
                    # processes do not participate.
                    with unswitch(4):
                        if sparse:
                            # Remember to not disregard a possible
                            # value of _kk_bgn = 1 from the above
                            # skip of the origin.
                            _kk_bgn |= ki > 0 or (𝔹[kj > 0] and ki == 0)
                    # Only the non-negative part of the k-dimension
                    # exists. Loop through this half, one complex number
                    # at a time, looping directly over kk instead of
                    # k == 2*kk. Here the Nyquist point kk = _nyquist is
                    # the last element along this dimension, and so we
                    # skip it by simply not including it in the range.
                    for kk in range(_kk_bgn, _nyquist):
                        # Bail out if beyond maximum frequency
                        with unswitch(5):
                            if k2_max != -1:
                                if ℤ[ℤ[ℤ[kj**2] + ki**2] + kk**2] > k2_max:
                                    break
                        # Index into (real part of) complex Fourier mode
                        index = _index_ij + 2*kk
                        # Combined factor for deconvolution
                        # and interlacing.
                        factor = 1
                        # The full deconvolution factor
                        with unswitch(5):
                            if deconv_order:
                                # The 3D NGP deconvolution factor
                                _deconv_k = kk*ℝ[π/gridsize_corrections] + machine_ϵ
                                _deconv_k /= sin(_deconv_k)
                                factor = _deconv_ij*_deconv_k
                                # The full deconvolution factor
                                factor **= deconv_order
                        # Include factor from interlacing
                        # and compute interlacing angle.
                        θ = 0
                        with unswitch(5):
                            if 𝔹[interlace_flag > 0]:
                                factor *= 0.5
                                # The angle θ by which to rotate the
                                # complex phase of the shifted grid at
                                # this k⃗, before ("harmonically")
                                # averaging together with the
                                # non-shifted grid.
                                # This angle is given by
                                #   θ = (kx + ky + kz)*Δx/2
                                #     = (ki + kj + kk)*π/gridsize.
                                # The above is correct for a positive
                                # shift, as chosen for a cell-centred
                                # strategy. For cell-vertex, θ should be
                                # negated.
                                θ = ℤ[ℤ[ki + kj] + kk]*ℝ[
                                    π/gridsize_corrections
                                    *(2*cell_centered - 1)
                                ]
                        # Yield the needed variables
                        yield index, ki, kj, kk, factor, θ

# Function performing in-place deconvolution and/or interlacing of
# Fourier slabs.
@cython.pheader(
    # Arguments
    slab='double[:, :, ::1]',
    deconv_order='int',
    interlace_flag='int',
    # Locals
    cosθ='double',
    factor='double',
    gridsize='Py_ssize_t',
    im='double',
    index='Py_ssize_t',
    ki='Py_ssize_t',
    kj='Py_ssize_t',
    kk='Py_ssize_t',
    re='double',
    sinθ='double',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    θ='double',
    returns='double[:, :, ::1]',
)
def deconvolve_and_interlace(slab, deconv_order=0, interlace_flag=0):
    """Deconvolution will be performed according to deconv_order.
    To do an interlacing you need to call this function twice; once with
    the non-shifted grid and interlace_flag = 1, and once with the
    shifted grid and interlace_flag = 2. With interlace_flag = 1, the
    values in the slabs are simply multiplied by 0.5. With
    interlace_flag = 2, the phases of the (complex) values are
    additionally rotated as defined by harmonic averaging.
    """
    # Bail out if neither deconvolution nor interlacing
    # is to be performed.
    if deconv_order == 0 and interlace_flag == 0:
        return slab
    # Extract slab shape and pointer
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    slab_ptr = cython.address(slab[:, :, :])
    # If no deconvolution and only the first (scaling only) stage of
    # interlacing is to be performed, we can do this directly, without
    # use of the Fourier loop.
    if deconv_order == 0 and interlace_flag == 1:
        for index in range(slab_size_j*slab_size_i*slab_size_k):
            slab_ptr[index] *= 0.5
        return slab
    # Perform the deconvolution and interlacing
    gridsize = slab_size_i
    for index, ki, kj, kk, factor, θ in fourier_loop(
        gridsize, deconv_order=deconv_order, interlace_flag=interlace_flag,
    ):
        # Extract real and imag part of slab_from
        re = slab_ptr[index    ]
        im = slab_ptr[index + 1]
        # Rotate the complex phase due to interlacing
        with unswitch:
            if 𝔹[interlace_flag == 2]:
                cosθ = cos(θ)
                sinθ = sin(θ)
                re, im = (
                    re*cosθ - im*sinθ,
                    re*sinθ + im*cosθ,
                )
        # Apply factor from deconvolution and interlacing
        re *= factor
        im *= factor
        # Store updated values back in slabs
        slab_ptr[index    ] = re
        slab_ptr[index + 1] = im
    return slab

# Function for nullifying sets of modes of Fourier space slabs
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    nullifications=object,  # bint, str or list of str's
    # Locals
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    i='Py_ssize_t',
    j='Py_ssize_t',
    j_bgn='Py_ssize_t',
    j_end='Py_ssize_t',
    j_global='Py_ssize_t',
    k='Py_ssize_t',
    k_min='Py_ssize_t',
    ki='Py_ssize_t',
    ki_bgn='Py_ssize_t',
    ki_end='Py_ssize_t',
    kj='Py_ssize_t',
    kj_bgn='Py_ssize_t',
    kj_end='Py_ssize_t',
    kk='Py_ssize_t',
    kk_bgn='Py_ssize_t',
    kk_end='Py_ssize_t',
    match=object,  # re.Match
    nullification=str,
    nyquist='Py_ssize_t',
    slab_ptr='double*',
    slab_size_i='Py_ssize_t',
    slab_size_j='Py_ssize_t',
    slab_size_k='Py_ssize_t',
    returns='void',
)
def nullify_modes(slab, nullifications):
    """The nullifications argument can be a boolean, a str of
    comma-separated words, or alternatively a list of str's, each being
    a single word. The words specify which types of modes to nullify:
    - False: Do not perform any nullification.
    - True, "all": Nullify the entire slab.
    - "origin": Nullify the origin ki = kj = kk = 0.
    - "Nyquist": Nullify the three Nyquist planes:
        ki = -nyquist, -nyquist ≤ kj < nyquist, 0 ≤ kk ≤ nyquist.
        kj = -nyquist, -nyquist ≤ ki < nyquist, 0 ≤ kk ≤ nyquist.
        kk = +nyquist, -nyquist ≤ ki < nyquist, -nyquist ≤ kj < nyquist.
    - "beyond cube of |k| < num" with num some positive int literal:
        Nullifies everything except points (ki, kj, kk) with at least
        one |kl| < num, leaving an un-nullified cube
        centred at the origin.
    """
    if slab is None:
        return
    slab_ptr = cython.address(slab[:, :, :])
    # Parse nullifications
    if isinstance(nullifications, str):
        nullifications = nullifications.split(',')
    else:
        nullifications = any2list(nullifications)
    nullifications = [
        str(nullification_obj).strip().lower()
        for nullification_obj in nullifications
    ]
    # Get slab dimensions
    slab_size_j, slab_size_i, slab_size_k = asarray(slab).shape
    gridsize = slab_size_i
    nyquist = gridsize//2
    # Perform nullifications
    for nullification in nullifications:
        if nullification == 'false':
            # Do not perform any nullification
            pass
        elif nullification in {'true', 'all'}:
            # Nullify entire slab
            slab[...] = 0
        elif nullification == 'origin':
            # Nullify the origin point ki == kj == kk == 0. This is
            # always located as the first element on the master process.
            if master:
                slab_ptr[0] = 0  # real part
                slab_ptr[1] = 0  # real part
        elif nullification == 'nyquist':
            # Nullify the three Nyquist planes ki = -nyquist,
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
                    index = ℤ[(j*slab_size_i + i)*slab_size_k] + k
                    slab_ptr[index    ] = 0  # real part
                    slab_ptr[index + 1] = 0  # imag part
            kj = -nyquist
            j_global = kj + gridsize*(kj < 0)
            j = j_global - ℤ[slab_size_j*rank]
            if 0 <= j < slab_size_j:
                for i in range(gridsize):
                    for k in range(0, gridsize, 2):  # exclude k = gridsize (kk = nyquist)
                        index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                        slab_ptr[index    ] = 0  # real part
                        slab_ptr[index + 1] = 0  # imag part
            kk = +nyquist
            k = 2*kk
            for j in range(slab_size_j):
                for i in range(gridsize):
                    index = (ℤ[j*slab_size_i] + i)*slab_size_k + k
                    slab_ptr[index    ] = 0  # real part
                    slab_ptr[index + 1] = 0  # imag part
        elif nullification.startswith('beyond cube'):
            # Nullify everything except for a central cube with
            # half-width k_min determined from the nullification str.
            # As only half of the full 3D Fourier space is represented
            # in memory, the region to be nullified can be partitioned
            # into 2⨉2 + 1 = 5 regions;
            #   ki <= -k_min, k_min <= ki,
            #   kj <= -k_min, k_min <= kj,
            #   k_min <= kk.
            # Below these 2⨉2 + 1 nullifications are performed
            # such that overlapping parts of the 5 regions are not
            # nullified more than once.
            match = re.search(r'^beyond cube of \|k\| < (\d+)$', nullification)
            if not match:
                abort(f'nullify_modes(): wrongly formatted nullification "{nullification}"')
            k_min = int(match.group(1))
            for ki_bgn, ki_end in zip((-nyquist, k_min), (-k_min + 1, nyquist)):
                for j in range(slab_size_j):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + gridsize*(ki < 0)
                        for k in range(0, slab_size_k, 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
            ki_bgn, ki_end = -k_min + 1, k_min
            for kj_bgn, kj_end in zip((-nyquist, k_min), (-k_min + 1, nyquist)):
                j_bgn = kj_bgn + ℤ[gridsize*(kj_bgn < 0) - ℤ[slab_size_j*rank]]
                j_end = kj_end + ℤ[gridsize*(kj_bgn < 0) - ℤ[slab_size_j*rank]]
                j_bgn = pairmax(j_bgn, 0)
                j_end = pairmin(j_end, slab_size_j)
                for j in range(j_bgn, j_end):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + gridsize*(ki < 0)
                        for k in range(0, slab_size_k, 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
            ki_bgn, ki_end = -k_min + 1, k_min
            kk_bgn, kk_end = k_min, nyquist + 1
            for kj_bgn, kj_end in zip((-k_min + 1, 0), (0, k_min)):
                j_bgn = kj_bgn + ℤ[gridsize*(kj_bgn < 0) - ℤ[slab_size_j*rank]]
                j_end = kj_end + ℤ[gridsize*(kj_bgn < 0) - ℤ[slab_size_j*rank]]
                j_bgn = pairmax(j_bgn, 0)
                j_end = pairmin(j_end, slab_size_j)
                for j in range(j_bgn, j_end):
                    for ki in range(ki_bgn, ki_end):
                        i = ki + gridsize*(ki < 0)
                        for k in range(ℤ[2*kk_bgn], ℤ[2*kk_end], 2):
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            index = ℤ[(ℤ[j*slab_size_i] + i)*slab_size_k] + k
                            slab_ptr[index    ] = 0  # real part
                            slab_ptr[index + 1] = 0  # imag part
        else:
            abort(f'nullify_modes(): nullification "{nullification}" not understood')

# Function that returns a slab decomposed grid,
# allocated by FFTW.
@cython.pheader(
    # Arguments
    gridsize='Py_ssize_t',
    buffer_name=object,  # int or str
    nullify=object,  # bint, str or list of str's
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
def get_fftw_slab(gridsize, buffer_name='slab_global', nullify=False):
    global fftw_plans_size, fftw_plans_forward, fftw_plans_backward
    # If this slab has already been constructed, fetch it
    slab = slabs.get((gridsize, buffer_name))
    if slab is not None:
        nullify_modes(slab, nullify)
        return slab
    # Checks on the passed gridsize
    if gridsize%nprocs != 0:
        abort(
            f'A grid size of {gridsize} was passed to the get_fftw_slab() function. '
            f'This grid size is not evenly divisible by {nprocs} processes.'
        )
    if gridsize%2 != 0:
        masterwarn(
            f'An odd grid size ({gridsize}) was passed to the get_fftw_slab() function. '
            f'Some operations may not function correctly.'
    )
    shape = (
        (gridsize//nprocs), # Distributed dimension
        gridsize,
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
    nullify_modes(slab, nullify)
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
# Mapping from memory addresses of slabs to indices in
# fftw_plans_forward and fftw_plans_backward.
cython.declare(fftw_plans_mapping=dict)
fftw_plans_mapping = {}
# Dict keeping track of what FFTW wisdom has already been acquired
cython.declare(wisdom_acquired=dict)
wisdom_acquired = {}

# Helper function for the get_fftw_slab() function,
# which construct the absolute path to the wisdom file to use.
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
    - The global FFTW wisdom rigour.
    - The FFTW version.
    - The name of the node "owning" the wisdom in the case of
      fftw_wisdom_share being True. Here a node is said to own the
      wisdom if it hosts the majority of the processes. A more elaborate
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
    )).encode('utf-8')).hexdigest()[:sha_length]
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
    slab='double[:, :, ::1]',
    direction=str,
    apply_forward_normalization='bint',
    # Locals
    fftw_plans_index='Py_ssize_t',
    slab_address='Py_ssize_t',
    returns='void',
)
def fft(slab, direction, apply_forward_normalization=False):
    """Fourier transform the given slab decomposed grid.
    For a forwards transformation from real to Fourier space, supply
    direction='forward'. By default this is an unnormalised transform,
    as defined by FFTW. To do the normalization, divide all elements of
    the slab by gridsize**3, where gridsize is the linear grid size
    of the cubic grid. This will be done if you further set
    apply_forward_normalization to True.
    For a backwards transformation from Fourier to real space, supply
    direction='backward'. Here, no further normalization is needed,
    as defined by FFTW.

    In pure Python, NumPy is used to carry out the Fourier transforms.
    To emulate the effects of FFTW perfectly, a lot of extra steps
    are needed.
    """
    if slab is None:
        return
    if direction not in ('forward', 'backward'):
        abort(
            f'fft() was called with the direction "{direction}", '
            f'which is neither "forward" nor "backward".'
        )
    if not cython.compiled:
        # The pure Python FFT implementation is serial.
        # Every process computes the entire FFT of the temporary
        # variable grid_global_pure_python.
        slab_size_i = slab_size_j = slab.shape[0]
        slab_start_i = slab_size_i*rank
        slab_start_j = slab_size_j*rank
        gridsize = slab.shape[1]
        gridsize_padding = slab.shape[2]
        grid_global_pure_python = empty(
            (gridsize, gridsize, gridsize_padding),
            dtype=C2np['double'],
        )
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
            tmp = empty((gridsize, gridsize, gridsize_padding), dtype=C2np['double'])
            for i in range(gridsize_padding):
                if i%2:
                    tmp[:, :, i] = grid_global_pure_python.imag[:, :, i//2]
                else:
                    tmp[:, :, i] = grid_global_pure_python.real[:, :, i//2]
            grid_global_pure_python = tmp
            # As in FFTW, distribute the slabs along the y-dimension
            # (which is the first dimension now, due to transposing).
            slab[...] = grid_global_pure_python[slab_start_j:(slab_start_j + slab_size_j), :, :]
        else:  # direction == 'backward':
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
            # Undo the auto-scaling provided by NumPy
            grid_global_pure_python *= gridsize**3
            # Add padding on last dimension, as in FFTW
            padding = empty(
                (gridsize, gridsize, gridsize_padding - gridsize),
                dtype=C2np['double'],
            )
            grid_global_pure_python = np.concatenate((grid_global_pure_python, padding), axis=2)
            # As in FFTW, distribute the slabs along the x-dimension
            slab[...] = grid_global_pure_python[slab_start_i:(slab_start_i + slab_size_i), :, :]
    else:  # Compiled mode
        # Look up the index of the FFTW plans for the passed slab.
        slab_address = cast(cython.address(slab[:, :, :]), 'Py_ssize_t')
        fftw_plans_index = fftw_plans_mapping[slab_address]
        # Look up the plan and let FFTW do the Fourier transformation
        if 𝔹[direction == 'forward']:
            fftw_execute(fftw_plans_forward[fftw_plans_index])
        else:  # direction == 'backward':
            fftw_execute(fftw_plans_backward[fftw_plans_index])
    # Apply normalization after forward transform, if specified
    if 𝔹[direction == 'forward'] and apply_forward_normalization:
        fft_normalize(slab)

# Function for normalizing Fourier slabs,
# needed after a forward FFT.
@cython.header(
    # Arguments
    slab='double[:, :, ::1]',
    # Locals
    factor='double',
    gridsize='Py_ssize_t',
    index='Py_ssize_t',
    slab_ptr='double*',
    returns='double[:, :, ::1]',
)
def fft_normalize(slab):
    gridsize = slab.shape[1]
    factor = float(gridsize)**(-3)
    slab_ptr = cython.address(slab[:, :, :])
    for index in range(slab.shape[0]*slab.shape[1]*slab.shape[2]):
        slab_ptr[index] *= factor
    return slab

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
    - Plane inversion through the centre of each Nyquist plane,
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

    From the bulk and Nyquist plane inversion symmetries it follows that
    (see the docstring of get_purely_reals() below) the grid point at
    the origin, grid points at centres of Nyquist planes, at centres of
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
    satisfied if the bulk symmetry and the corresponding ki = -nyquist
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
    and should not be called during actual simulation, and never with
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
        be purely real. This corresponds to the centre or the bulk
        (the origin), the centre of each of the 6 Nyquist faces,
        the centre of each of the 12 Nyquist edges
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
        respect to inversion through their centre (and complex
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
                0: 'centre of bulk (origin)',
                1: 'centre of face',
                2: 'centre of edge',
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
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist),
            lambda ki, kj, kk: (+ki, -kj, -kk),
            'inversion through centre of ki = -nyquist plane',
        )
    )
    # Symmetry of inverting the negative y Nyquist plane
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist),
            lambda ki, kj, kk: (-ki, +kj, -kk),
            'inversion through centre of kj = -nyquist plane',
        )
    )
    # Symmetry of inverting the positive z Nyquist plane
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kk == +nyquist),
            lambda ki, kj, kk: (-ki, -kj, +kk),
            'inversion through centre of kk = +nyquist plane',
        )
    )
    # Symmetry of inverting the negative x, positive z Nyquist edge
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (ki == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (+ki, -kj, +kk),
            'inversion through centre of ki = -nyquist, kk = +nyquist edge',
        )
    )
    # Symmetry of inverting the negative y, positive z Nyquist edge
    # through its centre.
    checks.append(
        generate_check(
            lambda ki, kj, kk: (kj == -nyquist and kk == +nyquist),
            lambda ki, kj, kk: (-ki, +kj, +kk),
            'inversion through centre of kj = -nyquist, kk = +nyquist edge',
        )
    )
    # Reality of special points
    checks.append(check_reality)
    # Check symmetries throughout the small grid
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
    buffer_or_buffer_name argument can be a (contiguous) buffer to store
    the results, or alternatively the name of a buffer to use. In either
    case, existing values will be overwritten.
    Note that a grid cannot be differentiated in-place by passing the
    grid as both the first and fifth argument, as the differentiation
    of each grid point requires information from the original
    (non-differentiated) grid.
    """
    # Sanity checks on input
    if dim not in (0, 1, 2):
        abort(f'diff_domaingrid() called with dim = {dim} ∉ {{0, 1, 2}}')
    if order not in (1, 2, 4, 6, 8):
        abort(
            f'diff_domaingrid() called with order = {order} ∉ {{1, 2, 4, 6, 8}}'
        )
    if direction not in ('forward', 'backward'):
        abort(
            f'diff_domaingrid() called with direction = {direction} '
            f'∉ {{"forward", "backward"}}'
        )
    # If no buffer is supplied, fetch the buffer with the name
    # given by buffer_or_buffer_name.
    if isinstance(buffer_or_buffer_name, (int, str)):
        ᐁgrid_dim = get_buffer(asarray(grid).shape, buffer_or_buffer_name, nullify=False)
    else:
        ᐁgrid_dim = buffer_or_buffer_name
        if asarray(ᐁgrid_dim).shape != asarray(grid).shape:
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
                    else:  # order == 8
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
                # Write result of differentiation to buffer
                ᐁgrid_dim[i, j, k] = value
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
    dist = x - index  # Distance between centre grid point and x; -0.5 <= dist < 0.5
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
