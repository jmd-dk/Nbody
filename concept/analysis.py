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
cimport('from communication import communicate_ghosts, get_buffer')
cimport('from graphics import plot_powerspec')
cimport('from linear import get_linear_powerspec')
cimport(
    'from mesh import            '
    '    diff_domaingrid,        '
    '    fft,                    '
    '    get_deconvolution,      '
    '    get_fftw_slab,          '
    '    interpolate_components, '
    '    slab_decompose,         '
)



# Top level function for computing, plotting and saving power spectra
@cython.header(
    # Arguments
    components=list,
    filename=str,
    # Locals
    powerspec_declarations=list,
    powerspec_declaration=object,  # PowerspecDeclaration
    returns='void',
)
def powerspec(components, filename):
    # Get needed power spectrum declarations
    powerspec_declarations = get_powerspec_declarations(components)
    # Compute power spectrum for each power spectrum declaration.
    for powerspec_declaration in powerspec_declarations:
        # Compute the power spectrum of the non-linearly evolved
        # components in this power spectrum declaration. The result is
        # stored in powerspec_declaration.power. Only the master process
        # holds the full power spectrum.
        compute_powerspec(powerspec_declaration)
        # If specified, also compute the linear power spectrum. The
        # result is stored in powerspec_declaration.linear_power. Only
        # the master process holds the linear power spectrum.
        if powerspec_declaration.power_linear is not None:
            compute_powerspec_linear(powerspec_declaration)
    # Saving and plotting the power spectra is solely up to the master
    if not master:
        return
    # Dump power spectra to collective data file
    save_powerspec(powerspec_declarations, filename)
    # Dump power spectra to individual image files
    plot_powerspec(powerspec_declarations, filename)

# Function for getting declarations for all needed power spectra,
# given a list of components.
@cython.header(
    # Arguments
    components=list,
    # Locals
    component_combination=list,
    component_combinations=list,
    data_select=dict,
    do_data='bint',
    do_plot='bint',
    gridsize='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_indices='Py_ssize_t[::1]',
    n_modes='Py_ssize_t[::1]',
    n_modes_max='Py_ssize_t',
    plot_select=dict,
    power='double[::1]',
    power_linear='double[::1]',
    powerspec_declarations=list,
    returns=list,
)
def get_powerspec_declarations(components):
    # Look up power spectrum declarations in cache
    powerspec_declarations = powerspec_declarations_cache.get(tuple(components), [])
    if powerspec_declarations:
        return powerspec_declarations
    # Generate list of lists storing all possible (unordered)
    # combinations of the passed components.
    component_combinations = list(
        map(
            list,
            itertools.chain.from_iterable(
                [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
            ),
        )
    )
    # Construct dicts to be used with the is_selected function
    data_select = {key: val['data'] for key, val in powerspec_select.items()}
    plot_select = {key: val['plot'] for key, val in powerspec_select.items()}
    # Construct power spectrum declarations
    for component_combination in component_combinations:
        do_data = is_selected(component_combination, data_select)
        do_plot = is_selected(component_combination, plot_select)
        if not do_data and not do_plot:
            continue
        # A power spectrum is to be computed of this component
        # combination. The power spectrum gridsize should be the largest
        # of the individual power spectrum gridsizes of each component.
        gridsize = np.max(
            [component.powerspec_gridsize for component in component_combination]
        )
        # Get k_bin_indices, k_bin_centers and n_modes
        # for the given grid size
        k_bin_indices, k_bin_centers, n_modes, n_modes_max = get_powerspec_bins(gridsize)
        # Allocate grid for storing the power
        power = empty(bcast(k_bin_centers.shape[0] if master else None), dtype=C2np['double'])
        power_linear = (asarray(power).copy() if powerspec_include_linear else None)
        # Add power spectrum declaration
        powerspec_declarations.append(
            PowerspecDeclaration(
                component_combination, do_data, do_plot, gridsize, k_bin_indices, k_bin_centers,
                n_modes, n_modes_max, power, power_linear,
            )
        )
    # Store power spectrum declarations in cache
    powerspec_declarations_cache[tuple(components)] = powerspec_declarations
    return powerspec_declarations
# Create the PowerspecDeclaration type
PowerspecDeclaration = collections.namedtuple(
    'PowerspecDeclaration',
    (
        'components', 'do_data', 'do_plot', 'gridsize', 'k_bin_indices', 'k_bin_centers',
        'n_modes', 'n_modes_max', 'power', 'power_linear',
    ),
)
# Cache used by the get_powerspec_declarations function
cython.declare(powerspec_declarations_cache=dict)
powerspec_declarations_cache = {}

# Function for constructing arrays k_bin_indices, k_bin_centers and
# n_modes, describing the binning of power spectra.
@cython.header(
    # Arguments
    gridsize='Py_ssize_t',
    # Locals
    any_particles='bint',
    compute_sumk='bint',
    deconv='double',
    i='Py_ssize_t',
    index_largest_mode='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    k_bin_center='double',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    k_bin_indices='Py_ssize_t[::1]',
    k_bin_size='double',
    k_max='double',
    k_min='double',
    k_magnitude='double',
    k2='Py_ssize_t',
    n_modes='Py_ssize_t[::1]',
    n_modes_fine='Py_ssize_t[::1]',
    n_modes_max='Py_ssize_t',
    powerspec_bins=tuple,
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    slab='double[:, :, ::1]',
    sumk='Py_ssize_t',
    symmetry_multiplicity='int',
    returns=tuple,
)
def get_powerspec_bins(gridsize):
    """The returned arrays are:
    - k_bin_indices: Mapping from k⃗² (grid units) to bin index, i.e.
        k_bin_index = k_bin_indices[k2]
      All processes will have a copy of this array.
    - k_bin_centers: Mapping from bin index to |k⃗|, i.e.
        k_bin_center = k_bin_centers[k_bin_index]
      This array lives on the master process only.
    - n_modes: Mapping from bin index to number of modes, i.e.
        n = n_modes[bin_index]
      This array lives on the master process only.
    """
    # Look up in the cache
    powerspec_bins = powerspec_bins_cache.get(gridsize)
    if powerspec_bins:
        return powerspec_bins
    # Maximum value of k² (grid units)
    k2_max = 3*(gridsize//2)**2
    # Maximum and minum k values
    k_min = ℝ[2*π/boxsize]
    k_max = ℝ[2*π/boxsize]*sqrt(k2_max)
    # Construct linear k bins, each with a linear size given by the
    # powerspec_binsize parameter. The k_bin_centers will be changed
    # later according to the k² values on the 3D grid that falls inside
    # each bin. The final placing of the bin centers are then really
    # defined indirectly by k_bin_indices below (which depend on the
    # initial values given to k_bin_centers).
    # A bin size below powerspec_binsize_min is guaranteed to never bin
    # separate k² together in the same bin, and so powerspec_binsize_min
    # is the smallest bin size allowed.
    powerspec_binsize_min = (0.5 - 1e-2)*(
          ℝ[2*π/boxsize]*sqrt(3*((gridsize + 2)//2)**2 + 1)
        - ℝ[2*π/boxsize]*sqrt(3*((gridsize + 2)//2)**2)
    )
    k_bin_size = np.max((powerspec_binsize, powerspec_binsize_min))
    k_bin_centers = np.arange(
        k_min + (0.5 - 1e+1*machine_ϵ)*k_bin_size,
        k_max + k_bin_size,
        k_bin_size,
    )
    # Construct array mapping k2 (grid units) to bin index
    k_bin_indices = empty(k2_max + 1, dtype=C2np['Py_ssize_t'])
    k_bin_indices[0] = 0
    i = 1
    for k2 in range(1, k_bin_indices.shape[0]):
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        # Find index of closest bin center
        for i in range(i, ℤ[k_bin_centers.shape[0]]):
            k_bin_center = k_bin_centers[i]
            if k_bin_center > k_magnitude:
                # k2 belongs to either bin (i - 1) or bin i
                if k_magnitude - k_bin_centers[ℤ[i - 1]] < k_bin_center - k_magnitude:
                    k_bin_indices[k2] = ℤ[i - 1]
                else:
                    k_bin_indices[k2] = i
                break
    # Array counting the multiplicity (number of modes) of each
    # k² in the 3D grid.
    n_modes_fine = zeros(k_bin_indices.shape[0], dtype=C2np['Py_ssize_t'])
    # Get distributed slab
    slab = get_fftw_slab(gridsize)
    # We only actually use the slab for its shape. In Fourier space,
    # the slab is transposed in the first two dimensions.
    size_j, size_i, size_k = slab.shape[0], slab.shape[1], slab.shape[2]
    # Loop over the slab
    any_particles = compute_sumk = False
    for i, j, k, k2, sumk, symmetry_multiplicity, deconv in slab_fourier_loop(
        gridsize, size_i, size_j, size_k, any_particles, compute_sumk,
    ):
        # Increase the multiplicity of this k²
        n_modes_fine[k2] += symmetry_multiplicity
    # Sum n_modes_fine into the master process
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else n_modes_fine),
        recvbuf=(n_modes_fine if master else None),
        op=MPI.SUM,
    )
    # The master process now holds all the information needed
    n_modes_max = 0
    if not master:
        # The slave processes return now.
        # Updated values of k_bin_indice are received from the master.
        # This is the only data known to the slaves.
        Bcast(k_bin_indices)
        k_bin_centers = n_modes = None
        powerspec_bins_cache[gridsize] = k_bin_indices, k_bin_centers, n_modes, n_modes_max
        return k_bin_indices, k_bin_centers, n_modes, n_modes_max
    # Redefine k_bin_centers so that each element is the mean of all the
    # k values that falls within the bin, using the multiplicity
    # (n_modes_fine) as weight. Simultaneously construct n_modes from
    # n_modes_fine, where n_modes is just like n_modes_fine, but
    # counting the multiplicity of the bins, rather than the individual
    # k² elements. Finally, we also find the n_modes_max, the largest
    # value in n_modes.
    k_bin_centers[:] = 0
    n_modes = zeros(k_bin_centers.shape[0], dtype=C2np['Py_ssize_t'])
    for k2 in range(1, n_modes_fine.shape[0]):
        if ℤ[n_modes_fine[k2]] == 0:
            continue
        k_magnitude = ℝ[2*π/boxsize]*sqrt(k2)
        k_bin_index = k_bin_indices[k2]
        n_modes[k_bin_index] += ℤ[n_modes_fine[k2]]
        k_bin_centers[k_bin_index] += ℤ[n_modes_fine[k2]]*k_magnitude
    for k_bin_index in range(k_bin_centers.shape[0]):
        if ℤ[n_modes[k_bin_index]] > 0:
            k_bin_centers[k_bin_index] /= ℤ[n_modes[k_bin_index]]
            if ℤ[n_modes[k_bin_index]] > n_modes_max:
                n_modes_max = ℤ[n_modes[k_bin_index]]
    # We wish to remove bins with a mode count of 0.
    # Modify k_bin_indices so that consecutive bin indices
    # correspond to non-empty bins.
    k_bin_index_prev = k_bin_indices[0]
    for k2 in range(1, k_bin_indices.shape[0]):
        k_bin_index = k_bin_indices[k2]
        if k_bin_index == k_bin_index_prev or n_modes[k_bin_index] == 0:
            k_bin_indices[k2] = k_bin_indices[k2 - 1]
        elif k_bin_index > k_bin_index_prev:
            k_bin_indices[k2] = k_bin_indices[k2 - 1] + 1
            k_bin_index_prev = k_bin_index
    # The final values of k_bin_indices should be known to all processes
    Bcast(k_bin_indices)
    # Remove bins with mode count 0
    mask = (asarray(n_modes) > 0)
    n_modes = asarray(n_modes)[mask]
    k_bin_centers = asarray(k_bin_centers)[mask]
    powerspec_bins_cache[gridsize] = k_bin_indices, k_bin_centers, n_modes, n_modes_max
    return k_bin_indices, k_bin_centers, n_modes, n_modes_max
# Cache used by the get_powerspec_bins function
cython.declare(powerspec_bins_cache=dict)
powerspec_bins_cache = {}

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its power spectrum.
@cython.header(
    # Arguments
    powerspec_declaration=object,  # PowerspecDeclaration
    # Locals
    a='double',
    any_fluid='bint',
    any_particles='bint',
    component='Component',
    components=list,
    components_str=str,
    compute_sumk='bint',
    deconv='double',
    deconv2='double',
    grid='double[:, :, ::1]',
    grids=dict,
    i='Py_ssize_t',
    im='double',
    j='Py_ssize_t',
    k='Py_ssize_t',
    k_bin_indices='Py_ssize_t[::1]',
    k_bin_index='Py_ssize_t',
    k2='Py_ssize_t',
    gridsize='Py_ssize_t',
    n_modes='Py_ssize_t[::1]',
    normalization='double',
    power='double[::1]',
    power_jik='double',
    re='double',
    representation=str,
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    slab='double[:, :, ::1]',
    slab_fluid='double[:, :, ::1]',
    slab_jik='double*',
    slab_particles='double[:, :, ::1]',
    slab_particles_shifted='double[:, :, ::1]',
    slabs=dict,
    sumk='Py_ssize_t',
    symmetry_multiplicity='int',
    θ='double',
    returns='void',
)
def compute_powerspec(powerspec_declaration):
    # Extract some variables from the power spectrum declaration
    components    = powerspec_declaration.components
    gridsize      = powerspec_declaration.gridsize
    k_bin_indices = powerspec_declaration.k_bin_indices
    n_modes       = powerspec_declaration.n_modes
    power         = powerspec_declaration.power
    # Begin progress message
    if len(components) == 1:
        component = components[0]
        masterprint(f'Computing power spectrum of {component.name} ...')
    else:
        components_str = ', '.join([component.name for component in components])
        masterprint(f'Computing power spectrum of {{{components_str}}} ...')
    # Populate grids with physical densities from all of the components.
    # A separate grid will be used for particle and fluid components.
    grids = interpolate_components(components, 'ρ', gridsize, powerspec_interpolation,
        include_shifted_particles=powerspec_interlacing)
    # Slab decompose the grids
    slabs = {
        representation: slab_decompose(grid, f'slab_{representation}', prepare_fft=True)
        for representation, grid in grids.items()
    }
    # Do a forward in-place Fourier transform of the slabs
    for slab in slabs.values():
        if slab is None:
            continue
        fft(slab, 'forward')
        # In Fourier space, the slab is transposed
        # in the first two dimensions.
        size_j, size_i, size_k = slab.shape[0], slab.shape[1], slab.shape[2]
    # Nullify the reused power array
    power[:] = 0
    # Loop over the slabs
    slab_particles = slabs['particles']
    if powerspec_interlacing:
        slab_particles_shifted = slabs['particles_shifted']
    slab_fluid = slabs['fluid']
    any_particles = (slab_particles is not None)
    any_fluid     = (slab_fluid     is not None)
    compute_sumk = (any_particles and powerspec_interlacing)
    for i, j, k, k2, sumk, symmetry_multiplicity, deconv in slab_fourier_loop(
        gridsize, size_i, size_j, size_k, any_particles, compute_sumk,
    ):
        # Power from the complex number at [j, i, k]
        power_jik = 0
        # Add deconvolved power from the particles slab
        with unswitch(3):
            if any_particles:
                # The deconvolution factor given by deconv is that of
                # order 1 (NGP) interpolation. The full deconvolution
                # needed is deconv**powerspec_interpolation.
                # Below we compute the square of the full deconvolution.
                deconv2 = deconv**ℤ[2*powerspec_interpolation]
                # Add power from the particles slab and possibly the
                # shifted particles slab.
                with unswitch(3):
                    if powerspec_interlacing:
                        # Rotate the phase of the complex number of the
                        # shifted particles slab at this [j, i, k] by θ,
                        # which according to harmonic averaging is
                        #   θ = (kx + ky + kz)*(gridsize/boxsize)/2
                        #     = π/gridsize*(ki + kj + kk)
                        slab_jik = cython.address(slab_particles_shifted[j, i, k:])
                        re, im = slab_jik[0], slab_jik[1]
                        θ = ℝ[π/gridsize]*sumk
                        re, im = re*ℝ[cos(θ)] - im*ℝ[sin(θ)], re*ℝ[sin(θ)] + im*ℝ[cos(θ)]
                        # Add the interlaced, deconvolved power
                        slab_jik = cython.address(slab_particles[j, i, k:])
                        power_jik += 0.25*((slab_jik[0] + re)**2 + (slab_jik[1] + im)**2)*deconv2
                    else:
                        # Add the deconvolved power
                        slab_jik = cython.address(slab_particles[j, i, k:])
                        power_jik += (slab_jik[0]**2 + slab_jik[1]**2)*deconv2
        # Add power from the fluid slab
        with unswitch(3):
            if any_fluid:
                slab_jik = cython.address(slab_fluid[j, i, k:])
                power_jik += slab_jik[0]**2 + slab_jik[1]**2
        # Add power at this k² to the corresponding bin
        k_bin_index = k_bin_indices[k2]
        power[k_bin_index] += symmetry_multiplicity*power_jik
    # Sum power into the master process
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else power),
        recvbuf=(power        if master else None),
        op=MPI.SUM,
    )
    # The master process now holds all the information needed
    if not master:
        return
    # We need to transform power from being the sum to being the
    # mean, by dividing by n_modes.
    # To completely remove the current normalization of the power, we
    # need to divide by the squared sum of values on the grids/slabs.
    # As we interpolated physical densities ρ to the grids, the sum of
    # all values will be
    # sum(ρᵢⱼₖ) = sum(ρᵢⱼₖ*V_cell)/V_cell = sum(massᵢⱼₖ)/V_cell,
    # with V_cell = (a*boxsize/gridsize)**3 the phyiscal cell volume and
    # massᵢⱼₖ the mass interpolated onto grid point [i, j, k]. For both
    # particle and fluid components, the total mass may be written as
    # sum(massᵢⱼₖ) = (a*boxsize)**3*ρ_bar
    #              = boxsize**3*a**(-3*w_eff)*ϱ_bar.
    # Thus, the sum of values in the interpolated grid is
    # sum(ρᵢⱼₖ) = gridsize**3*a**(-3(1 + w_eff))*ϱ_bar,
    # summed over all components.
    # As said, we need to divide the power by the square of sum(ρᵢⱼₖ).
    # To now add in a proper normalization, we need to multiply by
    # boxsize**3, resulting in a properly normalized power spectrum in
    # units of unit_length**3.
    a = universals.a
    normalization = 0
    for component in components:
        normalization += a**(-3*(1 + component.w_eff(a=a)))*component.ϱ_bar
    normalization *= (gridsize*ℝ[1/sqrt(boxsize)])**3
    normalization **= -2
    for k_bin_index in range(power.shape[0]):
        power[k_bin_index] *= normalization/n_modes[k_bin_index]
    # Done with the main power spectrum computation
    masterprint('done')

# Function which given a power spectrum declaration correctly populated
# with all fields will compute its linear CLASS power spectrum.
@cython.header(
    # Arguments
    powerspec_declaration=object,  # PowerspecDeclaration
    # Locals
    component='Component',
    components=list,
    components_str=str,
    k_bin_centers='double[::1]',
    power_linear='double[::1]',
    returns='void',
)
def compute_powerspec_linear(powerspec_declaration):
    # Extract some variables from the power spectrum declaration
    components    = powerspec_declaration.components
    k_bin_centers = powerspec_declaration.k_bin_centers
    power_linear  = powerspec_declaration.power_linear
    # Begin progress message
    if len(components) == 1:
        component = components[0]
        masterprint(f'Computing linear power spectrum of {component.name} ...')
    else:
        components_str = ', '.join([component.name for component in components])
        masterprint(f'Computing linear power spectrum of {{{components_str}}} ...')
    # Fill power_linear with values of the linear power spectrum.
    # Only the master will hold the values.
    get_linear_powerspec(components, k_bin_centers, power=power_linear)
    # Done with the linear power spectrum computation
    masterprint('done')

# Iterator implementing looping over Fourier space slabs
@cython.iterator
def slab_fourier_loop(
    gridsize, size_i, size_j, size_k, any_particles, compute_sumk,
):
    # Cython declarations for variables used for the iteration,
    # not including those to yield.
    # Do not write these using the decorator syntax above this function.
    cython.declare(
        j_global='Py_ssize_t',
        k2_max='Py_ssize_t',
        ki='Py_ssize_t',
        ki_plus_kj='Py_ssize_t',
        kj='Py_ssize_t',
        kk='Py_ssize_t',
        nyquist='Py_ssize_t',
        deconv_ij='double',
        deconv_j='double',
    )
    # Maximum value of k² (grid units)
    nyquist = gridsize//2
    k2_max = 3*nyquist**2
    # When any_particles is False,
    # this is the returned deconvolution factor.
    deconv = 1
    # When compute_sumk is False,
    # this is the returned sum of wave vector elements.
    sumk = 0
    # To satisfy the compiler
    deconv_j = deconv_ij = deconv_ijk = 1
    # When looping in Fourier space, remember that the first and second
    # dimension are transposed.
    for j in range(size_j):
        # The j-component of the wave vector (grid units).
        # Since the slabs are distributed along the j-dimension,
        # an offset must be used.
        j_global = ℤ[size_j*rank] + j
        kj = j_global - gridsize if j_global > ℤ[gridsize//2] else j_global
        # The j-component of the deconvolution
        with unswitch(1):
            if any_particles:
                deconv_j = get_deconvolution(kj*ℝ[π/gridsize])
        # Loop over the entire first dimension
        for i in range(gridsize):
            # The i-component of the wave vector
            ki = i - gridsize if i > ℤ[gridsize//2] else i
            # The product of the i- and the j-component
            # of the deconvolution.
            with unswitch(2):
                if any_particles:
                    deconv_ij = get_deconvolution(ki*ℝ[π/gridsize])*deconv_j
            # The sum of wave vector elements
            with unswitch(2):
                if compute_sumk:
                    ki_plus_kj = ki + kj
            # Loop over the entire last dimension in steps of two,
            # as contiguous pairs of elements are the real and
            # imaginary part of the same complex number.
            for k in range(0, size_k, 2):
                # The k-component of the wave vector
                kk = k//2
                # The squared magnitude of the wave vector
                k2 = ℤ[ℤ[kj**2] + ki**2] + kk**2
                # Skip the DC component.
                # For some reason, the k = k_max mode is
                # highly uncertain. Skip this as well.
                if k2 == 0 or k2 == k2_max:
                    continue
                # The sum of wave vector elements
                with unswitch(3):
                    if compute_sumk:
                        sumk = ki_plus_kj + kk
                # Because of the complex-conjugate symmetry, the slabs
                # only contain half of the data; the positive kk
                # frequencies. Including this missing half lead to truer
                # statistics, altering the binned power spectrum. The
                # symmetry_multiplicity variable counts the number of
                # times this grid point should be counted.
                if kk == 0 or kk == nyquist:
                    symmetry_multiplicity = 1
                else:
                    symmetry_multiplicity = 2
                # Construct the complete deconvolution
                with unswitch(3):
                    if any_particles:
                        # The total (NGP) deconvolution factor
                        deconv = deconv_ij*get_deconvolution(kk*ℝ[π/gridsize])
                # To get the complex number at this [j, i, k]
                # of a slab, use
                # slab_jik = cython.address(slab[j, i, k:])
                # after which the real and the imaginary part
                # can be accessed as
                # slab_jik[0]  # real part
                # slab_jik[1]  # imag part
                #
                # Yield the local indices, the global k2, the symmetry
                # multiplicity and the deconvolution factor.
                yield i, j, k, k2, sumk, symmetry_multiplicity, deconv

# Function for saving already computed power spectra to disk
@cython.header(
    # Arguments
    powerspec_declarations=list,
    filename=str,
    # Locals
    col='int',
    data='double[:, ::1]',
    delimiter=str,
    fmt=list,
    gridsize='Py_ssize_t',
    header=str,
    k_bin_centers='double[::1]',
    n_cols='int',
    n_modes='Py_ssize_t[::1]',
    n_modes_float='double[::1]',
    n_rows='int',
    power='double[::1]',
    power_linear='double[::1]',
    powerspec_declaration=object,  # PowerspecDeclaration
    powerspec_declaration_group=list,
    powerspec_declaration_groups=dict,
    powerspec_header_info=object,  # PowerspecHeaderInfo
    size='Py_ssize_t',
    spectrum_plural=str,
    topline=str,
    σ='double',
    returns='void',
)
def save_powerspec(powerspec_declarations, filename):
    """It is expected that this function
    is called by the master process only.
    All power spectra will be saves into a single text file.
    """
    # Discard power spectrum declarations that should not be saved
    powerspec_declarations = [
        powerspec_declaration
        for powerspec_declaration in powerspec_declarations
        if powerspec_declaration.do_data
    ]
    if not powerspec_declarations:
        return
    # Get header, format and delimiter specifier for the data file
    powerspec_header_info = get_powerspec_header(powerspec_declarations)
    header = powerspec_header_info.header
    spectrum_plural = 'spectrum' if len(powerspec_declarations) == 1 else 'spectra'
    masterprint(f'Saving power {spectrum_plural} to "{filename}" ...')
    # The top line of the header, stating general information
    topline = unicode(
        f'Power {spectrum_plural} from CO𝘕CEPT job {jobid} at t = '
        + f'{{:.{powerspec_significant_figures}g}} '.format(universals.t)
        + f'{unit_time}'
        + (
            f', a = ' + f'{{:.{powerspec_significant_figures}g}}'.format(universals.a)
            if enable_Hubble else ''
        )
        + '.'
    )
    # The output data consists of a "k" column and a "modes" column for
    # each unique gridsize, along with a "power" column for each power
    # spectrum and possibly another "power" if the linear power spectrum
    # should be outputted as well. The number of rows in a column
    # depends on the gridsize, but to make it easier to read back in we
    # make all columns the same length by appending NaNs as required
    # (zeros for the modes).
    # Get a 2D array with the right size for storing all data.
    for powerspec_declaration_group in powerspec_header_info.powerspec_declaration_groups.values():
        powerspec_declaration = powerspec_declaration_group[0]
        n_rows = powerspec_declaration.k_bin_centers.shape[0]
        break
    n_cols = (
        2*len(powerspec_header_info.powerspec_declaration_groups)
        + len(powerspec_declarations)*(2 if powerspec_include_linear else 1)
    )
    data = get_buffer((n_rows, n_cols))
    # Fill in data columns
    col = 0
    for gridsize, powerspec_declaration_group in (
        powerspec_header_info.powerspec_declaration_groups.items()
    ):
        powerspec_declaration = powerspec_declaration_group[0]
        k_bin_centers = powerspec_declaration.k_bin_centers
        n_modes       = powerspec_declaration.n_modes
        size = k_bin_centers.shape[0]
        # New k
        data[:size, col] = k_bin_centers
        data[size:, col] = NaN
        col += 1
        # New modes
        n_modes_float = asarray(n_modes, dtype=C2np['double'])
        data[:size, col] = n_modes_float
        data[size:, col] = 0
        col += 1
        for powerspec_declaration in powerspec_declaration_group:
            # New power
            power = powerspec_declaration.power
            data[:size, col] = power
            data[size:, col] = NaN
            col += 1
            # Compute the rms density variation
            # and insert it into the header.
            σ = compute_powerspec_σ(powerspec_declaration)
            header = re.sub(r'= \{.+?\}', lambda m, σ=σ: m.group().format(σ), header, 1)
            # New linear power and rms density variation
            with unswitch:
                if powerspec_include_linear:
                    power_linear = powerspec_declaration.power_linear
                    data[:size, col] = power_linear
                    data[size:, col] = NaN
                    col += 1
                    σ = compute_powerspec_σ(powerspec_declaration, linear=True)
                    header = re.sub(r'= \{.+?\}', lambda m, σ=σ: m.group().format(σ), header, 1)
    # Save data and header to text file
    np.savetxt(
        filename,
        data,
        fmt=powerspec_header_info.fmt,
        delimiter=powerspec_header_info.delimiter,
        header=f'{topline}\n{header}',
    )
    masterprint('done')

# Pure Python function for generating the header for a power spectrum
# data file, given a list of power spectrum declarations.
def get_powerspec_header(powerspec_declarations):
    """Besides the header, this function also returns a list of data
    format specifiers, the delimter needed between the data columns and
    a dict mapping power spectrum gridsizes to lists of
    power spectrum declarations.
    Importantly, the supplied list of power spectrum declarations should
    only contain declarations for which do_data is True.
    """
    # Look up in the cache
    key = tuple([
        tuple(powerspec_declaration.components)
        for powerspec_declaration in powerspec_declarations
    ])
    powerspec_header_info = powerspec_header_cache.get(key)
    if powerspec_header_info:
        # Cached result found.
        # Which components to include in the power spectrum
        # computation/plot may change over time due to components
        # changing their passive/active/terminated state, which in turn
        # change the passed powerspec_declarations. As the key used
        # above depends on the components only, the resulting cached
        # result may hold outdated PowerspecDeclaration instances.
        # Update these before returning.
        for powerspec_declaration_group in (
            powerspec_header_info.powerspec_declaration_groups.values()
        ):
            for i, powerspec_declaration_cached in enumerate(powerspec_declaration_group):
                for powerspec_declaration in powerspec_declarations:
                    if powerspec_declaration_cached.components == powerspec_declaration.components:
                        powerspec_declaration_group[i] = powerspec_declaration
                        break
        return powerspec_header_info
    # A column mapping each component to a number
    components = []
    for powerspec_declaration in powerspec_declarations:
        for component in powerspec_declaration.components:
            if component not in components:
                components.append(component)
    longest_name_size = np.max([len(component.name) for component in components])
    column_components = ['Below, the following component mapping is used:']
    for i, component in enumerate(components):
        column_components.append(
            f'  {{:<{longest_name_size + 1}}} {i}'.format(f'{component.name}:')
        )
    # Group power spectrum declarations according to their gridsize,
    # in descending order.
    powerspec_declaration_groups_unordered = collections.defaultdict(list)
    for gridsize, powerspec_declarations_iter in itertools.groupby(
        powerspec_declarations,
        key=operator.attrgetter('gridsize'),
    ):
        powerspec_declaration_groups_unordered[gridsize] += list(powerspec_declarations_iter)
    powerspec_declaration_groups = {
        gridsize: powerspec_declaration_groups_unordered[gridsize]
        for gridsize in sorted(powerspec_declaration_groups_unordered, reverse=True)
    }
    # The column headings
    column_headings = {
        'k': unicode(f'k [{unit_length}⁻¹]'),
        'modes': 'modes',
        'power': unicode(f'power [{unit_length}³]'),
    }
    # The rms density variation σ will be written above each power
    # spectrum column. Construct the "σ₈" (or similar) string based on
    # R_tophat. By convention, the unit is Mpc/h.
    σ_unit = units.Mpc/(H0/(100*units.km/(units.s*units.Mpc))) if enable_Hubble else units.Mpc
    σ_str = ''.join([unicode('σ'), unicode_subscript(f'{R_tophat/σ_unit:.3g}'), ' = '])
    # The output data consists of a "k" column and a "modes" column for
    # each unique gridsize, along with a "power" column for each power
    # spectrum.
    # Determine the headings for each column and their format specifier.
    group_spacing = 1
    group_delimiter = ' '*group_spacing
    col = 0
    components_heading = []
    σs_heading = []
    columns_heading = []
    fmt = []
    fmt_float = f'%-{{}}.{powerspec_significant_figures - 1}e'
    n_chars_nonsignificant = len(f'{1e+100:.1e}') - 2
    width_float = powerspec_significant_figures + n_chars_nonsignificant
    fmt_int = '%{}u'
    for gridsize, powerspec_declaration_group in powerspec_declaration_groups.items():
        powerspec_declaration = powerspec_declaration_group[0]
        if col > 0:
            # New group with new gridsize begins. Insert additional
            # spacing by modifying the last elements of the
            # *s_heading and fmt.
            components_heading.append(components_heading.pop() + group_delimiter)
            σs_heading.append(σs_heading.pop() + group_delimiter)
            columns_heading.append(columns_heading.pop() + group_delimiter)
            fmt.append(fmt.pop() + group_delimiter)
        # New k
        col += 1
        column_heading = column_headings['k']
        width = np.max((width_float, len(column_heading) + 2*(col == 1)))
        components_heading.append(' '*(width - 2*(col == 1)))
        σs_heading.append(' '*(width - 2*(col == 1)))
        extra_spacing = width - len(column_heading) - 2*(col == 1)
        columns_heading.append(
            ' '*(extra_spacing//2) + column_heading + ' '*(extra_spacing - extra_spacing//2)
        )
        fmt.append(fmt_float.format(width))
        # New modes
        col += 1
        column_heading = column_headings['modes']
        width = np.max((len(str(powerspec_declaration.n_modes_max)), len(column_heading)))
        components_heading.append(' '*width)
        σs_heading.append(' '*width)
        extra_spacing = width - len(column_heading)
        columns_heading.append(
            ' '*(extra_spacing//2) + column_heading + ' '*(extra_spacing - extra_spacing//2)
        )
        fmt.append(fmt_int.format(width))
        for powerspec_declaration in powerspec_declaration_group:
            # New power and possibly linear power
            for power_type in range(1 + powerspec_include_linear):
                col += 1
                if power_type == 0:  # Non-linear
                    component_heading = get_integerset_strrep([
                        components.index(component)
                        for component in powerspec_declaration.components
                    ])
                    if len(powerspec_declaration.components) == 1:
                        component_heading = f'component {component_heading}'
                    else:
                        component_heading = f'components {{{component_heading}}}'
                else:  # power_type == 1 (linear)
                    component_heading = '(linear)'
                σ_significant_figures = width_float - len(σ_str) - n_chars_nonsignificant
                if σ_significant_figures < 2:
                    σ_significant_figures = 2
                size1 = width_float - len(σ_str)
                size2 = len(f'{{:<{size1}.{σ_significant_figures - 1}e}}'.format(1e+100))
                size = np.max((size1, size2))
                σ_heading = σ_str + f'{{:<{size}.{σ_significant_figures - 1}e}}'
                column_heading = column_headings['power']
                width = np.max((
                    width_float,
                    len(column_heading),
                    len(component_heading),
                    len(σ_heading.format(1e+100)),
                ))
                extra_spacing = width - len(component_heading)
                components_heading.append(
                    ' '*(extra_spacing//2) + component_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - len(σ_heading.format(1e+100))
                σs_heading.append(
                    ' '*(extra_spacing//2) + σ_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - len(column_heading)
                columns_heading.append(
                    ' '*(extra_spacing//2) + column_heading
                    + ' '*(extra_spacing - extra_spacing//2)
                )
                extra_spacing = width - width_float
                fmt.append(' '*(extra_spacing//2) + fmt_float.format(width - extra_spacing//2))
    # Construct group header
    group_header_underlines = []
    delimiter = ' '*2
    for col, column_heading in enumerate(columns_heading):
        if column_heading.strip() == column_headings['k']:
            if col > 0:
                width -= group_spacing
                group_header_underlines.append('/' + unicode('‾')*(width - 2) + '\\')
            width = len(column_heading)
        else:
            width += len(delimiter) + len(column_heading)
    group_header_underlines.append('/' + unicode('‾')*(width - 2) + '\\')
    group_headers = []
    for gridsize, group_header_underline in zip(
        powerspec_declaration_groups, group_header_underlines,
    ):
        group_heading = f'grid size {gridsize}'
        extra_spacing = len(group_header_underline) - len(group_heading)
        group_headers.append(
            ' '*(extra_spacing//2) + group_heading + ' '*(extra_spacing - extra_spacing//2)
        )
    # Put it all together to a collective header string
    header = '\n'.join([
        '',
        *column_components,
        '',
        (delimiter + group_delimiter).join(group_headers),
        (delimiter + group_delimiter).join(group_header_underlines),
        delimiter.join(components_heading),
        delimiter.join(σs_heading),
        delimiter.join(columns_heading),
    ])
    # Store in cache and return
    powerspec_header_cache[key] = PowerspecHeaderInfo(
        header, fmt, delimiter, powerspec_declaration_groups,
    )
    return powerspec_header_cache[key]
# Cache and type used by the get_powerspec_header() function
cython.declare(powerspec_header_cache=dict)
powerspec_header_cache = {}
PowerspecHeaderInfo = collections.namedtuple(
    'PowerspecHeaderInfo',
    ('header', 'fmt', 'delimiter', 'powerspec_declaration_groups'),
)

# Function which given a power spectrum declaration with the
# k_bin_centers and power fields correctly populated will compute the
# rms density variation of the power spectrum.
@cython.header(
    # Arguments
    powerspec_declaration=object,  # PowerspecDeclaration
    linear='bint',
    # Locals
    W='double',
    i='Py_ssize_t',
    k_bin_centers='double[::1]',
    k_bin_index='Py_ssize_t',
    k_magnitude='double',
    kR='double',
    power='double[::1]',
    σ2='double',
    σ2_integrand='double[::1]',
    returns='double',
)
def compute_powerspec_σ(powerspec_declaration, linear=False):
    k_bin_centers = powerspec_declaration.k_bin_centers
    power         = powerspec_declaration.power
    # If the σ to be computed is of the linear power spectrum,
    # we need to truncate k_bin_centers and power so that they do
    # not contain NaN's.
    if linear:
        power = powerspec_declaration.power_linear
        power = asarray(power)[~np.isnan(power)]
        k_bin_centers = k_bin_centers[:power.shape[0]]
    # Ensure that the global σ2_integrand array is large enough
    size = k_bin_centers.shape[0]
    if σ2_integrand_arr.shape[0] < size:
        σ2_integrand_arr.resize(size, refcheck=False)
    σ2_integrand = σ2_integrand_arr
    # The rms density variation σ_R (usually σ₈) is given by
    # σ² = ᔑd³k/(2π)³ W² power
    #    = 1/(2π)³ᔑ_0^∞ dk 4πk² W² power
    #    = 1/(2π²)ᔑ_0^∞ dk k² W² power,
    # where W = 3(sin(kR) - kR*cos(kR))/(kR)³.
    # Note that below, the factor 3 has been left out,
    # meaing that the variable W is really W/3.
    for k_bin_index in range(size):
        k_magnitude = k_bin_centers[k_bin_index]
        kR = k_magnitude*R_tophat
        if kR < 1e-3:
            # Use a Taylor expansion of W/3 around kR = 0
            W = ℝ[1/3] - ℝ[1/30]*kR**2
        else:
            W = (sin(kR) - kR*cos(kR))/kR**3
        σ2_integrand[k_bin_index] = (k_magnitude*W)**2*power[k_bin_index]
    # Do the integral, disregarding constant factors
    σ2 = np.trapz(σ2_integrand[:size], k_bin_centers)
    # The integrand above starts from k = k_min > 0, which means that
    # the interval from 0 to k_min has been left out. At k = 0, the
    # integrand vanishes. According to the trapezoidal rule, this means
    # that the full integral is missing the area of the triangle with
    # vertices (0, 0), (k_min, 0), (k_min, σ2_integrand[0]),
    # with k_min = k_bin_centers[0].
    σ2 += 0.5*k_bin_centers[0]*σ2_integrand[0]
    # Finally, remember the constant factor 1/(2π²) from the integral,
    # as well as the 3² missing from W².
    σ2 *= ℝ[3**2/(2*π**2)]
    # Return the rms density variation σ
    return sqrt(σ2)
# Array used by the compute_powerspec_σ() function
cython.declare(σ2_integrand_arr=object)
σ2_integrand_arr = empty(1, dtype=C2np['double'])

# Function which can measure different quantities of a passed component
@cython.header(
    # Arguments
    component='Component',
    quantity=str,
    communicate='bint',
    # Locals
    J_noghosts=object, # np.ndarray
    J_over_ϱ_plus_𝒫_2_i='double',
    J_over_ϱ_plus_𝒫_2_max='double',
    J_over_ϱ_2_i='double',
    J_over_ϱ_2_max='double',
    Jx_mv='double[:, :, ::1]',
    Jx_ptr='double*',
    Jy_mv='double[:, :, ::1]',
    Jy_ptr='double*',
    Jz_mv='double[:, :, ::1]',
    Jz_ptr='double*',
    N='Py_ssize_t',
    N_elements='Py_ssize_t',
    Vcell='double',
    a='double',
    diff_backward='double[:, :, ::1]',
    diff_forward='double[:, :, ::1]',
    diff_max='double[::1]',
    diff_max_dim='double',
    diff_size='double',
    dim='int',
    fluidscalar='FluidScalar',
    h='double',
    i='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    mom='double*',
    mom2='double',
    mom2_max='double',
    mom2_i='double',
    momx='double*',
    momy='double*',
    momz='double*',
    mom_i='double',
    names=list,
    t='double',
    v_rms='double',
    v_max='double',
    w='double',
    w_eff='double',
    Δdiff='double',
    Δdiff_max='double[::1]',
    Δdiff_max_dim='double',
    Δdiff_max_list=list,
    Δdiff_max_normalized_list=list,
    ΣJ_over_ϱ_plus_𝒫_2='double',
    Σmass='double',
    Σmom='double[::1]',
    Σmom_dim='double',
    Σmom2_dim='double',
    Σϱ='double',
    Σϱ2='double',
    ϱ='FluidScalar',
    ϱ_bar='double',
    ϱ_min='double',
    ϱ_mv='double[:, :, ::1]',
    ϱ_noghosts=object, # np.ndarray
    ϱ_ptr='double*',
    σ2mom_dim='double',
    σ2ϱ='double',
    σmom='double[::1]',
    σmom_dim='double',
    σϱ='double',
    𝒫_mv='double[:, :, ::1]',
    𝒫_ptr='double*',
    ᐁgrid_dim='double[:, :, ::1]',
    returns=object,  # double or tuple
)
def measure(component, quantity, communicate=True):
    """Implemented quantities are:
    'v_max'
    'v_rms'
    'momentum'
    'ϱ'              (fluid quantity)
    'mass'           (fluid quantity)
    'discontinuity'  (fluid quantity)
    """
    t = universals.t
    a = universals.a
    # Extract variables
    N = component.N if communicate else component.N_local
    N_elements = component.gridsize**3 if communicate else component.size_noghosts
    Vcell = boxsize**3/N_elements
    w     = component.w    (a=a)
    w_eff = component.w_eff(a=a)
    ϱ = component.ϱ
    ϱ_noghosts = asarray(ϱ.grid_noghosts)
    # Quantities exhibited by both particle and fluid components
    if quantity == 'v_max':
        # The propagation speed of information in
        # comoving coordinates is
        # v = c*sqrt(w)/a + ẋ, ẋ = dx/dt = u/a,
        # where u is the peculiar velocity.
        # For fluids we have
        # ϱ = a**(3*(1 + w_eff))ρ, J = a**4*(ρ + c⁻²P)u,
        # and so
        # u = a**(-4)*J/(ρ + c⁻²P)
        #   = a**(3*w_eff - 1)*J/(ϱ + c⁻²𝒫),
        # and then
        # v = c*sqrt(w)/a + a**(3*w_eff - 2)*J/(ϱ + c⁻²𝒫),
        # where c*sqrt(w) is an approximation for the local sound speed.
        # For particles we have w = 0 and ẋ = mom/(a**2*m), and so
        # v = mom/(a**2*mass).
        # In the case of decyaing (matter) particles, the mass at time a
        # is really a**(-3*w_eff)*mass, and so we get
        # v = mom/(a**(2 - 3*w_eff)*mass)
        if component.representation == 'particles':
            mom2_max = 0
            momx = component.momx
            momy = component.momy
            momz = component.momz
            for i in range(component.N_local):
                mom2_i = momx[i]**2 + momy[i]**2 + momz[i]**2
                if mom2_i > mom2_max:
                    mom2_max = mom2_i
            if communicate:
                mom2_max = allreduce(mom2_max, op=MPI.MAX)
            v_max = sqrt(mom2_max)/(a**(2 - 3*w_eff)*component.mass)
        elif component.representation == 'fluid':
            if (    component.boltzmann_order == -1
                or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
            ):
                # Without J as a fluid variable,
                # no explicit velocity exists.
                v_max = 0
            elif component.boltzmann_order == 0 and component.boltzmann_closure == 'class':
                # With J as a linear fluid variable, we only need to
                # consider one of its components. Also, the P = wρ
                # approximation is guaranteed to be enabled.
                ϱ_ptr  = component.ϱ .grid
                Jx_ptr = component.Jx.grid
                J_over_ϱ_2_max = 0
                for i in range(component.size):
                    J_over_ϱ_2_i = (Jx_ptr[i]/ϱ_ptr[i])**2
                    if J_over_ϱ_2_i > J_over_ϱ_2_max:
                        J_over_ϱ_2_max = J_over_ϱ_2_i
                if communicate:
                    J_over_ϱ_2_max = allreduce(J_over_ϱ_2_max, op=MPI.MAX)
                J_over_ϱ_plus_𝒫_2_max = 3*J_over_ϱ_2_max/(1 + w)**2
                v_max = a**(3*w_eff - 2)*sqrt(J_over_ϱ_plus_𝒫_2_max)
                # Since no non-linear evolution happens for J, the Euler
                # equation and hence the gradient of the pressure will
                # never be computed. This means that sound waves
                # cannot form, and so we do not need to take the sound
                # speed into account.
            else:
                # J is non-linear
                ϱ_ptr  = component.ϱ .grid
                𝒫_ptr  = component.𝒫 .grid
                Jx_ptr = component.Jx.grid
                Jy_ptr = component.Jy.grid
                Jz_ptr = component.Jz.grid
                J_over_ϱ_plus_𝒫_2_max = 0
                for i in range(component.size):
                    J_over_ϱ_plus_𝒫_2_i = (
                        (Jx_ptr[i]**2 + Jy_ptr[i]**2 + Jz_ptr[i]**2)
                        /(ϱ_ptr[i] + ℝ[light_speed**(-2)]*𝒫_ptr[i])**2
                    )
                    if J_over_ϱ_plus_𝒫_2_i > J_over_ϱ_plus_𝒫_2_max:
                        J_over_ϱ_plus_𝒫_2_max = J_over_ϱ_plus_𝒫_2_i
                if communicate:
                    J_over_ϱ_plus_𝒫_2_max = allreduce(J_over_ϱ_plus_𝒫_2_max, op=MPI.MAX)
                v_max = a**(3*w_eff - 2)*sqrt(J_over_ϱ_plus_𝒫_2_max)
                # Add the sound speed. When the P=wρ approxiamation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_max += light_speed*sqrt(w)/a
        return v_max
    elif quantity == 'v_rms':
        if component.representation == 'particles':
            mom2 = 0
            momx = component.momx
            momy = component.momy
            momz = component.momz
            for i in range(component.N_local):
                mom2 += momx[i]**2 + momy[i]**2 + momz[i]**2
            if communicate:
                mom2 = allreduce(mom2, op=MPI.SUM)
            v_rms = sqrt(mom2/N)/(a**(2 - 3*component.w_eff(a=a))*component.mass)
        elif component.representation == 'fluid':
            if (    component.boltzmann_order == -1
                or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
            ):
                # Without J as a fluid variable, no velocity exists
                v_rms = 0
            elif component.boltzmann_order == 0 and component.boltzmann_closure == 'class':
                # With J as a linear fluid variable, we only need to
                # consider one of its components. Also, the P = wρ
                # approximation is guaranteed to be enabled.
                ϱ_mv  = component.ϱ .grid_mv
                Jx_mv = component.Jx.grid_mv
                ΣJ_over_ϱ_plus_𝒫_2 = 0
                for         i in range(nghosts, ℤ[component.shape[0] - nghosts]):
                    for     j in range(nghosts, ℤ[component.shape[1] - nghosts]):
                        for k in range(nghosts, ℤ[component.shape[2] - nghosts]):
                            ΣJ_over_ϱ_plus_𝒫_2 += 3*(Jx_mv[i, j, k]/(ϱ_mv[i, j, k]*(1 + w)))**2
                if communicate:
                    ΣJ_over_ϱ_plus_𝒫_2 = allreduce(ΣJ_over_ϱ_plus_𝒫_2, op=MPI.SUM)
                v_rms = a**(3*w_eff - 2)*sqrt(ΣJ_over_ϱ_plus_𝒫_2/N_elements)
                # Since no non-linear evolution happens for J, the Euler
                # equation and hence the gradient of the pressure will
                # never be computed. This means that sound waves
                # cannot form, and so we do not need to take the sound
                # speed into account.
            else:
                # J is non-linear
                ϱ_mv  = component.ϱ .grid_mv
                𝒫_mv  = component.𝒫 .grid_mv
                Jx_mv = component.Jx.grid_mv
                Jy_mv = component.Jy.grid_mv
                Jz_mv = component.Jz.grid_mv
                ΣJ_over_ϱ_plus_𝒫_2 = 0
                for         i in range(nghosts, ℤ[component.shape[0] - nghosts]):
                    for     j in range(nghosts, ℤ[component.shape[1] - nghosts]):
                        for k in range(nghosts, ℤ[component.shape[2] - nghosts]):
                            ΣJ_over_ϱ_plus_𝒫_2 += (
                                (Jx_mv[i, j, k]**2 + Jy_mv[i, j, k]**2 + Jz_mv[i, j, k]**2)
                                /(ϱ_mv[i, j, k] + ℝ[light_speed**(-2)]*𝒫_mv[i, j, k])**2
                            )
                if communicate:
                    ΣJ_over_ϱ_plus_𝒫_2 = allreduce(ΣJ_over_ϱ_plus_𝒫_2, op=MPI.SUM)
                v_rms = a**(3*w_eff - 2)*sqrt(ΣJ_over_ϱ_plus_𝒫_2/N_elements)
                # Add the sound speed. When the P=wρ approxiamation is
                # False, the sound speed is non-global and given by the
                # square root of δ𝒫/δϱ. However, constructing δ𝒫/δϱ
                # locally from the ϱ and 𝒫 grids leads to large
                # numerical errors. Regardless of whether the P=wρ
                # approximation is used or not, we simply use the
                # global sound speed.
                v_rms += light_speed*sqrt(w)/a
        return v_rms
    elif quantity == 'momentum':
        Σmom = empty(3, dtype=C2np['double'])
        σmom = empty(3, dtype=C2np['double'])
        if component.representation == 'particles':
            # Total momentum of all particles, for each dimension
            for dim in range(3):
                mom = component.mom[dim]
                Σmom_dim = Σmom2_dim = 0
                # Add up local particle momenta
                for i in range(component.N_local):
                    mom_i = mom[i]
                    Σmom_dim  += mom_i
                    Σmom2_dim += mom_i**2
                # Add up local particle momenta sums
                if communicate:
                    Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                    Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                # Compute global standard deviation
                σ2mom_dim = Σmom2_dim/N - (Σmom_dim/N)**2
                if σ2mom_dim < 0:
                    # Negative (about -machine_ϵ) σ² can happen due
                    # to round-off errors.
                    σ2mom_dim = 0
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
        elif component.representation == 'fluid':
            # Total momentum of all fluid elements, for each dimension.
            # Here the definition of momenta is chosen as
            # J*Vcell = (a**4*(ρ + c⁻²P))*Vcell
            #         = (V_phys*(ρ + c⁻²P))*a*u,
            # which reduces to mass*a*u for pressureless fluids and so
            # it is in correspondance with the momentum definition
            # for particles.
            for dim, fluidscalar in enumerate(component.J):
                J_noghosts = asarray(fluidscalar.grid_noghosts)
                # Total dim'th momentum of all fluid elements
                Σmom_dim = np.sum(J_noghosts)*Vcell
                # Total dim'th momentum squared of all fluid elements
                Σmom2_dim = np.sum(J_noghosts**2)*Vcell**2
                # Add up local fluid element momenta sums
                if communicate:
                    Σmom_dim  = allreduce(Σmom_dim,  op=MPI.SUM)
                    Σmom2_dim = allreduce(Σmom2_dim, op=MPI.SUM)
                # Compute global standard deviation
                σ2mom_dim = Σmom2_dim/N_elements - (Σmom_dim/N_elements)**2
                if σ2mom_dim < 0:
                    # Negative (about -machine_ϵ) σ² can happen due
                    # to round-off errors.
                    σ2mom_dim = 0
                σmom_dim = sqrt(σ2mom_dim)
                # Pack results
                Σmom[dim] = Σmom_dim
                σmom[dim] = σmom_dim
        return Σmom, σmom
    # Fluid quantities
    elif quantity == 'ϱ':
        # Compute mean(ϱ), std(ϱ), min(ϱ)
        if component.representation == 'particles':
            # Particle components have no ϱ
            abort(
                f'The measure function was called with {component.name} and '
                f'quantity=\'ϱ\', but particle components do not have ϱ'
            )
        elif component.representation == 'fluid':
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_noghosts)
            # Total ϱ² of all fluid elements
            Σϱ2 = np.sum(ϱ_noghosts**2)
            # Add up local sums
            if communicate:
                Σϱ  = allreduce(Σϱ,  op=MPI.SUM)
                Σϱ2 = allreduce(Σϱ2, op=MPI.SUM)
            # Compute mean value of ϱ
            ϱ_bar = Σϱ/N_elements
            # Compute global standard deviation
            σ2ϱ = Σϱ2/N_elements - ϱ_bar**2
            if σ2ϱ < 0:
                # Negative (about -machine_ϵ) σ² can happen due
                # to round-off errors.
                σ2ϱ = 0
            σϱ = sqrt(σ2ϱ)
            # Compute minimum value of ϱ
            ϱ_min = np.min(ϱ_noghosts)
            if communicate:
                ϱ_min = allreduce(ϱ_min, op=MPI.MIN)
        return ϱ_bar, σϱ, ϱ_min
    elif quantity == 'mass':
        if component.representation == 'particles':
            # Any change in the mass of particle a component is absorbed
            # into w_eff(a).
            Σmass = a**(-3*w_eff)*N*component.mass
        elif component.representation == 'fluid':
            # Total ϱ of all fluid elements
            Σϱ = np.sum(ϱ_noghosts)
            # Add up local sums
            if communicate:
                Σϱ = allreduce(Σϱ, op=MPI.SUM)
            # The total mass is
            # Σmass = (a**3*Vcell)*Σρ
            # where a**3*Vcell is the proper volume and Σρ is the sum of
            # proper densities. In terms of the fluid variable
            # ϱ = a**(3*(1 + w_eff))*ρ, the total mass is then
            # Σmass = a**(-3*w_eff)*Vcell*Σϱ.
            # Note that the total mass is generally constant.
            Σmass = a**(-3*w_eff)*Vcell*Σϱ
        return Σmass
    elif quantity == 'discontinuity':
        if component.representation == 'particles':
            # Particle components have no discontinuity
            abort(
                f'The measure function was called with {component.name} and '
                f'quantity=\'discontinuity\', which is not applicable to particle components'
            )
        elif component.representation == 'fluid':
            # Lists to store results which will be returned
            names = []
            Δdiff_max_normalized_list = []
            Δdiff_max_list = []
            # The grid spacing in physical units
            h = boxsize/component.gridsize
            # Find the maximum discontinuity in each fluid grid
            for fluidscalar in component.iterate_fluidscalars():
                # Store the name of the fluid scalar
                names.append(str(fluidscalar))
                # Communicate ghost points of the grid
                communicate_ghosts(fluidscalar.grid_mv, '=')
                # Differentiate the grid in all three directions via
                # both forward and backward difference. For each
                # direction, save the largest difference between
                # the two. Also save the largest differential in
                # each direction.
                Δdiff_max = empty(3, dtype=C2np['double'])
                diff_max = empty(3, dtype=C2np['double'])
                for dim in range(3):
                    # Do the differentiations
                    ᐁgrid_dim = diff_domaingrid(
                        fluidscalar.grid_mv, dim, 1, h, 0, direction='forward',
                    )
                    diff_forward = ᐁgrid_dim[
                        nghosts:(ᐁgrid_dim.shape[0] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[1] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[2] - nghosts),
                    ]
                    ᐁgrid_dim = diff_domaingrid(
                        fluidscalar.grid_mv, dim, 1, h, 1, direction='backward',
                    )
                    diff_backward = ᐁgrid_dim[
                        nghosts:(ᐁgrid_dim.shape[0] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[1] - nghosts),
                        nghosts:(ᐁgrid_dim.shape[2] - nghosts),
                    ]
                    # Find the largest difference between the results of
                    # the forward and backward difference,
                    Δdiff_max_dim = 0
                    diff_max_dim = 0
                    for         i in range(ℤ[ϱ_noghosts.shape[0]]):
                        for     j in range(ℤ[ϱ_noghosts.shape[1]]):
                            for k in range(ℤ[ϱ_noghosts.shape[2]]):
                                # The maximum difference of the two differentials
                                Δdiff = abs(diff_forward[i, j, k] - diff_backward[i, j, k])
                                if Δdiff > Δdiff_max_dim:
                                    Δdiff_max_dim = Δdiff
                                # The maximum differential
                                diff_size = abs(diff_forward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                                diff_size = abs(diff_backward[i, j, k])
                                if diff_size > diff_max_dim:
                                    diff_max_dim = diff_size
                    # Use the global maxima
                    if communicate:
                        Δdiff_max_dim = allreduce(Δdiff_max_dim, op=MPI.MAX)
                        diff_max_dim  = allreduce(diff_max_dim,  op=MPI.MAX)
                    # Pack results into lists
                    Δdiff_max[dim] = Δdiff_max_dim
                    diff_max[dim] = diff_max_dim
                Δdiff_max_list.append(Δdiff_max)
                # Maximum discontinuity (difference between forward and
                # backward difference) normalized accoring to
                # the largest slope.
                Δdiff_max_normalized_list.append(np.array([Δdiff_max[dim]/diff_max[dim]
                                                           if Δdiff_max[dim] > 0 else 0
                                                           for dim in range(3)
                                                           ], dtype=C2np['double'],
                                                          )
                                                 )
        return names, Δdiff_max_list, Δdiff_max_normalized_list
    elif master:
        abort(
            f'The measure function was called with '
            f'quantity=\'{quantity}\', which is not implemented'
        )

# Function for doing debugging analysis
@cython.header(# Arguments
               components=list,
               # Locals
               component='Component',
               dim='int',
               name=str,
               w_eff='double',
               Δdiff_max='double[::1]',
               Δdiff_max_normalized='double[::1]',
               Σmom='double[::1]',
               Σmom_prev_dim='double',
               ϱ_bar='double',
               ϱ_min='double',
               σmom='double[::1]',
               σϱ='double',
               )
def debug(components):
    """This function will compute many different quantities from the
    component data and print out the results. Warnings will be given for
    obviously erroneous results.
    """
    # Componentwise analysis
    for component in components:
        w_eff = component.w_eff()
        # sum(momentum) and std(momentum) in each dimension
        Σmom, σmom = measure(component, 'momentum')
        for dim in range(3):
            debug_print('total {}-momentum'.format('xyz'[dim]),
                        component,
                        Σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
            debug_print('standard deviation of {}-momentum'.format('xyz'[dim]),
                        component,
                        σmom[dim],
                        'm☉ Mpc Gyr⁻¹',
                        )
        # Warn if sum(momentum) does not agree with previous measurement
        if component.name in Σmom_prev:
            for dim in range(3):
                Σmom_prev_dim = Σmom_prev[component.name][dim]
                if not isclose(Σmom_prev_dim, Σmom[dim],
                               rel_tol=1e-6,
                               abs_tol=1e-6*σmom[dim],
                               ):
                    masterwarn(
                        'Previously {} had a total {}-momentum of {} m☉ Mpc Gyr⁻¹'
                        .format(
                            component.name,
                            'xyz'[dim],
                            significant_figures(
                                Σmom_prev_dim/(units.m_sun*units.Mpc/units.Gyr),
                                12,
                                fmt='unicode',
                                incl_zeros=False,
                                scientific=True,
                            ),
                        )
                    )
        Σmom_prev[component.name] = asarray(Σmom).copy()
        # mean(ϱ), std(ϱ) and min(ϱ)
        if component.representation == 'fluid':
            ϱ_bar, σϱ, ϱ_min = measure(component, 'ϱ')
            debug_print('mean ϱ',
                        component,
                        ϱ_bar,
                        'm☉ Mpc⁻³',
                        )
            debug_print('standard deviation of ϱ',
                        component,
                        σϱ,
                        'm☉ Mpc⁻³',
                        )
            debug_print('minimum ϱ',
                        component,
                        ϱ_min,
                        'm☉ Mpc⁻³',
                        )
            # Warn if any densities are negative
            if ϱ_min < 0:
                masterwarn(f'Negative density occured for {component.name}')
            # Warn if mean(ϱ) differs from the correct, constant result
            if not isclose(ϱ_bar, cast(component.ϱ_bar, 'double'), rel_tol=1e-6):
                masterwarn(
                    '{} ought to have a mean ϱ of {} m☉ Mpc⁻³'
                    .format(
                        component.name.capitalize(),
                        significant_figures(
                            component.ϱ_bar/(units.m_sun/units.Mpc**3),
                            12,
                            fmt='unicode',
                            incl_zeros=False,
                            scientific=True,
                        ),
                    )
                )
        # The maximum discontinuities in the fluid scalars,
        # for each dimension. Here, a discontinuity means a difference
        # in forward and backward difference.
        if component.representation == 'fluid':
            for name, Δdiff_max, Δdiff_max_normalized in zip(*measure(component, 'discontinuity')):
                for dim in range(3):
                    debug_print('maximum            {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max[dim],
                                'Mpc⁻¹',
                                )
                    debug_print('maximum normalized {}-discontinuity in {}'.format('xyz'[dim], name),
                                component,
                                Δdiff_max_normalized[dim],
                                )
# Dict storing sum of momenta for optained in previous call to the
# debug function, for all components.
cython.declare(Σmom_prev=dict)
Σmom_prev = {}

# Function for printing out debugging info,
# used in the debug function above.
@cython.header(# Arguments
               quantity=str,
               component='Component',
               value='double',
               unit_str=str,
               # Locals
               text=str,
               unit='double',
               value_str=str,
               )
def debug_print(quantity, component, value, unit_str='1'):
    unit = eval_unit(unit_str)
    value_str = significant_figures(value/unit,
                                    12,
                                    fmt='unicode',
                                    incl_zeros=False,
                                    scientific=True,
                                    )
    text = '{} {}({}) = {}{}'.format(terminal.bold_cyan('Debug info:'),
                                     quantity[0].upper() + quantity[1:],
                                     component.name,
                                     value_str,
                                     ' ' + unit_str if unit_str != '1' else '',
                                     )
    masterprint(text)
