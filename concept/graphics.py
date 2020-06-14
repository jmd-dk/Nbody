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
    '    domain_layout_local_indices, '
    '    domain_size_x,               '
    '    domain_size_y,               '
    '    domain_size_z,               '
    '    domain_start_x,              '
    '    domain_start_y,              '
    '    domain_start_z,              '
    '    get_buffer,                  '
)
cimport('from mesh import interpolate_components, interpolate_grid_to_grid')

# Pure Python imports
from mpl_toolkits.mplot3d import proj3d  # Importing from mpl_toolkits.mplot3d enables 3D plotting



# Function for plotting an already computed power spectrum
# and saving an image file to disk.
@cython.header(
    # Arguments
    powerspec_declaration=object,  # PowerspecDeclaration or list
    filename=str,
    # Locals
    a_str=str,
    component='Component',
    components=list,
    components_str=str,
    k_bin_centers='double[::1]',
    power='double[::1]',
    power_linear='double[::1]',
    powerspec_declarations=list,
    t_str=str,
    returns='void',
)
def plot_powerspec(powerspec_declaration, filename):
    """It is expected that this function
    is called by the master process only.
    """
    # Recursive dispatch
    if isinstance(powerspec_declaration, list):
        powerspec_declarations = powerspec_declaration
    else:
        powerspec_declarations = [powerspec_declaration]
    powerspec_declarations = [
        powerspec_declaration
        for powerspec_declaration in powerspec_declarations
        if powerspec_declaration.do_plot
    ]
    if not powerspec_declarations:
        return
    if len(powerspec_declarations) > 1:
        for powerspec_declaration in powerspec_declarations:
            # Since we have multiple plots --- one for each
            # set of components --- we augment each filename
            # with this information.
            plot_powerspec(
                powerspec_declaration,
                augment_filename(
                    filename,
                    '_'.join([
                        component.name.replace(' ', '-')
                        for component in powerspec_declaration.components
                    ]),
                    '.png',
                )
            )
        return
    powerspec_declaration = powerspec_declarations[0]
    # Ensure correct filename extension
    if not filename.endswith('.png'):
        filename += '.png'
    # Extract variables
    components    = powerspec_declaration.components
    k_bin_centers = powerspec_declaration.k_bin_centers
    power         = powerspec_declaration.power
    power_linear  = powerspec_declaration.power_linear
    # Begin progress message
    if len(components) == 1:
        components_str = components[0].name
    else:
        components_str = '{{{}}}'.format(
            ', '.join([component.name for component in components])
        )
    masterprint(
        f'Plotting power spectrum of {components_str} and saving to "{filename}" ...'
    )
    # Plot power spectrum in new figure
    fig, ax = plt.subplots()
    if np.any(power):
        ax.loglog(k_bin_centers, power, '-', label='simulation')
    else:
        # The odd case of no power at all
        ax.semilogx(k_bin_centers, power, '-', label='simulation')
    # Also plot linear CLASS power spectra, if specified
    if power_linear is not None:
        ylim = ax.get_ylim()
        if np.any(power_linear) and np.any(~np.isnan(power_linear)):
            ax.loglog(k_bin_centers, power_linear, 'k--', label='linear')
        else:
            # The odd case of no power at all
            ax.semilogx(k_bin_centers, power_linear, 'k--', label='linear')
        # Labels are only needed when both the non-linear (simulation)
        # and linear spectrum are plotted.
        ax.legend(fontsize=14)
        ax.set_ylim(ylim)
    ax.set_xlabel(rf'$k$ $[\mathrm{{{unit_length}}}^{{-1}}]$', fontsize=14)
    ax.set_ylabel(rf'power $[\mathrm{{{unit_length}}}^3]$',    fontsize=14)
    t_str = (
        rf'$t = {{}}\, \mathrm{{{{{unit_time}}}}}$'
        .format(significant_figures(universals.t, 4, fmt='tex'))
    )
    a_str = ''
    if enable_Hubble:
        a_str = ', $a = {}$'.format(significant_figures(universals.a, 4, fmt='tex'))
    components_str = (
        components_str
        .replace('{', r'$\{$')
        .replace('}', r'$\}$')
    )
    ax.set_title(f'{components_str}\nat {t_str}{a_str}', fontsize=16, horizontalalignment='center')
    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.tick_params(axis='both', which='minor', labelsize=11)
    plt.tight_layout()
    plt.savefig(filename)
    # Done with this plot.
    # Close the figure, leaving no trace in memory of the plot.
    plt.close(fig)
    masterprint('done')

# Function for plotting detrended CLASS perturbations
@cython.pheader(
    # Arguments
    k='Py_ssize_t',
    k_magnitude='double',
    transferfunction_info=object,  # TransferFunctionInfo
    class_species=str,
    factors='double[::1]',
    exponents='double[::1]',
    splines=object,  # np.ndarray of dtype object
    largest_trusted_k_magnitude='double',
    crossover='int',
    # Locals
    a_values_raw=object,  # np.ndarray
    exponent='double',
    exponent_str=str,
    factor='double',
    factor_str=str,
    filename=str,
    i='Py_ssize_t',
    k_str=str,
    key=str,
    loga_value='double',
    loga_values='double[::1]',
    loga_values_spline='double[::1]',
    n='Py_ssize_t',
    perturbations_detrended_spline='double[::1]',
    perturbations_raw=object,  # np.ndarray
    skip='Py_ssize_t',
    spline='Spline',
    val=str,
)
def plot_detrended_perturbations(k, k_magnitude, transferfunction_info, class_species,
    factors, exponents, splines, largest_trusted_k_magnitude, crossover):
    # All processes could carry out this work, but as it involves I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_detrended_perturbations()')
    n_subplots = 0
    for spline in splines:
        if spline is None:
            break
        n_subplots += 1
    fig, axes = plt.subplots(1, n_subplots, figsize=(6*n_subplots + 0.4, 4.8))
    axes = any2list(axes)
    k_str = significant_figures(k_magnitude, 3, fmt='tex', scientific=True)
    fig.suptitle(
        ('' if transferfunction_info.total else rf'{class_species}, ')
        + rf'$k = {k_str}\, \mathrm{{{unit_length}}}^{{-1}}$',
        fontsize=16,
        horizontalalignment='center',
    )
    for n, ax in enumerate(axes):
        factor, exponent, spline = factors[n], exponents[n], splines[n]
        a_values, perturbations_detrended = spline.x, spline.y
        index_left = 0
        if n != 0:
            index_left += crossover
        index_right = a_values.shape[0]
        if n != n_subplots - 1:
            index_right -= crossover
        a_values = a_values[index_left:index_right]
        a_min = significant_figures(a_values[0], 4, fmt='tex', scientific=True)
        a_max = significant_figures(a_values[a_values.shape[0] - 1], 4, fmt='tex', scientific=True)
        perturbations_detrended = perturbations_detrended[index_left:index_right]
        # Plot the detrended CLASS data
        ax.semilogx(a_values, perturbations_detrended, '.', markersize=3)
        # Plot the spline at values midway between the data points
        loga_values = np.log(a_values)
        loga_values_spline             = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
        perturbations_detrended_spline = empty(loga_values.shape[0] - 1, dtype=C2np['double'])
        skip = 0
        for i in range(loga_values_spline.shape[0]):
            loga_value = 0.5*(loga_values[i] + loga_values[i+1])
            if not (ℝ[spline.xmin] <= loga_value <= ℝ[spline.xmax]):
                skip += 1
                continue
            loga_values_spline[ℤ[i - skip]] = loga_value
            perturbations_detrended_spline[ℤ[i - skip]] = spline.eval(exp(loga_value))
        loga_values_spline = loga_values_spline[:ℤ[i - skip + 1]]
        perturbations_detrended_spline = perturbations_detrended_spline[:ℤ[i - skip + 1]]
        ax.semilogx(np.exp(loga_values_spline), perturbations_detrended_spline, '-',
            linewidth=1, zorder=0)
        ax.set_xlim(a_values[0], a_values[a_values.shape[0] - 1])
        # Decorate plot
        if n == 0:
            ax.set_ylabel(
                rf'$({transferfunction_info.name_latex} - \mathrm{{trend}})\, '
                rf'[{transferfunction_info.units_latex}]$'
                if transferfunction_info.units_latex else
                rf'${transferfunction_info.name_latex} - \mathrm{{trend}}$',
                fontsize=14,
            )
        ax.set_xlabel(rf'$a \in [{a_min}, {a_max}]$', fontsize=14)
        factor_str = significant_figures(factor, 6, fmt='tex', scientific=True)
        exponent_str = significant_figures(exponent, 6, scientific=False)
        trend_str = (
            rf'$\mathrm{{trend}} = 0$'
            if factor == 0 else
            rf'$\mathrm{{trend}} = {factor_str}'
            rf'{transferfunction_info.units_latex}a^{{{exponent_str}}}$'
        )
        ax.text(0.5, 0.8,
            trend_str,
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes,
            fontsize=14,
        )
        if k_magnitude > largest_trusted_k_magnitude:
            ax.text(0.5, 0.65,
                rf'(using data from $k = {largest_trusted_k_magnitude}\, '
                rf'\mathrm{{{unit_length}}}^{{-1}}$)',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=14,
            )
    # Finalise and save plot
    fig.subplots_adjust(wspace=0, hspace=0)
    filename = '/'.join([
        output_dirs['powerspec'],
        'class_perturbations',
        transferfunction_info.name_ascii.format(class_species),
    ])
    os.makedirs(filename, exist_ok=True)
    filename += f'/{k}.png'
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.1)
    plt.close()

# Function for plotting processed CLASS perturbations
@cython.pheader(
    # Arguments
    a_values='double[::1]',
    k_magnitudes='double[::1]',
    transfer='double[:, ::1]',
    transferfunction_info=object,  # TransferFunctionInfo
    class_species=str,
    n_plots_in_figure='Py_ssize_t',
    # Locals
    a='double',
    dirname=str,
    i='Py_ssize_t',
    i_figure='Py_ssize_t',
    key=str,
    nfigs='Py_ssize_t',
    val=str,
)
def plot_processed_perturbations(
    a_values, k_magnitudes, transfer, transferfunction_info, class_species,
    n_plots_in_figure=10,
):
    """The 2D transfer array is the tabulated transfer function values,
    indexed as transfer[a, k], with the values of a and k given by
    a_values and k_magnitudes.
    """
    # All processes could carry out this work, but as it involved I/O,
    # we only allow the master process to do so.
    if not master:
        abort(f'rank {rank} called plot_processed_perturbations()')
    if transferfunction_info.total:
        masterprint(f'Plotting processed {transferfunction_info.name} transfer functions ...')
    else:
        masterprint(
            f'Plotting processed {transferfunction_info.name} {class_species} '
            f'transfer functions ...'
        )
    dirname = '/'.join([
        output_dirs['powerspec'],
        'class_perturbations_processed',
        transferfunction_info.name_ascii.format(class_species),
    ])
    os.makedirs(dirname, exist_ok=True)
    nfigs = int(log10(a_values.shape[0])) + 1
    i_figure = 0
    plt.figure()
    for i in range(a_values.shape[0]):
        a = a_values[i]
        plt.semilogx(k_magnitudes, transfer[i, :],
            label='$a={}$'.format(significant_figures(a, nfigs, fmt='tex')))
        if ((i + 1)%n_plots_in_figure == 0) or i == ℤ[a_values.shape[0] - 1]:
            plt.legend()
            plt.xlabel(rf'$k\,[\mathrm{{{unit_length}}}^{{-1}}]$', fontsize=14)
            plt.ylabel(
                rf'${transferfunction_info.name_latex}\, [{transferfunction_info.units_latex}]$'
                if transferfunction_info.units_latex else
                rf'${transferfunction_info.name_latex}$',
                fontsize=14,
            )
            if not transferfunction_info.total:
                plt.title(
                    class_species,
                    fontsize=16,
                    horizontalalignment='center',
                )
            plt.gca().tick_params(axis='x', which='major', labelsize=13)
            plt.tight_layout()
            plt.savefig(f'{dirname}/{i_figure}.png')
            i_figure += 1
            plt.cla()
    plt.close()
    masterprint('done')

# This function produces 2D renders of the density fields of single
# and sets of components.
@cython.header(
    # Arguments
    components=list,
    filename=str,
    # Locals
    axis=str,
    buffer_number='int',
    colormap=str,
    colornumber='int',
    component_combination=tuple,
    component_combination_str=str,
    component_combinations=object,  # generator
    exponent='double',
    ext=str,
    extent=tuple,
    filename_combination=str,
    grid='double[:, :, ::1]',
    grid_fluid='double[:, :, ::1]',
    grid_fluid_ptr='double*',
    grid_particles='double[:, :, ::1]',
    grid_particles_ptr='double*',
    grid_terminal='double[:, :, ::1]',
    grids=dict,
    gridsize='Py_ssize_t',
    gridsize_component='Py_ssize_t',
    gridsize_terminal='Py_ssize_t',
    i='Py_ssize_t',
    interpolation_order='int',
    j='Py_ssize_t',
    n_data_outputs='Py_ssize_t',
    n_image_outputs='Py_ssize_t',
    names_str=str,
    projection='double[:, ::1]',
    projection_terminal='double[:, ::1]',
    row='double[::1]',
    terminal_projection_ansi=list,
    value='double',
    vmax='double',
    vmax_terminal='double',
    vmin='double',
    vmin_terminal='double',
)
def render2D(components, filename):
    """This function will produce 2D renders of the passed components.
    A slab of the density field will be projected onto a plane.
    The details of this projection is specified in the render2D_options
    user parameter.
    """
    # Remove any extension on the filename
    for ext in ('.hdf5', '.png'):
        if filename.endswith(ext):
            filename = filename[:len(filename) - len(ext)]
            break
    # Generator yielding tuples of all possible combinations
    # of the passed components.
    component_combinations = itertools.chain.from_iterable(
        [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
    )
    # Count the number of output files
    n_data_outputs = n_image_outputs = 0
    for component_combination in component_combinations:
        if is_selected(component_combination, render2D_data_select):
            n_data_outputs += 1
        if is_selected(component_combination, render2D_image_select):
            n_image_outputs += 1
    # Rebuild the generator
    component_combinations = itertools.chain.from_iterable(
        [itertools.combinations(components, i) for i in range(1, len(components) + 1)]
    )
    # Produce 2D renders of each combination of components,
    # if they have been selected for in the render2D_select parameter.
    for component_combination in component_combinations:
        if not (
                is_selected(component_combination, render2D_data_select)
            or  is_selected(component_combination, render2D_image_select)
            or  is_selected(component_combination, render2D_terminal_image_select)
        ):
            continue
        component_combination_str = ', '.join(
            [component.name for component in component_combination]
        )
        if len(component_combination) > 1:
            component_combination_str = f'{{{component_combination_str}}}'
        masterprint(f'Rendering 2D projection of {component_combination_str} ...')
        # Get the gridsize of the interpolation grid. If none is set,
        # choose a gridsize based on the component with the largest
        # number of particles or fluid elements.
        gridsize = is_selected(component_combination, render2D_options['gridsize'])
        if gridsize is None or gridsize == -1:
            gridsize = -1
            for component in component_combination:
                if component.representation == 'particles':
                    gridsize_component = int(round(cbrt(component.N)))
                elif component.representation == 'fluid':
                    gridsize_component = component.gridsize
                if gridsize_component > gridsize:
                    gridsize = gridsize_component
            if gridsize < 2*nghosts:
                gridsize = 2*nghosts
        # We now do the interpolation of the components onto grids.
        # A separate grid will be used for particles and fluids.
        # We choose to interpolate the physical density ρ.
        interpolation_order = is_selected(
            component_combination,
            render2D_options['interpolation'],
        )
        grids = interpolate_components(
            list(component_combination),
            'ρ',
            gridsize,
            interpolation_order,
        )
        # Sum the grids into a single grid storing the total density
        grid_particles = grids['particles']
        grid_fluid     = grids['fluid']
        if grid_particles is not None and grid_fluid is not None:
            grid_particles_ptr = cython.address(grid_particles[:, :, :])
            grid_fluid_ptr     = cython.address(grid_fluid    [:, :, :])
            grid = grid_particles
            for i in range(grid.shape[0]*grid.shape[1]*grid.shape[2]):
                grid_particles_ptr[i] += grid_fluid_ptr[i]
        elif grid_particles is not None:
            grid = grid_particles
        else:  # grid_fluid is not None
            grid = grid_fluid
        # Get projected 2D grid
        axis   = is_selected(component_combination, render2D_options['axis'])
        extent = is_selected(component_combination, render2D_options['extent'])
        buffer_number = 0
        projection = project(grid, gridsize, axis, extent, buffer_number)
        buffer_number += 1
        # If we additionally need a terminal image, interpolate the
        # grid onto another grid of the needed size
        # and produce a projection.
        projection_terminal = None
        if is_selected(component_combination, render2D_terminal_image_select):
            gridsize_terminal = is_selected(
                component_combination,
                render2D_options['terminal resolution'],
            )
            grid_terminal = interpolate_grid_to_grid(grid, buffer_number, gridsize_terminal)
            buffer_number += 1
            projection_terminal = project(
                grid_terminal, gridsize_terminal, axis, extent, buffer_number,
            )
            buffer_number += 1
            # Since each monospaced character cell in the terminal is
            # rectangular with about double the height compared to the
            # width, the terminal projection should only have half as
            # many rows as it has columns.
            for i in range(projection_terminal.shape[0]//2):
                row = 0.5*(
                    asarray(projection_terminal[2*i, :]) + asarray(projection_terminal[2*i + 1, :])
                )
                projection_terminal[i, :] = row
            projection_terminal = projection_terminal[:projection_terminal.shape[0]//2, :]
        # The master now holds all needed information
        if not master:
            continue
        # Store projected image as an hdf5 file
        if is_selected(component_combination, render2D_data_select):
            # The filename should reflect the component combination
            filename_combination = filename
            if not filename_combination.endswith('.hdf5'):
                filename_combination += '.hdf5'
            if n_data_outputs > 1:
                filename_combination = augment_filename(
                    filename_combination,
                    '_'.join([
                        component.name.replace(' ', '-')
                        for component in component_combination
                    ]),
                    '.hdf5',
                )
            masterprint(f'Saving data to "{filename_combination}" ...')
            with open_hdf5(filename_combination, mode='w') as hdf5_file:
                # Save used base unit
                hdf5_file.attrs['unit time'  ] = unit_time
                hdf5_file.attrs['unit length'] = unit_length
                hdf5_file.attrs['unit mass'  ] = unit_mass
                # Save attributes
                hdf5_file.attrs['boxsize'              ] = boxsize
                hdf5_file.attrs['component combination'] = component_combination_str
                hdf5_file.attrs['axis'                 ] = axis
                hdf5_file.attrs['extent'               ] = extent
                if enable_Hubble:
                    hdf5_file.attrs['a'] = universals.a
                hdf5_file.attrs['t'    ] = universals.t
                # Store the 2D projection
                dset = hdf5_file.create_dataset(
                    'data',
                    asarray(projection).shape,
                    dtype=C2np['double']
                )
                dset[...] = projection
            masterprint('done')
        # If no colorized image should be produced from the projection,
        # skip the following.
        if not (
               is_selected(component_combination, render2D_image_select)
            or is_selected(component_combination, render2D_terminal_image_select)
        ):
            masterprint('done')
            continue
        # The colormap specified for this component combination
        colormap = is_selected(component_combination, render2D_options['colormap'])
        # Enhance projections if specified
        vmin_terminal = vmin_terminal = 0
        if is_selected(component_combination, render2D_options['enhance']):
            projection, vmin, vmax, exponent = enhance(projection)
            if projection_terminal is not None:
                projection_terminal, vmin_terminal, vmax_terminal, exponent = enhance(
                    projection_terminal, exponent,
                )
        else:
            vmin = np.min(projection)
            vmax = np.max(projection)
            if projection_terminal is not None:
                vmin_terminal = np.min(projection_terminal)
                vmax_terminal = np.max(projection_terminal)
        # If vmin and vmax are exactly the same, it is because the
        # projection is completely homogeneous, with vmin and vmax equal
        # to the common pixel value. In this case, change vmin and vmax
        # such that the homogeneous value sits right between
        # vmin and vmax.
        if vmin == vmax:
            vmin = 0
            vmax *= 2
        if projection_terminal is not None:
            if vmin_terminal == vmax_terminal:
                vmin_terminal = 0
                vmax_terminal *= 2
        # Draw projected image in the terminal
        if projection_terminal is not None:
            # Apply the colormap
            set_terminal_colormap(colormap)
            # Construct list of strings, each string being a space
            # prepended with an ANSI/VT100 control sequences which sets
            # the background color. When printed together, these strings
            # produce an ANSI image of the terminal projection.
            # We need to map the values between vmin and vmax to
            # the 238 higher integer color numbers 18–255 (the lowest 18
            # color numbers are already occupied).
            terminal_projection_ansi = []
            for     i in range(ℤ[projection_terminal.shape[0]]):
                for j in range(ℤ[projection_terminal.shape[1]]):
                    value = projection_terminal[i, j]
                    if value > vmax_terminal:
                        value = vmax_terminal
                    elif value < vmin_terminal:
                        value = vmin_terminal
                    colornumber = 18 + cast(
                        round((value - vmin_terminal)*ℝ[237/(vmax_terminal - vmin_terminal)]),
                        'int',
                    )
                    # Insert a space with colored background
                    terminal_projection_ansi.append(esc_background.format(colornumber) + ' ')
                # Insert newline with no background color
                terminal_projection_ansi.append(f'{esc_normal}\n')
            # Print the ANSI image to the terminal
            masterprint(''.join(terminal_projection_ansi), end='', indent=-1, wrap=False)
        # Save colorized image to disk
        if is_selected(component_combination, render2D_image_select):
            # The filename should reflect the component combination
            filename_combination = filename + '.png'
            if n_image_outputs > 1:
                filename_combination = augment_filename(
                    filename_combination,
                    '_'.join([
                        component.name.replace(' ', '-')
                        for component in component_combination
                    ]),
                    '.png',
                )
            masterprint(f'Saving image to "{filename_combination}" ...')
            plt.imsave(
                filename_combination,
                asarray(projection),
                cmap=colormap,
                vmin=vmin,
                vmax=vmax,
            )
            masterprint('done')
        # Done with the entire rendering process
        # for this component combination.
        masterprint('done')
# Construct the render2D_data_select, render2D_image_select
# and render2D_terminal_image_select dicts from
# the render2D_select parameter.
cython.declare(
    render2D_data_select=dict,
    render2D_image_select=dict,
    render2D_terminal_image_select=dict,
)
render2D_data_select = {
    key: val['data'] for key, val in render2D_select.items()
}
render2D_image_select = {
    key: val['image'] for key, val in render2D_select.items()
}
render2D_terminal_image_select = {
    key: val['terminal image'] for key, val in render2D_select.items()
}

# Function for converting distributed 3D domain grids
# into a 2D projection grid.
@cython.header(
    # Arguments
    grid='double[:, :, ::1]',
    gridsize='Py_ssize_t',
    axis=str,
    extent=tuple,
    buffer_name=object,  # str or int
    # Locals
    cellsize='double',
    dim='int',
    domain_start_indices='Py_ssize_t[::1]',
    gridshape_local='Py_ssize_t[::1]',
    indices_2D_bgn='Py_ssize_t[::1]',
    indices_2D_end='Py_ssize_t[::1]',
    indices_global_bgn='Py_ssize_t[::1]',
    indices_global_end='Py_ssize_t[::1]',
    indices_local_bgn='Py_ssize_t[::1]',
    indices_local_end='Py_ssize_t[::1]',
    participate='bint',
    projection=object,  # np.ndarray
    returns='double[:, ::1]',
)
def project(grid, gridsize, axis, extent, buffer_name=0):
    """Note that only the master will return the full projection.
    """
    # Get global index range into the grids, specifying the chunk
    # that should be used for the projection.
    indices_global_bgn = asarray([0       ]*3, dtype=C2np['Py_ssize_t'])
    indices_global_end = asarray([gridsize]*3, dtype=C2np['Py_ssize_t'])
    for dim in range(3):
        if axis == 'xyz'[dim]:
            indices_global_bgn[dim] = int(round(extent[0]*gridsize/boxsize))
            indices_global_end[dim] = int(round(extent[1]*gridsize/boxsize))
            break
    # Convert the global indices to local indices,
    # disregarding ghost points, for now.
    cellsize = boxsize/gridsize
    domain_start_indices = asarray(
        [
            int(round(domain_start_x/cellsize)),
            int(round(domain_start_y/cellsize)),
            int(round(domain_start_z/cellsize)),
        ],
        dtype=C2np['Py_ssize_t'],
    )
    gridshape_local = asarray(
        asarray(asarray(grid).shape) - ℤ[2*nghosts],
        dtype=C2np['Py_ssize_t'],
    )
    participate = True
    indices_local_bgn = asarray(indices_global_bgn) - asarray(domain_start_indices)
    for dim in range(3):
        if indices_local_bgn[dim] < 0:
            indices_local_bgn[dim] = 0
        elif indices_local_bgn[dim] > gridshape_local[dim]:
            participate = False
            break
    indices_local_end = asarray(indices_global_end) - asarray(domain_start_indices)
    for dim in range(3):
        if indices_local_end[dim] < 0:
            participate = False
            break
        elif indices_local_end[dim] > gridshape_local[dim]:
            indices_local_end[dim] = gridshape_local[dim]
    for dim in range(3):
        if indices_local_bgn[dim] == indices_local_end[dim]:
            participate = False
            break
    # Redefine the global indices so that they correspond to the
    # local chunk, but indexing into a global grid.
    indices_global_bgn = asarray(indices_local_bgn) + asarray(domain_start_indices)
    indices_global_end = asarray(indices_local_end) + asarray(domain_start_indices)
    # Instantiate full 2D projection array on every process
    projection = get_buffer((gridsize, )*2, buffer_name, nullify=True)
    # Fill in the local part of the projection on each process
    if participate:
        if axis == 'x':
            # The projection will be onto the yz plane
            # with y right and z up.
            indices_2D_bgn = asarray([indices_global_bgn[1], indices_global_bgn[2]],
                dtype=C2np['Py_ssize_t'])
            indices_2D_end = asarray([indices_global_end[1], indices_global_end[2]],
                dtype=C2np['Py_ssize_t'])
        elif axis == 'y':
            # The projection will be onto the xz plane
            # with x right and z up.
            indices_2D_bgn = asarray([indices_global_bgn[0], indices_global_bgn[2]],
                dtype=C2np['Py_ssize_t'])
            indices_2D_end = asarray([indices_global_end[0], indices_global_end[2]],
                dtype=C2np['Py_ssize_t'])
        elif axis == 'z':
            # The projection will be onto the xy plane
            # with x right and y up.
            indices_2D_bgn = asarray([indices_global_bgn[0], indices_global_bgn[1]],
                dtype=C2np['Py_ssize_t'])
            indices_2D_end = asarray([indices_global_end[0], indices_global_end[1]],
                dtype=C2np['Py_ssize_t'])
        projection[
            indices_2D_bgn[0]:indices_2D_end[0],
            indices_2D_bgn[1]:indices_2D_end[1],
        ] += np.sum(grid[
            nghosts + indices_local_bgn[0]:nghosts + indices_local_end[0],
            nghosts + indices_local_bgn[1]:nghosts + indices_local_end[1],
            nghosts + indices_local_bgn[2]:nghosts + indices_local_end[2],
        ], 'xyz'.index(axis))
    # Sum up contributions from all processes into the master,
    # after which the master process holds the full projection.
    Reduce(
        sendbuf=(MPI.IN_PLACE if master else projection),
        recvbuf=(projection   if master else None),
        op=MPI.SUM,
    )
    # Transform the layout of the projection such that the first
    # dimension (rows) correspond to the upward/downward direction and
    # the second dimension (columns) correspond to the left/right
    # direction. Also flip the upward/downward axis by flipping the
    # rows. Together, this put the projection into the proper state for
    # saving it as an image.
    if master:
        projection = np.ascontiguousarray(projection.transpose()[::-1, :])
    return projection

# Function for enhancing the contrast of a 2D image
@cython.header(
    # Arguments
    image='double[:, ::1]',
    exponent='double',
    # Locals
    N_pixels='Py_ssize_t',
    N_bins='Py_ssize_t',
    bin_edges='double[::1]',
    bins='Py_ssize_t[::1]',
    color_truncation_factor_lower='double',
    color_truncation_factor_upper='double',
    exponent_lower='double',
    exponent_max='double',
    exponent_min='double',
    exponent_upper='double',
    i='Py_ssize_t',
    i_center='Py_ssize_t',
    i_max='Py_ssize_t',
    i_min='Py_ssize_t',
    image_enhanced='double[:, ::1]',
    image_max='double',
    image_min='double',
    occupation='Py_ssize_t',
    shifting_factor='double',
    vmax='double',
    vmin='double',
    Σbins='Py_ssize_t',
    returns=tuple,
)
def enhance(image, exponent=0):
    """This function enhances an image by applying a non-linear
    transformation of the form
    image → image**exponent.
    If not provided, we find a value for the exponent which leads to a
    nice distribution of the values in the image. We take this to be the
    case when the histogram of these values is "centered" at the value
    specified by the shifting_factor parameter. A shifting_factor of 0.5
    implies that the histogram of the pixel values is "centered" in the
    middle of the axis, with the same distance to the first and last
    bin. For Gaussian data, this require a value of the exponent tending
    to 0. Thus, the shifting factor should be below 0.5. A
    shifting_factor between 0 and 0.5 shifts the center of the histogram
    to be at the location of shifting_factor, measured relative to the
    histogram axis. Here, the center is defined to be the point which
    partitions the histogram into two parts which integrate to the
    same value.
    """
    # Enforce all pixel values to be between 0 and 1
    image_min = np.min(image)
    if image_min != 0:
        image = asarray(image) - image_min
    image_max = np.max(image)
    if image_max not in (0, 1):
        image = asarray(image)*(1/image_max)
    # Find exponent if not given
    N_pixels = np.prod(asarray(image).shape)
    N_bins = np.max([25, N_pixels//100])
    if exponent == 0:
        shifting_factor = 0.28
        # Find a good value for the exponent using a binary search
        exponent_min = 1e-2
        exponent_max = 1e+2
        exponent_lower = exponent_min
        exponent_upper = exponent_max
        exponent = 1
        i_min = -4
        i_max = -2
        while True:
            # Construct histogram over image**exponent
            image_enhanced = asarray(image)**exponent
            bins, bin_edges = np.histogram(image_enhanced, N_bins)
            # Compute the sum of all bins. This is equal to the sum of
            # values in the image. However, we skip bins[0] since
            # sometimes empty cells results in a large spike there.
            Σbins = N_pixels - bins[0]
            # Find the position of the center of the histogram,
            # defined by the sums of bins being the same on both
            # sides of this center. We again skip bins[0].
            occupation = 0
            for i in range(1, N_bins):
                occupation += bins[i]
                if occupation >= ℤ[Σbins//2]:
                    i_center = i
                    break
            else:
                # Something went wrong. Bail out.
                masterwarn('Something went wrong during image enhancement')
                vmin = np.min(image)
                vmax = np.max(image)
                return image, vmin, vmax, 0
            if i_center < ℤ[N_bins*shifting_factor]:
                # The exponent should be decreased
                exponent_upper = exponent
                i_min = i_center
            elif i_center > ℤ[N_bins*shifting_factor]:
                # The exponent should be increased
                exponent_lower = exponent
                i_max = i_center
            else:
                # Good choice of exponent found
                break
            # The current value of the exponent does not place the
            # "center" of the histogram at the desired location
            # specified by shifting_factor.
            # Check if the the binary seach has (almost) converged on
            # some other value.
            if i_max >= i_min and i_max - i_min <= 1:
                break
            # Check if the exponent is very close
            # to one of the extreme values.
            if exponent/exponent_min < 1.001:
                exponent = exponent_min
                break
            elif exponent_max/exponent < 1.001:
                exponent = exponent_max
                break
            # Update the exponent. As the range of the exponent is
            # large, the binary step is done in logarithmic space.
            exponent = sqrt(exponent_lower*exponent_upper)
    # Apply the image enhancement
    image_enhanced = asarray(image)**exponent
    bins, bin_edges = np.histogram(image_enhanced, N_bins)
    Σbins = N_pixels - bins[0]
    # To further enhance the image, we set the color limits so as to
    # truncate the color space at both ends, saturating pixels with very
    # little or very high intensity. The color limits vmin and vmax are
    # determined based on color_truncation_factor_lower and
    # color_truncation_factor_upper, respectively. These specify the
    # accumulated fraction of Σbins at which the histogram should be
    # truncated, for the lower and upper intensity ends. For images with
    # a lot of structure the best results are obtained by giving the
    # lower color truncation quite a large value (this effectively
    # removes the background), while giving the higher color truncation
    # a small value, so that small very overdense regions
    # appear clearly.
    color_truncation_factor_lower = 0.005
    color_truncation_factor_upper = 0.0001
    occupation = 0
    for i in range(1, N_bins):
        occupation += bins[i]
        if occupation >= ℝ[color_truncation_factor_lower*Σbins]:
            vmin = bin_edges[i - 1]
            break
    occupation = 0
    for i in range(N_bins - 1, 0, -1):
        occupation += bins[i]
        if occupation >= ℝ[color_truncation_factor_upper*Σbins]:
            vmax = bin_edges[i + 1]
            break
    # Return the enhanced image, the color limits and the exponent
    return image_enhanced, vmin, vmax, exponent

# Function for chancing the colormap of the terminal
def set_terminal_colormap(colormap):
    """This function constructs and apply a terminal colormap with
    256 - (16 + 2) = 238 ANSI/VT100 control sequences, remapping the
    238 higher color numbers. The 16 + 2 = 18 lowest are left alone in
    order not to mess with standard terminal coloring and the colors
    used for the CO𝘕CEPT logo at startup.
    We apply the colormap even if the specified colormap is already
    in use, as the resulting log file is easier to parse with every
    colormap application present.
    """
    if not master:
        return
    colormap_ansi = getattr(matplotlib.cm, colormap)(linspace(0, 1, 238))[:, :3]
    for i, rgb in enumerate(colormap_ansi):
        colorhex = matplotlib.colors.rgb2hex(rgb)
        statechange = esc_set_color.format(18 + i, *[colorhex[c:c+2] for c in range(1, 7, 2)])
        # As this does not actually print anything on the screen,
        # we use the normal print function as to not mess with the
        # bookkeeping inside fancyprint.
        print(statechange, end='')

# Function for 3D renderings of the components
@cython.header(
    # Arguments
    components=list,
    filename=str,
    cleanup='bint',
    tmp_dirname=str,
    # Locals
    N='Py_ssize_t',
    N_local='Py_ssize_t',
    a_str=str,
    artists_text=dict,
    color='double[::1]',
    component='Component',
    component_dict=dict,
    domain_start_i='Py_ssize_t',
    domain_start_j='Py_ssize_t',
    domain_start_k='Py_ssize_t',
    figname=str,
    filename_component=str,
    filename_component_alpha=str,
    filename_component_alpha_part=str,
    filenames_component_alpha=list,
    filenames_component_alpha_part=list,
    filenames_components=list,
    gridsize='Py_ssize_t',
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    k='Py_ssize_t',
    label_props=list,
    label_spacing='double',
    name=str,
    names=tuple,
    part='int',
    posx_mv='double[::1]',
    posy_mv='double[::1]',
    posz_mv='double[::1]',
    render3D_dir=str,
    rgbα='double[:, ::1]',
    scatter_size='double',
    size='Py_ssize_t',
    size_i='Py_ssize_t',
    size_j='Py_ssize_t',
    size_k='Py_ssize_t',
    t_str=str,
    xi='double',
    yj='double',
    zk='double',
    α='double',
    α_factor='double',
    α_homogeneous='double',
    α_min='double',
    ϱ_noghosts='double[:, :, :]',
    ϱbar_component='double',
)
def render3D(components, filename, cleanup=True, tmp_dirname='.renders3D'):
    global render3D_image
    # Do not 3D render anything if
    # render3D_select does not contain any True values.
    if not any(render3D_select.values()):
        return
    # Attach missing extension to filename
    if not filename.endswith('.png'):
        filename += '.png'
    # The directory for storing the temporary 3D renders
    render3D_dir = '{}/{}'.format(os.path.dirname(filename), tmp_dirname)
    # Initialize figures by building up render3D_dict, if this is the
    # first time this function is called.
    if not render3D_dict:
        masterprint('Initializing 3D renders ...')
        # Make cyclic default colors as when doing multiple plots in
        # one figure. Make sure that none of the colors are identical
        # to the background color.
        default_colors = itertools.cycle([to_rgb(prop['color'])
                                          for prop in matplotlib.rcParams['axes.prop_cycle']
                                          if not all(to_rgb(prop['color']) == render3D_bgcolor)])
        for component in components:
            if not is_selected(component, render3D_select):
                continue
            # This component should be 3D rendered.
            # Prepare a figure for the 3D render of the i'th component.
            figname = 'render3D_{}'.format(component.name)
            dpi = 100  # This only affects the font size relative to the figure
            fig = plt.figure(figname, figsize=[render3D_resolution/dpi]*2, dpi=dpi)
            ax = fig.gca(projection='3d', facecolor=render3D_bgcolor)
            # The color and α (of a homogeneous column through the
            # entire box) of this component.
            if component.name.lower() in render3D_colors:
                # This component is given a specific color by the user
                color, α_homogeneous = render3D_colors[component.name.lower()]
            elif 'all' in render3D_colors:
                # All components are given the same color by the user
                color, α_homogeneous = render3D_colors['all']
            else:
                # No color specified for this particular component.
                # Assign the next color from the default cyclic colors.
                color = next(default_colors)
                α_homogeneous = 0.2
            # Alpha values below this small value appear completely
            # invisible, for whatever reason.
            α_min = 0.0059
            # The artist for the component
            if component.representation == 'particles':
                # The particle size on the figure.
                # The size is chosen such that the particles stand side
                # by side in a homogeneous universe (more or less).
                N = component.N
                scatter_size = 1550*np.prod(fig.get_size_inches())/N**ℝ[2/3]
                # Determine the α value which ensures that a homogeneous
                # column through the entire box will result in a
                # combined α value of α_homogeneous. Alpha blending
                # is non-linear, but via the code given in
                # https://stackoverflow.com/questions/28946400
                # /is-it-possible-for-matplotlibs-alpha-transparency
                # -values-to-sum-to-1
                # I have found that 4/∛N is a good approximation to
                # the α value needed to make the combined α equal to 1.
                α = α_homogeneous*4/cbrt(N)
                # Alpha values lower than α_min are not allowed.
                # Shrink the scatter size to make up for the larger α.
                if α < α_min:
                    scatter_size *= α/α_min
                    α = α_min
                # Apply size and alpha
                artist_component = ax.scatter(0, 0, 0,
                                              alpha=α,
                                              c=np.expand_dims(color, 0),
                                              s=scatter_size,
                                              depthshade=False,
                                              lw=0,
                                              )
            elif component.representation == 'fluid':
                # To 3D render fluid elements, their explicit positions
                # are needed. In the following, these are computed and
                # stored in the variables posx_mv, posy_mv and posz_mv.
                size_i = component.shape_noghosts[0]
                size_j = component.shape_noghosts[1]
                size_k = component.shape_noghosts[2]
                # Number of local fluid elements
                size = size_i*size_j*size_k
                # Allocate arrays for storing grid positions
                posx_mv = empty(size, dtype='double')
                posy_mv = empty(size, dtype='double')
                posz_mv = empty(size, dtype='double')
                # Fill the arrays
                gridsize = component.gridsize
                domain_start_i = domain_layout_local_indices[0]*size_i
                domain_start_j = domain_layout_local_indices[1]*size_j
                domain_start_k = domain_layout_local_indices[2]*size_k
                index = 0
                for i in range(size_i):
                    xi = (ℝ[domain_start_i + 0.5] + i)*ℝ[boxsize/gridsize]
                    for j in range(size_j):
                        yj = (ℝ[domain_start_j + 0.5] + j)*ℝ[boxsize/gridsize]
                        for k in range(size_k):
                            zk = (ℝ[domain_start_k + 0.5] + k)*ℝ[boxsize/gridsize]
                            posx_mv[index] = xi
                            posy_mv[index] = yj
                            posz_mv[index] = zk
                            index += 1
                # 2D array with rgbα rows, one row for each
                # fluid element. This is the only array which will be
                # updated for each new 3D render, and only the α column
                # will be updated.
                rgbα = empty((size, 4), dtype=C2np['double'])
                for i in range(size):
                    for dim in range(3):
                        rgbα[i, dim] = color[dim]
                    rgbα[i, 3] = 1
                # The particle (fluid element) size on the figure.
                # The size is chosen such that the particles stand side
                # by side in a homogeneous universe (more or less).
                N = gridsize**3
                scatter_size = 1550*np.prod(fig.get_size_inches())/N**ℝ[2/3]
                # Determine the α multiplication factor which ensures
                # that a homogeneous column through the entire box will
                # result in an α value of α_homogeneous. Alpha blending
                # is non-linear, but via the code given in
                # https://stackoverflow.com/questions/28946400
                # /is-it-possible-for-matplotlibs-alpha-transparency
                # -values-to-sum-to-1
                # I have found that 4/∛N is a good approximation to
                # the α value needed to make the combined α equal to 1.
                α_factor = α_homogeneous*4/cbrt(N)
                # An α_factor below α_min are not allowed.
                # Shrink the scatter size to make up for the larger α.
                if α_factor < α_min:
                    scatter_size *= α_factor/α_min
                    α_factor = α_min
                # Plot the fluid elements as a 3D scatter plot
                artist_component = ax.scatter(posx_mv, posy_mv, posz_mv,
                                              c=rgbα,
                                              s=scatter_size,
                                              depthshade=False,
                                              lw=0,
                                              )
                # The set_facecolors method on the artist can be used
                # to update the α values on the plot. This function is
                # called internally by matplotlib with wrong arguments,
                # cancelling the α updates. For this reason, we
                # replace this method with a dummy method, while
                # keeping the original as _set_facecolors (though we
                # do not use this, as we set the _facecolors attribute
                # manually instead).
                artist_component._set_facecolors = artist_component.set_facecolors
                artist_component.set_facecolors = dummy_func
            # The artists for the cosmic time and scale factor text
            artists_text = {}
            label_spacing = 0.07
            label_props = [(label_spacing,     label_spacing, 'left'),
                           (1 - label_spacing, label_spacing, 'right')]
            artists_text['t'] = ax.text2D(label_props[0][0],
                                          label_props[0][1],
                                          '',
                                          fontsize=16,
                                          horizontalalignment=label_props[0][2],
                                          transform=ax.transAxes,
                                          )
            if enable_Hubble:
                artists_text['a'] = ax.text2D(label_props[1][0],
                                              label_props[1][1],
                                              '',
                                              fontsize=16,
                                              horizontalalignment=label_props[1][2],
                                              transform=ax.transAxes,
                                              )
            # Configure axis options
            ax.dist = 9  # Zoom level
            ax.set_proj_type('ortho')
            ax.set_xlim(0, boxsize)
            ax.set_ylim(0, boxsize)
            ax.set_zlim(0, boxsize)
            ax.axis('off')  # Remove panes, gridlines, axes, ticks, etc.
            for spine in ax.spines.values():
                # Needed due to bug in matplotlib 3.0.0
                spine.set_visible(False)
            plt.tight_layout(pad=-1)  # Extra tight layout, to prevent white frame
            # Store the figure, axes and the component
            # and text artists in the render3D_dict.
            render3D_dict[component.name] = {'fig': fig,
                                             'ax': ax,
                                             'artist_component': artist_component,
                                             'artists_text': artists_text,
                'α_factor': (α_factor if component.representation == 'fluid' else None),
                'rgbα'    : (rgbα     if component.representation == 'fluid' else None),
                                           }
        masterprint('done')
        # Return if no component is to be 3D rendered
        if not render3D_dict:
            return
    # Create the temporary 3D render directory if necessary
    if not (nprocs == 1 == len(render3D_dict)):
        if master:
            os.makedirs(render3D_dir, exist_ok=True)
        Barrier()
    # Print out progress message
    names = tuple(render3D_dict.keys())
    if len(names) == 1:
        masterprint(f'Rendering {names[0]} in 3D and saving to "{filename}" ...')
    else:
        filenames_components = []
        for name in names:
            filename_component = augment_filename(filename, name, '.png')
            filenames_components.append(f'"{filename_component}"')
        masterprint('3D rendering {} and saving to {} ...'
                    .format(', '.join(names), ', '.join(filenames_components)))
    # 3D render each component separately
    for component in components:
        if component.name not in render3D_dict:
            continue
        # Switch to the render3D figure
        figname = 'render3D_{}'.format(component.name)
        plt.figure(figname)
        # Extract figure elements
        component_dict   = render3D_dict[component.name]
        fig              = component_dict['fig']
        ax               = component_dict['ax']
        artist_component = component_dict['artist_component']
        artists_text     = component_dict['artists_text']
        if component.representation == 'particles':
            # Update particle positions on the figure
            N_local = component.N_local
            artist_component._offsets3d = (component.posx_mv[:N_local],
                                           component.posy_mv[:N_local],
                                           component.posz_mv[:N_local])
        elif component.representation == 'fluid':
            rgbα     = component_dict['rgbα']
            α_factor = component_dict['α_factor']
            # Measure the mean value of the ϱ grid
            ϱ_noghosts = component.ϱ.grid_noghosts
            ϱbar_component = allreduce(np.sum(ϱ_noghosts), op=MPI.SUM)/component.gridsize**3
            # Update the α values in rgbα array based on the values of
            # ϱ at each grid point. The rgb-values remain the same for
            # all 3D renders of this component.
            index = 0
            for         i in range(ℤ[ϱ_noghosts.shape[0]]):
                for     j in range(ℤ[ϱ_noghosts.shape[1]]):
                    for k in range(ℤ[ϱ_noghosts.shape[2]]):
                        α = ℝ[α_factor/ϱbar_component]*ϱ_noghosts[i, j, k]
                        if α > 1:
                            α = 1
                        rgbα[index, 3] = α
                        index += 1
            # Apply the new α values to the artist.
            # We do this by setting the attribute _facecolors,
            # which is much faster than using the set_facecolors
            # method.
            artist_component._facecolors = rgbα
        # Print the current cosmic time and scale factor on the figure
        if master:
            t_str = a_str = ''
            t_str = '$t = {}\, \mathrm{{{}}}$'.format(significant_figures(universals.t, 4, 'tex'),
                                                      unit_time)
            artists_text['t'].set_text(t_str)
            if enable_Hubble:
                a_str = '$a = {}$'.format(significant_figures(universals.a, 4, 'tex'))
                artists_text['a'].set_text(a_str)
            # Make the text color black or white,
            # dependent on the background color.
            for artist_text in artists_text.values():
                if sum(render3D_bgcolor) < 1:
                    artist_text.set_color('white')
                else:
                    artist_text.set_color('black')
        # Save the 3D render
        if nprocs == 1:
            filename_component_alpha_part = ('{}/{}_alpha.png'
                                              .format(render3D_dir,
                                                      component.name.replace(' ', '-')))
        else:
            filename_component_alpha_part = ('{}/{}_alpha_{}.png'
                                             .format(render3D_dir,
                                                     component.name.replace(' ', '-'),
                                                     rank))
        if nprocs == 1 == len(render3D_dict):
            # As this is the only 3D render which should be done, it can
            # be saved directly in its final, non-transparent state.
            plt.savefig(filename, transparent=False)
            masterprint('done')
        else:
            # Save transparent 3D render
            plt.savefig(filename_component_alpha_part, transparent=True)
    # All 3D rendering done
    Barrier()
    # The partial 3D renders will now be combined into full 3D renders,
    # stored in the 'render3D_image', variable. Partial 3D renders of
    # the j'th component will be handled by the process with rank j.
    if not (nprocs == 1 == len(render3D_dict)):
        # Loop over components designated to each process
        for i in range(1 + len(render3D_dict)//nprocs):
            # Break out when there is no more work for this process
            j = rank + nprocs*i
            if j >= len(names):
                break
            name = names[j].replace(' ', '-')
            if nprocs == 1:
                # Simply load the already fully constructed image
                filename_component_alpha = '{}/{}_alpha.png'.format(render3D_dir, name)
                render3D_image = plt.imread(filename_component_alpha)
            else:
                # Create list of filenames for the partial 3D renders
                filenames_component_alpha_part = ['{}/{}_alpha_{}.png'
                                                  .format(render3D_dir, name, part)
                                                  for part in range(nprocs)]
                # Read in the partial 3D renders and blend
                # them together into the render3D_image variable.
                blend(filenames_component_alpha_part)
                # Save combined 3D render of the j'th component
                # with transparency. Theese are then later combined into
                # a 3D render containing all components.
                if len(names) > 1:
                    filename_component_alpha = '{}/{}_alpha.png'.format(render3D_dir, name)
                    plt.imsave(filename_component_alpha, asarray(render3D_image))
            # Add opaque background to render3D_image
            add_background()
            # Save combined 3D render of the j'th component
            # without transparency.
            filename_component = filename
            if len(names) > 1:
                filename_component = augment_filename(filename, name, '.png')
            plt.imsave(filename_component, asarray(render3D_image))
        Barrier()
        masterprint('done')
        # Finally, combine the full 3D renders of individual components
        # into a total 3D render containing all components.
        if master and len(names) > 1:
            masterprint('Combining component 3D renders and saving to "{}" ...'.format(filename))
            filenames_component_alpha = ['{}/{}_alpha.png'.format(render3D_dir,
                                                                  name.replace(' ', '-'))
                                         for name in names]
            blend(filenames_component_alpha)
            # Add opaque background to render3D_image and save it
            add_background()
            plt.imsave(filename, asarray(render3D_image))
            masterprint('done')
    # Remove the temporary directory, if cleanup is requested
    if master and cleanup and not (nprocs == 1 == len(render3D_dict)):
        shutil.rmtree(render3D_dir)
# Declare global variables used in the render3D function
cython.declare(render3D_dict=object,  # OrderedDict
               render3D_image='float[:, :, ::1]',
               )
# (Ordered) dictionary containing the figure, axes, component
# artist and text artist for each component.
render3D_dict = collections.OrderedDict()
# The array storing the 3D render
render3D_image = empty((render3D_resolution, render3D_resolution, 4), dtype=C2np['float'])
# Dummy function
def dummy_func(*args, **kwargs):
    return None

# Function which takes in a list of filenames of images and blend them
# together into the global render3D_image array.
@cython.header(# Arguments
               filenames=list,
               # Locals
               alpha_A='float',
               alpha_B='float',
               alpha_tot='float',
               i='int',
               j='int',
               rgb='int',
               rgbα='int',
               tmp_image='float[:, :, ::1]',
               )
def blend(filenames):
    # Make render3D_image black and transparent
    render3D_image[...] = 0
    for filename in filenames:
        tmp_image = plt.imread(filename)
        for     i in range(render3D_resolution):
            for j in range(render3D_resolution):
                # Pixels with 0 alpha has (r, g, b) = (1, 1, 1)
                # (this is a defect of plt.savefig).
                # These should be disregarded completely.
                alpha_A = tmp_image[i, j, 3]
                if alpha_A != 0:
                    # Combine render3D_image with tmp_image by
                    # adding them together, using their alpha values
                    # as weights.
                    alpha_B = render3D_image[i, j, 3]
                    alpha_tot = alpha_A + alpha_B - alpha_A*alpha_B
                    for rgb in range(3):
                        render3D_image[i, j, rgb] = (
                            (alpha_A*tmp_image[i, j, rgb] + alpha_B*render3D_image[i, j, rgb])
                            /alpha_tot
                        )
                    render3D_image[i, j, 3] = alpha_tot
    # Some pixel values in the combined 3D render may have overflown.
    # Clip at saturation value.
    for     i in range(render3D_resolution):
        for j in range(render3D_resolution):
            for rgbα in range(4):
                if render3D_image[i, j, rgbα] > 1:
                    render3D_image[i, j, rgbα] = 1

# Function for adding background color to render3D_image
@cython.header(# Locals
               alpha='float',
               i='int',
               j='int',
               rgb='int',
               )
def add_background():
    for     i in range(render3D_resolution):
        for j in range(render3D_resolution):
            alpha = render3D_image[i, j, 3]
            # Add background using "A over B" alpha blending
            for rgb in range(3):
                render3D_image[i, j, rgb] = (
                    alpha*render3D_image[i, j, rgb] + (1 - alpha)*render3D_bgcolor[rgb]
                )
                render3D_image[i, j, 3] = 1

# Function for augmenting a filename with a given text
def augment_filename(filename, text, ext=''):
    """Example of use:
    augment_filename('/path/to/powerspec_a=1.0.png', 'matter', 'png')
      -> '/path/to/powerspec_matter_a=1.0.png'
    """
    text = text.lstrip('_')
    ext = '.' + ext.lstrip('.')
    dirname, basename = os.path.split(filename)
    basename, baseext = os.path.splitext(basename)
    if baseext != ext:
        basename += baseext
    time_param_indices = collections.defaultdict(int)
    for time_param in ('t', 'a'):
        try:
            time_param_indices[time_param] = basename.index(f'_{time_param}=')
        except ValueError:
            continue
    if time_param_indices['t'] == time_param_indices['a']:
        basename += f'_{text}'
    else:
        time_param = sorted(time_param_indices.items(), key=lambda tup: tup[::-1])[-1][0]
        basename = (f'_{text}_{time_param}='
            .join(basename.rsplit(f'_{time_param}=', 1))
        )
    if ext != '.':
        basename += ext
    return os.path.join(dirname, basename)
