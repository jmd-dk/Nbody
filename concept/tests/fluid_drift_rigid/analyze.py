# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the CO𝘕CEPT snapshots
species.allow_similarly_named_components = True
fluid_components = []
particle_components = []
a = []
for fname in sorted(glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    for component in snapshot.components:
        if component.representation == 'fluid':
            fluid_components.append(component)
        elif component.representation == 'particles':
            particle_components.append(component)
    a.append(snapshot.params['a'])
gridsize = fluid_components[0].gridsize
N = particle_components[0].N
N_snapshots = len(a)
# Sort data chronologically
order = np.argsort(a)
a                   = [a[o]                   for o in order]
fluid_components    = [fluid_components[o]    for o in order]
particle_components = [particle_components[o] for o in order]

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Extract ϱ(x) of fluids and y(x) of particles.
# To compare ϱ to y, a scaling is needed.
# Since the x's in ϱ(x) are discretized, but the x's in y(x) are not,
# we interpolate y to the disretized x-values.
x_fluid = asarray([boxsize*i/gridsize for i in range(gridsize)])
ϱ = []
y = []
y_interp = []
for fluid, particles in zip(fluid_components, particle_components):
    ϱ.append(fluid.ϱ.grid_noghosts[:gridsize, 0, 0])
    y_i = particles.posy.copy()
    A_fluid          = 0.5*(max(ϱ[0]) - min(ϱ[0]))
    offset_fluid     = 0.5*(max(ϱ[0]) + min(ϱ[0]))
    A_particles      = 0.5*(max(y_i)  - min(y_i))
    offset_particles = 0.5*(max(y_i)  + min(y_i))
    y_i -= offset_particles
    y_i *= A_fluid/A_particles
    y_i += offset_fluid
    y.append(y_i)
    # Interpolation is made by a simple polynomial fit,
    # but with a large order.
    order = 15
    y_interp.append(np.polyval(np.polyfit(particles.posx, y_i, order), x_fluid))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, figsize=(8, 3*N_snapshots))
for ax_i, particles, ϱ_i, y_i, y_interp_i, a_i in zip(ax, particle_components, ϱ, y, y_interp, a):
    indices_sorted = np.argsort(particles.posx)
    index_min = np.argmin(particles.posx)
    index_max = np.argmax(particles.posx)
    ax_i.plot(np.concatenate(([max(particles.posx) - boxsize],
                              particles.posx[indices_sorted],
                              [min(particles.posx) + boxsize])),
              np.concatenate(([y_i[index_max]],
                              y_i[indices_sorted],
                              [y_i[index_min]])),
              '-', label='Particle simulation')
    ax_i.plot(x_fluid, ϱ_i, '.', markersize=10, alpha=0.7, label='Fluid simulation')
    ax_i.set_ylabel(
        'scaled and shifted $y$,\n'
        r'$\varrho$ $\mathrm{{[{}\,m_{{\odot}}\,{}^{{-3}}]}}$'
        .format(
            significant_figures(
                1/units.m_sun,
                3,
                fmt='tex',
                incl_zeros=False,
                scientific=False,
            ),
            unit_length,
        )
    )
    ax_i.set_title(rf'$a={a_i:.3g}$')
plt.xlim(0, boxsize)
plt.xlabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
ax[0].legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same ϱ
# and all fluid elements should have the same u = J/ϱ.
tol_fac_ϱ = 1e-6
tol_fac_u = 1e-3
for fluid, a_i in zip(fluid_components, a):
    for fluidscalar in fluid.iterate_fluidscalars():
        varnum = fluidscalar.varnum
        grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
        if varnum == 0:
            # ϱ
            ϱ_grid = grid
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(np.std(yz_slice), 0,
                               rel_tol=0,
                               abs_tol=max((tol_fac_ϱ*np.std(grid), 1e+1*gridsize**2*machine_ϵ))):
                    abort('Non-uniformities have emerged at a = {} '
                          'in yz-slices of fluid scalar variable {}.\n'
                          'See "{}" for a visualization.'
                          .format(a_i, fluidscalar, fig_file))
        elif varnum == 1:
            # J
            u_grid = grid/ϱ_grid
            if not isclose(np.std(u_grid), 0,
                           rel_tol=0,
                           abs_tol=(tol_fac_u*abs(np.mean(u_grid)) + machine_ϵ)):
                abort('Non-uniformities have emerged at a = {} '
                      'in fluid scalar variable {}'
                      .format(a_i, fluidscalar))

# Compare ϱ to the fluid from the snapshots
tol_fac = 0.02
for ϱ_i, y_interp_i, a_i in zip(ϱ, y_interp, a):
    if not isclose(mean(abs(ϱ_i - y_interp_i)), 0,
                   rel_tol=0,
                   abs_tol=(tol_fac*np.std(ϱ_i) + machine_ϵ)):
        abort('Fluid drift differs from particle drift at a = {:.3g}.\n'
              'See "{}" for a visualization.'
              .format(a_i, fig_file))

# Done analyzing
masterprint('done')
