# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the snapshots
species.allow_similarly_named_components = True
fluids = {'particles simulations': [], 'fluid simulations': []}
times = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = '{}/output/{}/snapshot_t=*_converted*'.format(this_dir, kind)
    elif kind == 'fluid':
        regex = '{}/output/{}/snapshot_t=*'.format(this_dir, kind)
    for fname in sorted(glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        fluids[kind + ' simulations'].append(snapshot.components[0])
        if kind == 'particles':
            times.append(float(re.search('snapshot_t=(.*)' + unit_time, fname).group(1)))
N_snapshots = len(times)
gridsize = fluids['particles simulations'][0].gridsize
# Sort data chronologically
order = np.argsort(times)
times = [times[o]  for o in order]
for kind in ('particles', 'fluid'):
    fluids[kind + ' simulations'] = [fluids[kind + ' simulations'][o] for o in order]
# Use precise times
times = snapshot_times['t']

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x = [boxsize*i/gridsize for i in range(gridsize)]
ϱ = {'particles simulations': [], 'fluid simulations': []}
for kind, markersize in zip(('particles', 'fluid'), (15, 10)):
    for ax_i, fluid, t in zip(ax, fluids[kind + ' simulations'], times):
        ϱ[kind + ' simulations'].append(fluid.ϱ.grid_noghosts[:gridsize,
                                                              :gridsize,
                                                              :gridsize].mean((1, 2)))
        ax_i.plot(x, ϱ[kind + ' simulations'][-1],
                  '.',
                  markersize=markersize,
                  alpha=0.7,
                  label=(kind.rstrip('s').capitalize() + ' simulation'),
                  )
        ax_i.set_ylabel(
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
        ax_i.set_title(rf'$t={t:.3g}\,\mathrm{{{unit_time}}}$')
plt.xlim(0, boxsize)
plt.xlabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
ax[0].legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Fluid elements in yz-slices should all have the same ϱ and J
tol_fac = 1e-6
for kind in ('particles', 'fluid'):
    for fluid, t in zip(fluids[kind + ' simulations'], times):
        for fluidscalar in fluid.iterate_fluidscalars():
            grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
            for i in range(gridsize):
                yz_slice = grid[i, :, :]
                if not isclose(np.std(yz_slice), 0,
                               rel_tol=0,
                               abs_tol=max((tol_fac*np.std(grid), 1e+1*gridsize**2*machine_ϵ))):
                    abort('Non-uniformities have emerged after {} {} '
                          'in yz-slices of fluid scalar variable {} '
                          'in {} simulation.\n'
                          'See "{}" for a visualization.'
                          .format(t, unit_time, fluidscalar, kind.rstrip('s'), fig_file))

# Compare ϱ's from the fluid and snapshot simulations
discontinuity_tol = 2
rel_tol = 0.1
for ϱ_fluid, ϱ_particles, t in zip(ϱ['fluid simulations'], ϱ['particles simulations'], times):
    abs_tol = rel_tol*np.std(ϱ_particles)
    slope_left  = ϱ_particles - np.roll(ϱ_particles, -1)
    slope_right = np.roll(ϱ_particles, +1) - ϱ_particles
    discontinuities = abs(slope_right - slope_left)
    discontinuities = [max(d) for d in zip(*[np.roll(discontinuities, r) for r in range(-3, 4)])]
    if not all(isclose(ϱ_fluid_i, ϱ_particles_i,
                       rel_tol=0,
                       abs_tol=(discontinuity_tol*discontinuity + abs_tol),
                       ) for ϱ_fluid_i, ϱ_particles_i, discontinuity in zip(ϱ_fluid,
                                                                            ϱ_particles,
                                                                            discontinuities)):
        abort('Fluid did not evolve correctly up to t = {} {}.\n'
              'See "{}" for a visualization.'
              .format(t, unit_time, fig_file))

# Done analysing
masterprint('done')
