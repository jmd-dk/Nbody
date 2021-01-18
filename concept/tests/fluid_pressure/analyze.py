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
fluids = []
times = []
for fname in sorted(glob(this_dir + '/output/snapshot_t=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    fluids.append(snapshot.components[0])
    times.append(float(re.search('snapshot_t=(.*)' + unit_time, fname).group(1)))
gridsize = fluids[0].gridsize
N_snapshots = len(fluids)
# Sort data chronologically
order = np.argsort(times)
times  = [times[o]  for o in order]
fluids = [fluids[o] for o in order]
# Use precise times
times = snapshot_times['t']

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Extract hidden parameters
w = user_params['_w']
T = user_params['_T']
A = user_params['_A']
ρ0 = user_params['_ρ0']

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(N_snapshots, sharex=True, sharey=True, figsize=(8, 3*N_snapshots))
x_values = [boxsize*i/gridsize for i in range(gridsize)]
ρ = []
ρ_snapshot = []
for ax_i, fluid, t in zip(ax, fluids, times):
    ρ.append(asarray([ρ0 + A*sin(x/boxsize*2*π)*cos(t/T*2*π) for x in x_values]))
    ρ_snapshot.append(fluid.ϱ.grid_noghosts[:gridsize, 0, 0])
    ax_i.plot([0, boxsize], [ρ0    ]*2, 'k:' )
    ax_i.plot([0, boxsize], [ρ0 + A]*2, 'k--')
    ax_i.plot([0, boxsize], [ρ0 - A]*2, 'k--')
    ax_i.plot(x_values, ρ[-1]         , '-', label='Analytical solution')
    ax_i.plot(x_values, ρ_snapshot[-1], '.', markersize=10, alpha=0.7, label='Simulation')
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

# Fluid elements in yz-slices should all have the same values
for fluid, t in zip(fluids, times):
    for fluidscalar in fluid.iterate_fluidscalars():
        varnum = fluidscalar.varnum
        grid = fluidscalar.grid_noghosts[:gridsize, :gridsize, :gridsize]
        for i in range(gridsize):
            yz_slice = grid[i, :, :]
            yz_mean = np.mean(yz_slice)
            if not isclose(
                np.std(yz_slice) if yz_mean == 0 else np.std(yz_slice)/yz_mean,
                0,
                rel_tol=0,
                abs_tol=1e+1*machine_ϵ,
            ):
                abort('Non-uniformities have emerged at t = {} {} '
                      'in yz-slices of fluid scalar variable {}.\n'
                      'See "{}" for a visualization.'
                      .format(t, unit_time, fluidscalar, fig_file))

# Compare ρ from the snapshots to the analytical solution
abs_tol = 1e-2*A
for ρ_i, ρ_snapshot_i, t in zip(ρ, ρ_snapshot, times):
    if not isclose(np.mean(abs(ρ_i - ρ_snapshot_i)), 0,
                   rel_tol=0,
                   abs_tol=abs_tol):
        abort('Fluid evolution differs from the analytical solution at t = {} {}.\n'
              'See "{}" for a visualization.'
              .format(t, unit_time, fig_file))

# Done analysing
masterprint('done')
