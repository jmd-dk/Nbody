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
nprocs_list = sorted(int(dname[(dname.index('_') + 1):])
                     for dname in [os.path.basename(dname)
                                   for dname in glob('{}/output_*'.format(this_dir))])
a = []
components = {n: [] for n in nprocs_list}
for n in nprocs_list:
    for fname in sorted(glob('{}/output_{}/snapshot_a=*'.format(this_dir, n)),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        if n == 1:
            a.append(snapshot.params['a'])
        components[n].append(snapshot.components[0])
N_snapshots = len(a)

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Using the particle order of the n=1 snapshot as the standard,
# find the corresponding ID's in the snapshots and order these
# particles accordingly.
N = components[1][0].N
D2 = zeros(N, dtype=float)
ID = zeros(N, dtype=int)
for i in range(N_snapshots):
    x = components[1][i].posx
    y = components[1][i].posy
    z = components[1][i].posz
    for n in nprocs_list:
        x_procs = components[n][i].posx
        y_procs = components[n][i].posy
        z_procs = components[n][i].posz
        for j in range(N):
            for k in range(N):
                dx = x[j] - x_procs[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y[j] - y_procs[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z[j] - z_procs[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[j] = np.argmin(D2)
        components[n][i].posx = components[n][i].posx[ID]
        components[n][i].posy = components[n][i].posy[ID]
        components[n][i].posz = components[n][i].posz[ID]
        components[n][i].momx = components[n][i].momx[ID]
        components[n][i].momy = components[n][i].momy[ID]
        components[n][i].momz = components[n][i].momz[ID]

# Compute distance between particles in the two snapshots
dist = collections.OrderedDict((n, []) for n in nprocs_list[1:])
for i in range(N_snapshots):
    x = {n: components[n][i].posx for n in nprocs_list}
    y = {n: components[n][i].posy for n in nprocs_list}
    z = {n: components[n][i].posz for n in nprocs_list}
    for n in nprocs_list[1:]:
        dist[n].append(sqrt(asarray([
            min([
                + (x[1][j] - x[n][j] + xsgn*boxsize)**2
                + (y[1][j] - y[n][j] + ysgn*boxsize)**2
                + (z[1][j] - z[n][j] + zsgn*boxsize)**2
                for xsgn in (-1, 0, +1)
                for ysgn in (-1, 0, +1)
                for zsgn in (-1, 0, +1)
            ])
            for j in range(N)
        ])))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(len(nprocs_list) - 1, sharex=True, sharey=True)
for n, d, ax_i in zip(dist.keys(), dist.values(), ax):
    for i in range(N_snapshots):
        ax_i.semilogy(
            machine_ϵ + asarray(d[i])/boxsize,
            '.',
            alpha=0.7,
            label=f'$a={a[i]}$',
            zorder=-i,
        )
    ax_i.set_ylabel(
        rf'$|\mathbf{{x}}_{{{n}}} - \mathbf{{x}}_1|'
        rf'/\mathrm{{boxsize}}$'
    )
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax_i.get_xticklabels() for ax_i in ax[:-1]], visible=False)
ax[0].legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1e-3
if any(np.mean(asarray(d)/boxsize) > tol for d in dist.values()):
    abort('Runs with different numbers of processes yield different results!\n'
          'See "{}" for a visualization.'.format(fig_file))

# Done analysing
masterprint('done')
