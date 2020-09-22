# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)
species.allow_similarly_named_components = True

# Read in data from the CO𝘕CEPT snapshots
a = []
nprocs_list = sorted(int(dname[(dname.index('python_') + 7):])
                     for dname in [os.path.basename(dname)
                                   for dname in glob('{}/output_python_*'.format(this_dir))])
components = {'cython': {n: [] for n in nprocs_list},
              'python': {n: [] for n in nprocs_list}}
for cp in components.keys():
    for n in nprocs_list:
        for fname in sorted(glob('{}/output_{}_{}/snapshot_a=*'.format(this_dir, cp, n)),
                            key=lambda s: s[(s.index('=') + 1):]):
            snapshot = load(fname, compare_params=False)
            if cp == 'cython' and n == 1:
                a.append(snapshot.params['a'])
            components[cp][n].append(snapshot.components[0])
N_snapshots = len(a)

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Using the particle order of the cython snapshot as the standard, find the corresponding
# ID's in the python snapshots and order these particles accoringly.
N = components['cython'][1][0].N
D2 = zeros(N)
ID = zeros(N, dtype='int')
for i in range(N_snapshots):
    for n in nprocs_list:
        x_cython = components['cython'][n][i].posx
        y_cython = components['cython'][n][i].posy
        z_cython = components['cython'][n][i].posz
        x_python = components['python'][n][i].posx
        y_python = components['python'][n][i].posy
        z_python = components['python'][n][i].posz
        for j in range(N):
            for k in range(N):
                dx = x_cython[j] - x_python[k]
                if dx > 0.5*boxsize:
                    dx -= boxsize
                elif dx < -0.5*boxsize:
                    dx += boxsize
                dy = y_cython[j] - y_python[k]
                if dy > 0.5*boxsize:
                    dy -= boxsize
                elif dy < -0.5*boxsize:
                    dy += boxsize
                dz = z_cython[j] - z_python[k]
                if dz > 0.5*boxsize:
                    dz -= boxsize
                elif dz < -0.5*boxsize:
                    dz += boxsize
                D2[k] = dx**2 + dy**2 + dz**2
            ID[j] = np.argmin(D2)
        components['python'][n][i].posx = components['python'][n][i].posx[ID]
        components['python'][n][i].posy = components['python'][n][i].posy[ID]
        components['python'][n][i].posz = components['python'][n][i].posz[ID]
        components['python'][n][i].momx = components['python'][n][i].momx[ID]
        components['python'][n][i].momy = components['python'][n][i].momy[ID]
        components['python'][n][i].momz = components['python'][n][i].momz[ID]

# Compute distance between particles in the two snapshots
dist = collections.OrderedDict((n, []) for n in nprocs_list)
for i in range(N_snapshots):
    x = {(cp, n): components[cp][n][i].posx for cp in ('cython', 'python') for n in nprocs_list}
    y = {(cp, n): components[cp][n][i].posy for cp in ('cython', 'python') for n in nprocs_list}
    z = {(cp, n): components[cp][n][i].posz for cp in ('cython', 'python') for n in nprocs_list}
    for n in nprocs_list:
        dist[n].append(sqrt(np.array([min([  (x['cython', n][j] - x['python', n][j]
                                              + xsgn*boxsize)**2
                                           + (y['cython', n][j] - y['python', n][j]
                                              + ysgn*boxsize)**2
                                           + (z['cython', n][j] - z['python', n][j]
                                              + zsgn*boxsize)**2
                                           for xsgn in (-1, 0, +1)
                                           for ysgn in (-1, 0, +1)
                                           for zsgn in (-1, 0, +1)])
                                      for j in range(N)])))

# Plot
fig_file = this_dir + '/result.png'
fig, ax = plt.subplots(len(nprocs_list), sharex=True, sharey=True)
for n, d, ax_i in zip(dist.keys(), dist.values(), ax):
    for i in range(N_snapshots):
        ax_i.semilogy(machine_ϵ + np.array(d[i])/boxsize,
                      '.',
                      alpha=0.7,
                      label='$a={}$'.format(a[i]),
                      zorder=-i,
                      )
    ax_i.set_ylabel('$|\mathbf{{x}}_{{\mathrm{{pp}}{n}}} - \mathbf{{x}}_{{\mathrm{{c}}{n}}}|'
                    '/\mathrm{{boxsize}}$'.format(n=n))
ax[-1].set_xlabel('Particle number')
plt.xlim(0, N - 1)
fig.subplots_adjust(hspace=0)
plt.setp([ax_i.get_xticklabels() for ax_i in ax[:-1]], visible=False)
ax[0].legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1e-10
if any(np.mean(np.array(d)/boxsize) > tol for d in dist.values()):
    abort(
        f'Some or all pure Python runs with nprocs = {nprocs_list} yielded results '
        f'different from their compiled counterparts!\n'
        f'See "{fig_file}" for a visualization.'
    )

# Done analyzing
masterprint('done')
