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
a = []
components = []
for fname in sorted(glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    a.append(snapshot.params['a'])
    components.append(snapshot.components[0])
N_snapshots = len(a)

# Read in data from the GADGET-2 snapshots
components_gadget = []
for fname in sorted(glob(this_dir + '/Gadget2/output/snapshot_*'))[:N_snapshots]:
    components_gadget.append(load(fname, compare_params=False, only_components=True)[0])

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Using the particle order of CO𝘕CEPT as the standard,
# find the corresponding ID's in the GADGET-2 snapshots
# and order these particles accordingly.
N = components[0].N
D2 = zeros(N, dtype=float)
ID = zeros(N, dtype=int)
for i in range(N_snapshots):
    x = components[i].posx
    y = components[i].posy
    z = components[i].posz
    x_gadget = components_gadget[i].posx
    y_gadget = components_gadget[i].posy
    z_gadget = components_gadget[i].posz
    for j in range(N):
        for k in range(N):
            dx = x[j] - x_gadget[k]
            if dx > 0.5*boxsize:
                dx -= boxsize
            elif dx < -0.5*boxsize:
                dx += boxsize
            dy = y[j] - y_gadget[k]
            if dy > 0.5*boxsize:
                dy -= boxsize
            elif dy < -0.5*boxsize:
                dy += boxsize
            dz = z[j] - z_gadget[k]
            if dz > 0.5*boxsize:
                dz -= boxsize
            elif dz < -0.5*boxsize:
                dz += boxsize
            D2[k] = dx**2 + dy**2 + dz**2
        ID[j] = np.argmin(D2)
    components_gadget[i].posx = components_gadget[i].posx[ID]
    components_gadget[i].posy = components_gadget[i].posy[ID]
    components_gadget[i].posz = components_gadget[i].posz[ID]
    components_gadget[i].momx = components_gadget[i].momx[ID]
    components_gadget[i].momy = components_gadget[i].momy[ID]
    components_gadget[i].momz = components_gadget[i].momz[ID]

# Compute distance between particles in the two snapshots
dist = []
for i in range(N_snapshots):
    x = components[i].posx
    y = components[i].posy
    z = components[i].posz
    x_gadget = components_gadget[i].posx
    y_gadget = components_gadget[i].posy
    z_gadget = components_gadget[i].posz
    dist.append(sqrt(asarray([
        min([
            + (x[j] - x_gadget[j] + xsgn*boxsize)**2
            + (y[j] - y_gadget[j] + ysgn*boxsize)**2
            + (z[j] - z_gadget[j] + zsgn*boxsize)**2
            for xsgn in (-1, 0, +1)
            for ysgn in (-1, 0, +1)
            for zsgn in (-1, 0, +1)
        ])
        for j in range(N)
    ])))
    # Plot
    plt.semilogy(
        machine_ϵ + dist[i]/boxsize,
        '.',
        alpha=0.7,
        label=f'$a={a[i]}$',
        zorder=-i,
    )

# Finalize plot
fig_file = this_dir + '/result.png'
plt.xlabel('Particle number')
plt.ylabel(
    r'$|\mathbf{x}_{\mathrm{CO}N\mathrm{CEPT}} - \mathbf{x}_{\mathrm{GADGET}}|'
    r'/\mathrm{boxsize}$'
)
plt.xlim(0, N - 1)
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Printout error message for unsuccessful test
tol = 1.2e-2
if any(np.mean(d/boxsize) > tol for d in dist):
    abort('The results from CO𝘕CEPT disagree with those from GADGET-2.\n'
          'See "{}" for a visualization.'.format(fig_file))

# Done analysing
masterprint('done')
