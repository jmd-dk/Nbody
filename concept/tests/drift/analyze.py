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
x = []
x_std = []
for fname in sorted(glob(this_dir + '/output/snapshot_a=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    posx = snapshot.components[0].posx
    a.append(snapshot.params['a'])
    x.append(np.mean(posx))
    x_std.append(np.std(posx))
N_snapshots = len(a)

# Read in data from the GADGET-2 snapshots
x_gadget = []
x_std_gadget = []
for fname in sorted(glob(this_dir + '/Gadget2/output/snapshot_*'))[:N_snapshots]:
    components_gadget = load(fname, compare_params=False, only_components=True)[0]
    x_gadget.append(np.mean(components_gadget.posx))
    x_std_gadget.append(np.std(components_gadget.posx))

# Begin analysis
masterprint('Analyzing {} data ...'.format(this_test))

# Plot
fig_file = this_dir + '/result.png'
plt.text(0.5*max(a), 0.93*boxsize, r'$\uparrow$ End of simulation box $\uparrow$', ha='center')
plt.plot(a, x       , '.', markersize=15, alpha=0.7, label='CO$N$CEPT')
plt.plot(a, x_gadget, '.', markersize=10, alpha=0.7, label='GADGET')
plt.xlabel('$a$')
plt.ylabel(rf'$x\,\mathrm{{[{unit_length}]}}$')
plt.ylim(0, boxsize)
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# There should be no variance on the x positions
tol = 1e+2*N_snapshots*machine_ϵ
if sum(x_std_gadget) > tol:
    abort('Unequal x-positions for the 4 particles in the GADGET-2 snapshots.\n'
          'It is no good to compare the CO𝘕CEPT results to these.')
if sum(x_std) > tol:
    abort('Unequal x-positions for the 4 particles in the snapshots.\n'
          'The symmetric initial conditions have produced asymmetrical results!')

# Print out error message for unsuccessful test
tol = 1e-3
if max(abs(asarray(x)/asarray(x_gadget) - 1)) > tol:
    abort('The results from CO𝘕CEPT disagree with those from GADGET-2.\n'
          'See "{}" for a visualization.'.format(fig_file))

# Done analyzing
masterprint('done')
