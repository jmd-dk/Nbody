# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the CO𝘕CEPT snapshots
for fname in sorted(glob(this_dir + '/output/snapshot_t=*'),
                    key=lambda s: s[(s.index('=') + 1):]):
    snapshot = load(fname, compare_params=False)
    x = snapshot.components[0].posx[0]
    y = snapshot.components[0].posy[0]
    z = snapshot.components[0].posz[0]

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# The particle should have a position of (0.25, 0.25, 0.25)*boxsize
if (   not isclose(x, 0.25*boxsize)
    or not isclose(y, 0.25*boxsize)
    or not isclose(z, 0.25*boxsize)):
    abort('The particle should have drifted to '
          '(0.25, 0.25, 0.25)*boxsize,\n'
          'but the actual position is '
          '({}, {}, {})*boxsize.'.format(x/boxsize,
                                         y/boxsize,
                                         z/boxsize))

# Done analysing
masterprint('done')
