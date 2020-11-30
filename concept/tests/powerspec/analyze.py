# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in the particles
component = load(initial_conditions, only_components=True)[0]
N = component.N
posx = component.posx
posy = component.posy
posz = component.posz

# Begin analysis
masterprint(f'Analyzing {this_test} data ...')

# Volume and linear size of cube with the volume of a sphere
# with radius powerspec_options['tophat'].
tophat = is_selected(component, powerspec_options['tophat'])
V = 4*π/3*tophat**3
L = cbrt(V)
# The number of complete L*L*L cubes within the box
N_cubes_lin = int(boxsize//L)
N_cubes = N_cubes_lin**3
# The number of particles in each of these cubes, if the snapshot is completely homogeneous
N_in_cubes_homo = N*V/boxsize**3

# Count how many particles lie within each of the L*L*L cubes
counts = np.zeros([N_cubes_lin + 1]*3, dtype=C2np['int'])
posx_grid = asarray(posx//L, dtype=C2np['int'])
posy_grid = asarray(posy//L, dtype=C2np['int'])
posz_grid = asarray(posz//L, dtype=C2np['int'])
for i in range(N):
    counts[posx_grid[i], posy_grid[i], posz_grid[i]] += 1
# The upper "cubes" are not cubic and should be discarded
counts = counts[:-1, :-1, :-1]

# Normalize counts to get the contrast from homogeneity
counts_contrast = counts.flatten()/N_in_cubes_homo - 1

# The rms (std) of the count contrast is also the rms of the density contrast
σ = np.std(counts_contrast)

# Is the distribution Gaussian? If not, the snapshot has not been generated correctly
plt.figure()
plt.hist(counts_contrast, 100)
s = (r'{:g}% within $1\sigma$'
     .format(np.round(100*sum(abs(counts_contrast) < σ)/counts_contrast.size)))
plt.text(np.sum(plt.xlim())/2, np.sum(plt.ylim())/2, s, ha='center',
         bbox={'facecolor': 'white', 'alpha': 0.85, 'edgecolor': 'none'})
plt.xlabel('Count contrast')
plt.ylabel('# of occurrences')
plt.savefig(this_dir + '/histogram.png')
if abs(erf(1/sqrt(2)) - sum(abs(counts_contrast) < σ)/counts_contrast.size) > 0.1:
    abort('The particle distribution does not seem to be Gaussian.\n'
          'See "{}".'.format(this_dir + '/histogram.png'))

# Load in σ
powerspec_filename = '{}/{}_{}'.format(this_dir,
                                       output_bases['powerspec'],
                                       os.path.basename(os.path.splitext(initial_conditions)[0]))
with open(powerspec_filename, encoding='utf-8') as powespec_file:
    search = None
    while not search:
        header = powespec_file.readline()
        search = re.search(
            ''.join([
                'σ',
                unicode_subscript(f'{tophat/units.Mpc:.2g}'),
                r' = ([0-9\.e+-]*)',
            ]),
            header,
        )
σ_concept = float(search.group(1))

# Do the σ from CO𝘕CEPT agree with the one computed via the cubic boxes?
rel_tol = 3e-2
if not isclose(σ, σ_concept, rel_tol=rel_tol):
    abort('The rms density variation σ = {:.6g} from "{}" do not agree with direct computation '
           '({:.6g}). The power spectrum from which σ is calulated is plotted in "{}"'
           .format(σ_concept, powerspec_filename, σ, powerspec_filename + '.png'))

# Check the scaling of the power spectrum against the boxsize.
# Doubling the boxsize (and expanding the particle configuration with it)
# should result in k being halved and the power being multiplied by 2³.
powerspec_filename_single_boxsize = powerspec_filename
powerspec_filename_double_boxsize = '{}_double_boxsize'.format(powerspec_filename)
(k_single_boxsize,
 modes,
 power_single_boxsize,
 ) = np.loadtxt(powerspec_filename_single_boxsize, unpack=True)
(k_double_boxsize,
 modes,
 power_double_boxsize,
 ) = np.loadtxt(powerspec_filename_double_boxsize, unpack=True)
tol = 1e-4
if not all(np.abs((k_single_boxsize/2 - k_double_boxsize)/k_double_boxsize) < tol):
    abort('Bad scaling of k against the boxsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_single_boxsize, powerspec_filename_double_boxsize)
          )
if not all(np.abs((power_single_boxsize*2**3 - power_double_boxsize)/power_double_boxsize) < tol):
    abort('Bad scaling of power against the boxsize. '
          'The compared power spectra are plotted in "{}.png" and "{}.png"'
          .format(powerspec_filename_single_boxsize, powerspec_filename_double_boxsize)
          )

# Check the scaling of the power spectrum against the gridsize.
# Halving the gridsize should result in the same min(k), but max(k) should be halved.
# Also, halving the gridsize should not affect the power at a given k.
powerspec_filename_whole_gridsize = powerspec_filename
powerspec_filename_half_gridsize = f'{powerspec_filename}_half_gridsize'
k_whole_gridsize, power_whole_gridsize = k_single_boxsize, power_single_boxsize
k_half_gridsize, modes, power_half_gridsize = np.loadtxt(
    powerspec_filename_half_gridsize,
    unpack=True,
)
if k_whole_gridsize[0] != k_half_gridsize[0]:
    abort(
        f'The smallest k value should not depend on the grid size. '
        f'The compared power spectra are plotted in '
        f'"{powerspec_filename_whole_gridsize}.png" and '
        f'"{powerspec_filename_half_gridsize}.png".'
    )
if not isclose(k_whole_gridsize[-1], 2*k_half_gridsize[-1], rel_tol=rel_tol):
    abort(
        f'The largest k value should be proportional to the grid size. '
        f'The compared power spectra are plotted in '
        f'"{powerspec_filename_whole_gridsize}.png" and '
        f'"{powerspec_filename_half_gridsize}.png".'
    )
# New, interpolated (k, power) of whole_gridsize onto the k's of half_gridsize
k_whole_gridsize_interp = k_half_gridsize
power_whole_gridsize_interp = exp(np.interp(
    log(k_half_gridsize), log(k_whole_gridsize), log(power_whole_gridsize)
))
# Compare the powers(k) below the lowest Nyquist frequency
k_max = k_half_gridsize[-1]/1.5
power_half_gridsize_firstpart = power_half_gridsize[k_half_gridsize < k_max]
power_whole_gridsize_interp_firstpart = power_whole_gridsize_interp[
    k_whole_gridsize_interp < k_max
]
if not all(
    abs(
        (power_whole_gridsize_interp_firstpart - power_half_gridsize_firstpart)
        /power_half_gridsize_firstpart
    ) < rel_tol
):
    abort(
        f'Bad scaling of power against the gridsize. '
        f'The compared power spectra are plotted in '
        f'"{powerspec_filename_whole_gridsize}.png" and '
        f'"{powerspec_filename_half_gridsize}.png".'
    )

# Done analyzing
masterprint('done')
