# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)
masterprint(f'Analysing {this_test} data ...')



############################################
# Upstream/global (power spectrum) subtest #
############################################
subtest = 'upstream_global'
subtest_dir = f'{this_dir}/{subtest}'
masterprint(f'Analysing {subtest} data ...')
# Read in power spectra
powerspecs = {}
for f in glob(f'{subtest_dir}/powerspec*'):
    if f.endswith('.png'):
        continue
    gridsize = int(os.path.basename(f).split('_')[1])
    powerspecs[gridsize] = data = {}
    data['k'], data['modes'], data['P'] = np.loadtxt(f, unpack=True)
powerspecs = {gridsize: powerspecs[gridsize] for gridsize in sorted(powerspecs.keys())}
# Two power spectra should be identical up until the Nyquist frequency
# of the spectrum with lower grid size.
# Check that this is so for the power spectra with the
# small and the medium grid sizes, and for the power spectra
# with the medium and the large grid sizes.
def check(data_small, data_large, gridsize_small):
    nyquist_small = gridsize_small//2
    k_nyquist_small = 2*π/boxsize*nyquist_small
    n = sum(data_small['k'] < k_nyquist_small) - 1
    for quantity in ('k', 'modes', 'P'):
        if any(data_large[quantity][:n] != data_small[quantity][:n]):
            abort(
                f'Found different {quantity} below Nyquist frequency '
                f'of the power spectra in "{subtest_dir}". '
                f'See the plots in "{subtest_dir}" for a visualization.'
            )
gen = iter(powerspecs.items())
gridsize_small, data_small = next(gen)
gridsize_middl, data_middl = next(gen)
gridsize_large, data_large = next(gen)
check(data_small, data_middl, gridsize_small)
check(data_middl, data_large, gridsize_middl)
# Done analysing this subtest
masterprint('done')



###################################
# Phase shift (2D render) subtest #
###################################
subtest = 'phaseshift'
subtest_dir = f'{this_dir}/{subtest}'
masterprint(f'Analyzing {subtest} data ...')
# Read in 2D renders
render2Ds = {}
for f in glob(f'{subtest_dir}/render2D*'):
    if f.endswith('.png'):
        continue
    gridsize = int(rstrip_exact(os.path.basename(f).split('_')[1], '.hdf5'))
    with open_hdf5(f, mode='r') as hdf5_file:
        render2Ds[gridsize] = hdf5_file['data'][...]
    # Convert from image to data coordinates
    render2Ds[gridsize] = render2Ds[gridsize][::-1, :].transpose()
    # Convert from mass to density
    render2Ds[gridsize] /= (boxsize/gridsize)**2*boxsize
render2Ds = {gridsize: render2Ds[gridsize] for gridsize in sorted(render2Ds.keys())}
# Check that the 2D renders are homogeneous along the y direction
for gridsize, render2D in render2Ds.items():
    for i in range(render2D.shape[0]):
        if len(set(render2D[i, :])) != 1:
            abort(
                f'2D render with grid size {gridsize} is '
                f'inhomogeneous along the y direction. '
                f'See the 2D renders in "{subtest_dir}" for a visualization.'
            )
# Plot the 1D cross sections of the 2D renders
# along the x direction, revealing the sines.
plot_file = f'{subtest_dir}/plot.png'
sines = {}
for (gridsize, render2D), linestyle in zip(render2Ds.items(), ('-', '--', ':')):
    x = (0.5 + arange(gridsize))*boxsize/gridsize
    y = render2D[:, 0]
    sines[gridsize] = (x, y)
    plt.plot(x, y, linestyle, label=f'gridsize {gridsize}')
plt.xlabel(rf'$x\, [\mathrm{{{unit_length}}}]$')
plt.ylabel(
    r'$\rho$ $\mathrm{{[{}\, m_{{\odot}}\, {}^{{-3}}]}}$'
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
plt.legend()
plt.savefig(plot_file)
# Check whether the sines are in phase
rel_tol = 1e-9
extrema = {}
for gridsize, (x, y) in sines.items():
    # Find index of first trough
    safety = 1e-6
    miny = min(y)
    height = max(y) - miny
    for index in range(gridsize):
        if y[index] <= miny*(1 + safety):
            break
    # Store height of troughs and peaks
    troughs = y[index::gridsize//2]
    peaks = y[index+gridsize//4::gridsize//2]
    extrema[gridsize] = (troughs, peaks)
for troughs1, peaks1 in extrema.values():
    break
for gridsize, (troughs, peaks) in extrema.items():
    if (
           not np.allclose(troughs, troughs1, rel_tol, 0)
        or not np.allclose(peaks  , peaks1  , rel_tol, 0)
    ):
        abort(
            f'Erroneous phase shift obtained through grid scaling. '
            f'See the plot "{plot_file}" along with the 2D renders '
            f'in "{subtest_dir}" for a visualization.'
        )
# Done analysing this subtest
masterprint('done')



###################################################
# Upstream/global/downstream (simulation) subtest #
###################################################
subtest = 'upstream_global_downstream'
subtest_dir = f'{this_dir}/{subtest}'
masterprint(f'Analyzing {subtest} data ...')
# Read in power spectra
powerspecs = {}
for f in glob(f'{subtest_dir}/powerspec*'):
    if f.endswith('.png'):
        continue
    gridsize = int(os.path.basename(f).split('_')[1])
    powerspecs[gridsize] = np.loadtxt(f)
powerspecs = {gridsize: powerspecs[gridsize] for gridsize in sorted(powerspecs.keys())}
# Check that the power spectra are identical
data0, data1 = powerspecs.values()
data0[np.isnan(data0)] = -1
data1[np.isnan(data1)] = -1
if np.any(data0 != data1):
    abort(
        f'The two power spectra within "{subtest_dir}" '
        f'should be identical but are not. '
        f'See the plots in "{subtest_dir}" for a visualization.'
    )
# Done analyzing this subtest
masterprint('done')



############################################
# Number of processes (simulation) subtest #
############################################
subtest = 'nprocs'
subtest_dir = f'{this_dir}/{subtest}'
masterprint(f'Analyzing {subtest} data ...')
# Read in power spectra
powerspecs = {}
for f in glob(f'{subtest_dir}/*/powerspec*'):
    if f.endswith('.png'):
        continue
    n = int(os.path.basename(os.path.dirname(f)))
    powerspecs[n] = np.loadtxt(f)
powerspecs = {n: powerspecs[n] for n in sorted(powerspecs.keys())}
# Check that the power spectra are identical
for powerspec1 in powerspecs.values():
    break
powerspec1[np.isnan(powerspec1)] = -1
for powerspec in powerspecs.values():
    powerspec[np.isnan(powerspec)] = -1
    if np.any(powerspec != powerspec1):
        abort(
            f'The power spectra of the different subdirectories '
            f'within "{subtest_dir}" should all be identical but are not. '
            f'See the plots in the subdirectories of "{subtest_dir}" '
            f'for a visualization.'
        )
# Done analyzing this subtest
masterprint('done')



# Done analyzing
masterprint('done')

