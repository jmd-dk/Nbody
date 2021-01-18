# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in power spectra
powerspecs_all = {'particles': {}, 'fluid': {}}
for kind in ('particles', 'fluid'):
    for fname in sorted(glob('{}/output_{}/powerspec*'.format(this_dir, kind))):
        n = int(re.search('nprocs=(.*)_a=', fname).group(1))
        k, modes, power = np.loadtxt(fname, unpack=True)
        powerspecs_all[kind][n] = (k, power)
k_values = k
n_values = list(powerspecs_all['particles'].keys())

# Begin analysis
masterprint(f'Analysing {this_test} data ...')

# Plot power spectra from realised components
fig_file = this_dir + '/result.png'
fig = plt.figure(figsize=(8, 6))
for kind, powerspecs in powerspecs_all.items():
    for n, (k, power) in powerspecs.items():
        plt.loglog(k, power, alpha=0.7,
                   label='{}, nprocs = {}'.format(kind, n))
# Compute and plot power spectrum from CLASS
a = a_begin
z = 1/a - 1
class_params_specialized = class_params.copy()
class_params_specialized.update({
    'A_s'    : primordial_spectrum['A_s'  ],
    'n_s'    : primordial_spectrum['n_s'  ],
    'alpha_s': primordial_spectrum['α_s'  ],
    'k_pivot': primordial_spectrum['pivot']/units.Mpc**(-1),
    'output' : 'mPk',
    'z_pk'   : str(z),
    'k_output_values': '{}, {}'.format(
        min(k_values)/units.Mpc**(-1),
        max(k_values)/units.Mpc**(-1),
    ),
})
cosmo = Class()
cosmo.set(class_params_specialized)
cosmo.compute()
power_class = asarray([cosmo.pk(k/units.Mpc**(-1), z) for k in k_values])*units.Mpc**3
plt.loglog(k_values, power_class, 'k--', label='CLASS')
plt.xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
plt.ylabel(rf'matter power $\mathrm{{[{unit_length}^3]}}$')
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
plt.savefig(fig_file)

# Compare the power spectra of the realisations with
# the power spectrum from CLASS.
# Ignore the power at the largest scales due to low
# mode count. For particles, further ignore the power at
# the smallest scales due to low particle resolution.
k_min = min(k_values)
k_max = max(k_values)
masks = {
    'particles': np.logical_and(
        k_values > 5*k_min,
        k_values < 0.6*k_max,
    ),
    'fluid': np.logical_and(
        k_values > 5*k_min,
        k_values < k_max,
    ),
}
rel_tol = 0.02
for kind in ('particles', 'fluid'):
    for n in n_values:
        k, power = powerspecs_all[kind][n]
        power_trimmed = power[masks[kind]]
        power_class_trimmed = power_class[masks[kind]]
        rel_realisation_noise = mean(
            abs((power_trimmed - power_class_trimmed)/power_class_trimmed)
        )
        if rel_realisation_noise > rel_tol:
            abort(
                f'Power spectrum of realised matter {kind} with {n} processes '
                f'disagree with that of CLASS.\n'
                f'See "{fig_file}" for a visualization.'
            )

# Done analysing
masterprint('done')
