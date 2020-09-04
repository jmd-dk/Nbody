# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

masterprint(f'Analyzing {this_test} data ...')

# Read in all power spectra. Note that the neutrino
# spectra used a different k grid than the matter.
filenames = sorted([
    os.path.basename(filename) for filename in
    glob(f'{this_dir}/output_massless/powerspec_a=*')
    if not filename.endswith('.png')
])
scalefactors = asarray([
    float(re.search('a=(.*)', filename).group(1))
    for filename in filenames
])
km, kν = np.loadtxt(
    f'{this_dir}/output_massless/{filenames[0]}',
    usecols=(0, 6),
    unpack=True,
)
kν = kν[~np.isnan(kν)]
km = km[km <= kν[-1]]
ktot = km
Pm = {}
Pν = {}
Ptot = {}
for sim in ('massless', 'massive_linear', 'massive_nonlinear'):
    for a, filename in zip(scalefactors, filenames):
        Pm  [sim, 'sim', a], Pm  [sim, 'lin', a], \
        Ptot[sim, 'sim', a], Ptot[sim, 'lin', a], \
        Pν  [sim, 'sim', a], Pν  [sim, 'lin', a] = np.loadtxt(
            f'{this_dir}/output_{sim}/{filename}',
            usecols=(2, 3, 4, 5, 8, 9),
            unpack=True,
        )
        for kind in ('sim', 'lin'):
            Pm  [sim, kind, a] = Pm  [sim, kind, a][:len(km  )]
            Pν  [sim, kind, a] = Pν  [sim, kind, a][:len(kν  )]
            Ptot[sim, kind, a] = Ptot[sim, kind, a][:len(ktot)]

# Plot absolute neutrino power spectra
fig, axes = plt.subplots(len(scalefactors), sharex=True, figsize=(6.4, 4.8*0.5*len(scalefactors)))
sim = 'massive_nonlinear'
for a, ax in zip(scalefactors, axes):
    ax.loglog(
        kν,
        kν**3*Pν[sim, 'sim', a],
        '-',
        label=r'CO$N$CEPT (non-linear $\nu$)',
    )
    ax.loglog(
        kν,
        kν**3*Pν[sim, 'lin', a],
        '--k',
        label=r'CLASS',
    )
    ax.set_ylabel(rf'$k^3 P_{{\nu}}$')
    ax.text(0.5, 0.85, f'$a = {a}$',
        horizontalalignment='center',
        verticalalignment='center',
        fontsize=12,
        transform=ax.transAxes,
    )
axes[0].legend(loc='best').get_frame().set_alpha(0.7)
axes[-1].set_xlabel(rf'$k\, [\mathrm{{{unit_length}}}^{{-1}}]$')
plt.tight_layout(h_pad=0)
fig.subplots_adjust(hspace=0)
fig_file = f'{this_dir}/result_abs.png'
plt.savefig(fig_file)

# Check that the linear and non-linear neutrino
# power spectra are very similar on large scales,
# for all times. We access this by comparing the
# difference between the linear and non-linear power
# spectra, compared to the difference at the initial time.
k_min = kν[1]  # Avoid the first couple of points
k_max = 0.06*units.Mpc**(-1)
err_allowed_fac = 2
mask = np.logical_and(k_min <= kν, kν <= k_max)
for a in scalefactors:
    error = np.abs(Pν[sim, 'lin', a][mask]/Pν[sim, 'sim', a][mask] - 1)
    if a == a_begin:
        error_init = error
        continue
    if np.any(error > err_allowed_fac*error_init):
        abort(
            f'The linear and non-linear neutrino power spectra does not match '
            f'at large scales at a = {a}.\n'
            f'See "{fig_file}" for a visualization.'
        )

# Compute the relative (massive vs. massless) total power spectra,
# find the position of the non-linear dip and
# get the value of the relative power spectra at this dip,
# all at the present time.
rel_linear    = Ptot['massive_linear'   , 'sim', 1]/Ptot['massless', 'sim', 1] - 1
rel_nonlinear = Ptot['massive_nonlinear', 'sim', 1]/Ptot['massless', 'sim', 1] - 1
rel_class     = Ptot['massive_linear'   , 'lin', 1]/Ptot['massless', 'lin', 1] - 1
rel_min_index = np.argmin(rel_linear)
k_dip = ktot[rel_min_index]
suppression_linear    = np.mean(rel_linear   [rel_min_index - 1:rel_min_index + 2])
suppression_nonlinear = np.mean(rel_nonlinear[rel_min_index - 1:rel_min_index + 2])
suppression_class     = np.mean(rel_class    [rel_min_index - 1:rel_min_index + 2])

# Plot relative total power spectra
plt.figure()
plt.semilogx(
    ktot,
    100*rel_nonlinear,
    '-',
    label=r'CO$N$CEPT (non-linear $\nu$)',
)
plt.semilogx(
    ktot,
    100*rel_linear,
    '--',
    label=r'CO$N$CEPT (linear $\nu$)',
)
plt.semilogx(
    ktot,
    100*rel_class,
    '--k',
    label='CLASS',
)
plt.semilogx(
    ktot[asarray([rel_min_index - 2, rel_min_index + 3])],
    [100*suppression_linear]*2,
    '-k',
    alpha=0.7,
)
plt.xlabel(rf'$k\,[\mathrm{{{unit_length}}}^{{-1}}]$')
Σmν = float(class_params['m_ncdm'])*float(class_params['deg_ncdm'])
plt.ylabel(
    rf'$'
    rf'P_{{\mathrm{{tot}}}}^{{\Sigma m_{{\nu}} = {Σmν:g}\mathrm{{eV}}}}'
    rf'/'
    rf'P_{{\mathrm{{tot}}}}^{{\Sigma m_{{\nu}} = 0.0    \mathrm{{eV}}}}'
    rf'- 1'
    rf'\, [\%]'
    rf'$'
)
plt.title(f'$a = 1$')
plt.legend(loc='best').get_frame().set_alpha(0.7)
plt.tight_layout()
fig_file = f'{this_dir}/result_rel.png'
plt.savefig(fig_file)

# Check whether the CLASS supression around the non-linear dip
# is about the theoretically predicted value.
rel_tol = 0.10
suppression_class_theoretical = -8*Ων/(Ωcdm + Ωb + Ων)
if not isclose(
    suppression_class,
    suppression_class_theoretical,
    rel_tol=rel_tol,
):
    abort(
        f'The total power suppression predicted by CLASS ({suppression_class*100:.2f}%) '
        f'at the non-linear dip (k ≈ {k_dip}/{unit_length}) is far from being equal '
        f'to the theoretical prediction -8*Ων/(Ωcdm + Ωb + Ων) '
        f'= {suppression_class_theoretical*100:.2f}%.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Check whether the CO𝘕CEPT supression around the non-linear dip
# is about that found in https://arxiv.org/pdf/0802.3700.pdf .
rel_tol = 0.11
suppression_nonlinear_theoretical = -9.8*Ων/(Ωcdm + Ωb + Ων)
if not isclose(
    suppression_linear,
    suppression_nonlinear_theoretical,
    rel_tol=rel_tol,
):
    abort(
        f'The total power suppression predicted by CO𝘕CEPT ({suppression_linear*100:.2f}%) '
        f'at the non-linear dip (k ≈ {k_dip}/{unit_length}) is far from being equal '
        f'to the prediction -9.8*Ων/(Ωcdm + Ωb + Ων) '
        f'= {suppression_nonlinear_theoretical*100:.2f}%.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Check that the supression in the linear and non-linear neutrino
# runs are very similar.
rel_tol = 0.04
if not all(isclose(
    rel_linear_i,
    rel_nonlinear_i,
    rel_tol=rel_tol,
) for rel_linear_i, rel_nonlinear_i in zip(rel_linear, rel_nonlinear)):
    abort(
        f'The total power suppression in the CO𝘕CEPT runs using linear and non-linear '
        f'neutrinos should be more similar than they are.\n'
        f'See "{fig_file}" for a visualization.'
    )

# Done analyzing relative total power spectra
masterprint('done')
