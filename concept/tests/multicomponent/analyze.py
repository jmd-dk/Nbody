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
def get_directory(subtest, ncomponents, nprocs, subtiling=None):
    directory = f'{this_dir}/{subtest}/output_{ncomponents}components_{nprocs}procs'
    if subtiling is not None:
        directory += f'_subtiling{subtiling}'
    return directory
def load_results(subtest):
    ncomponents_values = set()
    nprocs_values = set()
    subtiling_values = set()
    directory_pattern = get_directory(subtest, '*', '*',
        subtiling={'domain': None, 'tile': '*'}[subtest],
    )
    for directory in glob(directory_pattern):
        match = re.search(directory_pattern.replace('*', '(.+)'), directory)
        ncomponents_values.add(int(match.group(1)))
        nprocs_values.add(int(match.group(2)))
        if subtest == 'tile':
            subtiling_values.add(int(match.group(3)))
    ncomponents_values = sorted(ncomponents_values)
    nprocs_values = sorted(nprocs_values)
    if subtest == 'domain':
        subtiling_values = [None]
    elif subtest == 'tile':
        subtiling_values = sorted(subtiling_values)
    pos = {}
    times = set()
    lattice_sites = None
    for ncomponents in ncomponents_values:
        for nprocs in nprocs_values:
            for subtiling in subtiling_values:
                directory = get_directory(subtest, ncomponents, nprocs, subtiling)
                for fname in sorted(glob(f'{directory}/snapshot_t=*')):
                    if not fname.endswith('.hdf5'):
                        continue
                    t = float(re.search(rf't=(.+){unit_time}', fname).group(1))
                    times.add(t)
                    snapshot = load(fname)
                    if subtest == 'domain':
                        posx, posy, posz, = [], [], []
                    elif subtest == 'tile':
                        if lattice_sites is None:
                            N = sum([component.N for component in snapshot.components])
                            N_lin = int(round(cbrt(N)))
                            distance = boxsize/N_lin
                            lattice_sites = []
                            for i in range(N_lin):
                                x = (0.5 + i)*distance
                                for j in range(N_lin):
                                    y = (0.5 + j)*distance
                                    for k in range(N_lin):
                                        z = (0.5 + k)*distance
                                        lattice_sites.append((x, y, z))
                            lattice_sites = asarray(lattice_sites)
                        posx = empty(N_lin**3, dtype=float)
                        posy = empty(N_lin**3, dtype=float)
                        posz = empty(N_lin**3, dtype=float)
                        indices_all = []
                    for component in snapshot.components:
                        if component.N == 0:
                            continue
                        if subtest == 'domain':
                            posx += list(component.posx)
                            posy += list(component.posy)
                            posz += list(component.posz)
                        elif subtest == 'tile':
                            # Sort particles according to the lattice
                            indices = []
                            for x, y, z in zip(component.posx, component.posy, component.posz):
                                orders = np.argsort(
                                    sum((asarray((x, y, z)) - lattice_sites)**2, 1)
                                )
                                for order in orders:
                                    if order not in indices_all:
                                        indices    .append(order)
                                        indices_all.append(order)
                                        break
                            indices = asarray(indices)
                            posx[indices] = component.posx
                            posy[indices] = component.posy
                            posz[indices] = component.posz
                    if subtest == 'tile':
                        key = (ncomponents, nprocs, subtiling, t)
                    elif subtest == 'domain':
                        key = (ncomponents, nprocs, t)
                    pos[key] = [asarray(posx), asarray(posy), asarray(posz)]
    times = sorted(times)
    softening_length = is_selected(component, select_softening_length)
    return ncomponents_values, nprocs_values, subtiling_values, times, pos, softening_length
ncomponents_values = {}
nprocs_values      = {}
subtiling_values   = {}
times              = {}
pos                = {}
softening_length   = {}
for subtest in ('domain', 'tile'):
    (
        ncomponents_values[subtest],
        nprocs_values     [subtest],
        subtiling_values  [subtest],
        times             [subtest],
        pos               [subtest],
        softening_length  [subtest],
    ) = load_results(subtest)

# Analyse domain subtest data
masterprint(f'Analyzing {this_test} data ...')
subtest = 'domain'
abs_tol = 1*softening_length[subtest]
for ncomponents in ncomponents_values[subtest]:
    for nprocs in nprocs_values[subtest]:
        directory = get_directory(subtest, ncomponents, nprocs)
        for t in times[subtest]:
            posx, posy, posz = pos[subtest][ncomponents, nprocs, t]
            if not (
                    isclose(mean(posx), 0.5*boxsize, rel_tol=0, abs_tol=abs_tol)
                and isclose(mean(posy), 0.5*boxsize, rel_tol=0, abs_tol=abs_tol)
                and isclose(mean(posz), 0.5*boxsize, rel_tol=0, abs_tol=abs_tol)
            ):
                abort(
                  f'Asymmetric results obtained from running with {ncomponents} '
                  f'particle components on {nprocs} processes, for t >= {t}.\n'
                  f'See the renders in "{directory}" for visualizations.'
                )
            if t < times[subtest][-1]:
                if not (
                        isclose(np.std(posx), np.std(posy), rel_tol=0, abs_tol=abs_tol)
                    and isclose(np.std(posy), np.std(posz), rel_tol=0, abs_tol=abs_tol)
                    and isclose(np.std(posz), np.std(posx), rel_tol=0, abs_tol=abs_tol)
                ):
                    abort(
                      f'Anisotropic results obtained from running with {ncomponents} '
                      f'particle components on {nprocs} processes, for t >= {t}.\n'
                      f'See the renders in "{directory}" for visualizations.'
                    )
            else:
                if not (
                        isclose(np.std(posx), 0, rel_tol=0, abs_tol=abs_tol)
                    and isclose(np.std(posy), 0, rel_tol=0, abs_tol=abs_tol)
                    and isclose(np.std(posz), 0, rel_tol=0, abs_tol=abs_tol)
                ):
                    abort(
                        f'Spherical symmetric collapse did not take place within the '
                        f'correct amount of time when running with {ncomponents} '
                        f'particle components on {nprocs} processes.\n'
                        f'See the renders in "{directory}" for visualizations.'
                    )

# Analyse tile subtest data
subtest = 'tile'
abs_tol = 1e-9*softening_length[subtest]
for ncomponents in ncomponents_values[subtest]:
    for nprocs in nprocs_values[subtest]:
        for subtiling in subtiling_values[subtest]:
            directory = get_directory(subtest, ncomponents, nprocs, subtiling)
            for t in times[subtest]:
                posx, posy, posz = pos[subtest][ncomponents, nprocs, subtiling, t]
                if t == times[subtest][0]:
                    posx_0, posy_0, posz_0 = posx, posy, posz
                    continue
                if not (
                        isclose(max(abs(posx - posx_0)), 0, rel_tol=0, abs_tol=abs_tol)
                    and isclose(max(abs(posy - posy_0)), 0, rel_tol=0, abs_tol=abs_tol)
                    and isclose(max(abs(posz - posz_0)), 0, rel_tol=0, abs_tol=abs_tol)
                ):
                    abort(
                        f'Non-trivial particle evolution resulted from homogeneous and static '
                        f'initial condition when running with {ncomponents} particle '
                        f'components on {nprocs} processes using subtiling = {subtiling}, '
                        f'for t >= {t}.\n'
                        f'See the renders in "{directory}" for visualizations.'
                    )

# Done analyzing
masterprint('done')

