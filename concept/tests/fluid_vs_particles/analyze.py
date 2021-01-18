# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from snapshot import load
import species

# Absolute path and name of the directory of this file
this_dir  = os.path.dirname(os.path.realpath(__file__))
this_test = os.path.basename(this_dir)

# Read in data from the snapshots
species.allow_similarly_named_components = True
fluids = {'particles': [], 'fluid': []}
a = []
for kind in ('particles', 'fluid'):
    if kind == 'particles':
        regex = '{}/output_{}/snapshot_a=*_converted*'.format(this_dir, kind)
    elif kind == 'fluid':
        regex = '{}/output_{}/snapshot_a=*'.format(this_dir, kind)
    for fname in sorted(glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        snapshot = load(fname, compare_params=False)
        fluids[kind].append(snapshot.components[0])
        if kind == 'particles':
            a.append(snapshot.params['a'])
gridsize = fluids['particles'][0].gridsize
# Sort data chronologically
order = np.argsort(a)
a = [a[o] for o in order]
for kind in ('particles', 'fluid'):
    fluids[kind] = [fluids[kind][o] for o in order]

# Begin analysis
masterprint('Analysing {} data ...'.format(this_test))

# Load in power spectra
powerspecs = {'particles': [], 'fluid': []}
powerspec_filenames = {'particles': [], 'fluid': []}
for kind in ('particles', 'fluid'):
    regex = '{}/output_{}/powerspec_a=*[!.png]'.format(this_dir, kind)
    for fname in sorted(glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        powerspec_filenames[kind].append(fname)
        k, power, σ = np.loadtxt(fname, unpack=True)
        powerspecs[kind].append(power)
    # Sort data chronologically
    powerspec_filenames[kind] = [powerspec_filenames[kind][o] for o in order]
    powerspecs[kind] = [powerspecs[kind][o] for o in order]

# Compare the particle and fluid power spectra.
# Because of the smoothing effects (vacuum corrections) on the fluid,
# only the large scale power should be comparable between the
# particle and fluid power spectra.
n_points = 5
rel_tol = 2.0e-1
for (a_i,
     power_particles,
     power_fluid,
     fname_particles,
     fname_fluid,
     ) in zip(a,
              powerspecs['particles'],
              powerspecs['fluid'],
              powerspec_filenames['particles'],
              powerspec_filenames['fluid'],
              ):
    # Compare the first n points of the two power spectra
    for i in range(n_points):
        if not isclose(power_particles[i], power_fluid[i], rel_tol=rel_tol):
            abort('Large-scale power of particle and fluid simulations disagree at a = {}.\n'
                  'See "{}.png" and "{}.png" for a visualization.'
                  .format(a_i, fname_particles, fname_fluid))

# Compare the biggest halos of the particle and the fluid simulation
def find_biggest_halo(component):
    ϱ = component.ϱ.grid_noghosts[:-1, :-1, :-1]
    indices = np.unravel_index(np.argmax(ϱ), ϱ.shape)
    δ_halo_boundary = 0.5
    halo = zeros(ϱ.shape[0]//2, dtype=float)
    for r in range(ϱ.shape[0]//2):
        count = 0
        for         i in range(-r, r + 1):
            for     j in range(-r, r + 1):
                for k in range(-r, r + 1):
                    if i**2 + j**2 + k**2 > r**2:
                        continue
                    count += 1
                    I = mod(i + indices[0], ϱ.shape[0])
                    J = mod(j + indices[1], ϱ.shape[1])
                    K = mod(k + indices[2], ϱ.shape[2])
                    halo[r] += ϱ[I, J, K]
                    ϱ[I, J, K] = 0
        halo[r] = halo[r]/(count*ρ_mbar) - 1
        if halo[r] < δ_halo_boundary:
            return indices, r
render3D_filenames = {'particles': [], 'fluid': []}
for kind in ('particles', 'fluid'):
    regex = '{}/output_{}/render3D_a=*'.format(this_dir, kind)
    for fname in sorted(glob(regex),
                        key=lambda s: s[(s.index('=') + 1):]):
        render3D_filenames[kind].append(fname)
    # Sort filenames chronologically
    render3D_filenames[kind] = [render3D_filenames[kind][o] for o in order]
rel_tol = 0.1
abs_tol = 2 + 0.02*gridsize
N_largest_halos = 5
for (a_i,
     particles_component,
     fluid_component,
     fname_particles,
     fname_fluid,
     ) in zip(a,
              fluids['particles'],
              fluids['fluid'],
              render3D_filenames['particles'],
              render3D_filenames['fluid'],
              ):
    # Find the largest halo in the particle simulation
    indices_particles, r_particles = find_biggest_halo(particles_component)
    # Find the same halo in the fluid simulation.
    # This should be the largest halo as well,
    # but due to smoothing errors it might not be quite the largest one.
    # Search the N_largest_halos largest halos and use the one closest
    # to the one in the particle simulation.
    indices_fluid = []
    r_fluid = []
    for _ in range(N_largest_halos):
        halo_info = find_biggest_halo(fluid_component)
        indices_fluid.append(halo_info[0])
        r_fluid.append(halo_info[1])
    distances = []
    for i in range(N_largest_halos):
        d2 = 0
        for dim in range(3):
            d2 += min([
                abs(indices_particles[dim] - indices_fluid[i][dim]),
                abs(indices_particles[dim] - indices_fluid[i][dim] + gridsize),
                abs(indices_particles[dim] - indices_fluid[i][dim] - gridsize),
            ])**2
        distances.append(sqrt(d2))
    index = np.argmin(distances)
    indices_fluid = indices_fluid[index]
    r_fluid = r_fluid[index]
    distance = distances[index]
    # Compare sizes
    if not isclose(r_particles, r_fluid, rel_tol=rel_tol, abs_tol=abs_tol):
        if r_particles > r_fluid:
            abort('At a = {}, the largest halo in the particle simulation is significantly larger '
                  'than the largest halo in the fluid simulation.\n'
                  'See "{}" and "{}" for a visualization.'
                  .format(a_i, fname_particles, fname_fluid))
        else:
            abort('At a = {}, the largest halo in the fluid simulation is significantly larger '
                  'than the largest halo in the particle simulation.\n'
                  'See "{}" and "{}" for a visualization.'
                  .format(a_i, fname_fluid, fname_particles))
    # The largest halo should be in the same location in the two simulations
    if distance > abs_tol:
        abort('At a = {}, the largest halo of the particle simulation does not coincide with '
              'the largest halo of the fluid simulation.\n'
              'See "{}" and "{}" for a visualization.'
              .format(a_i, fname_particles, fname_fluid))

# Done analysing
masterprint('done')
