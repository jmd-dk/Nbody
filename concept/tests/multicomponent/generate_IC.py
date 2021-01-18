# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Special parameters
subtest = user_params['_subtest']
T = user_params['_T']
ncomponents = user_params['_ncomponents']

# Define particle positions
if subtest == 'domain':
    # 6 particles placed spherically symmetric around the box centre
    distance = 0.4*boxsize
    pos = []
    for dim in range(3):
        for sign in (-1, +1):
            pos.append(0.5*boxsize*ones(3) + np.roll([sign*distance, 0, 0], dim))
elif subtest == 'tile':
    # 12³ particles placed in a cubic lattice
    N_lin = 12
    distance = boxsize/N_lin
    pos = []
    for i in range(N_lin):
        x = (0.5 + i)*distance
        for j in range(N_lin):
            y = (0.5 + j)*distance
            for k in range(N_lin):
                z = (0.5 + k)*distance
                pos.append((x, y, z))

# Define the mass such that the six symmetrically placed
# particles will collide after time T.
mass = π**2/((2 + 8*sqrt(2))*G_Newton)*distance**3/T**2

# For the stationary tile test, increase the particle mass
# in order to enhance any possible erroneous evolution.
if subtest == 'tile':
    mass *= 10

# Distribute particles into multiple components
posx = collections.defaultdict(list)
posy = collections.defaultdict(list)
posz = collections.defaultdict(list)
for n, (x, y, z) in enumerate(pos):
    n %= ncomponents
    posx[n].append(x)
    posy[n].append(y)
    posz[n].append(z)
components = []
for n in range(ncomponents):
    N = len(posx[n])
    component = Component(f'component{n}', 'matter', N=N, mass=mass)
    component.populate(asarray(posx[n]), 'posx')
    component.populate(asarray(posy[n]), 'posy')
    component.populate(asarray(posz[n]), 'posz')
    component.populate(zeros(N, dtype=float), 'momx')
    component.populate(zeros(N, dtype=float), 'momy')
    component.populate(zeros(N, dtype=float), 'momz')
    components.append(component)

# Save snapshot
save(components, initial_conditions)

