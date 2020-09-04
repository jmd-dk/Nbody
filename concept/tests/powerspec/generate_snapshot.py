# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from linear import random_gaussian
from species import Component
from snapshot import save

# Create close to homogeneous particles
N_lin = 128
N = N_lin**3
mass = ρ_mbar*boxsize**3/N
component = Component('test particles', 'matter', N=N, mass=mass)
posx = empty(N)
posy = empty(N)
posz = empty(N)
momx = zeros(N)
momy = zeros(N)
momz = zeros(N)
count = 0
boxsize_over_N_lin = boxsize/N_lin
for i in range(N_lin):
    x = i*boxsize_over_N_lin
    for j in range(N_lin):
        y = j*boxsize_over_N_lin
        for k in range(N_lin):
            z = k*boxsize_over_N_lin
            posx[count] = mod(x + random_gaussian(R_tophat), boxsize)
            posy[count] = mod(y + random_gaussian(R_tophat), boxsize)
            posz[count] = mod(z + random_gaussian(R_tophat), boxsize)
            count += 1
component.populate(posx, 'posx')
component.populate(posy, 'posy')
component.populate(posz, 'posz')
component.populate(momx, 'momx')
component.populate(momy, 'momy')
component.populate(momz, 'momz')

# Save snapshot
save(component, initial_conditions)

# Expand particle locations by a factor of 2
posx = component.posx
posy = component.posy
posz = component.posz
for i in range(N):
    posx[i] *= 2
    posy[i] *= 2
    posz[i] *= 2

# Save another snapshot, this time with an enlarged boxsize,
# matching the expanded particle locations.
save(
    component,
    '{}_double_boxsize{}'.format(*os.path.splitext(initial_conditions)),
    {'boxsize': 2*boxsize},
)
