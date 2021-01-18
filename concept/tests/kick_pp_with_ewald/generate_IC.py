# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create the particles
N = 8
mass = ρ_mbar*boxsize**3/N
components = Component('GADGET-2 halos', 'matter', N=N, mass=mass)
d = 0.005
components.populate(asarray([0.25 - d]*4 + [0.75 + d]*4)*boxsize, 'posx')
components.populate(asarray([0.25, 0.25, 0.75, 0.75]*2 )*boxsize, 'posy')
components.populate(asarray([0.25, 0.75, 0.75, 0.25]*2 )*boxsize, 'posz')
components.populate(zeros(N, dtype=float), 'momx')
components.populate(zeros(N, dtype=float), 'momy')
components.populate(zeros(N, dtype=float), 'momz')

# Save snapshot
save(components, initial_conditions)
