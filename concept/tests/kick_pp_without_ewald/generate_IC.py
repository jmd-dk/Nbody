# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create the particles.
# It is important that no interparticle separation exceeds boxsize/2 in
# any direction, as the nearest particle image in all cases must be the
# actual particle itself.
N = 8
mass = ρ_mbar*boxsize**3/N
component = Component('GADGET halos', 'matter', N=N, mass=mass)
component.populate(asarray([0.26]*4 + [0.74]*4       )*boxsize, 'posx')
component.populate(asarray([0.25, 0.25, 0.75, 0.75]*2)*boxsize, 'posy')
component.populate(asarray([0.25, 0.75, 0.75, 0.25]*2)*boxsize, 'posz')
component.populate(zeros(N, dtype=float), 'momx')
component.populate(zeros(N, dtype=float), 'momy')
component.populate(zeros(N, dtype=float), 'momz')

# Save snapshot
save(component, initial_conditions)
