# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create the particles
N = 4
mass = ρ_mbar*boxsize**3/N
particles = Component('GADGET-2 halos', 'matter', N=N, mass=mass)
particles.populate(asarray([0.1]*N)*boxsize, 'posx')
particles.populate(asarray([0.25, 0.25, 0.75, 0.75])*boxsize, 'posy')
particles.populate(asarray([0.25, 0.75, 0.75, 0.25])*boxsize, 'posz')
particles.populate(ones(N)*100*units.km/units.s*mass, 'momx')
particles.populate(zeros(N, dtype=float), 'momy')
particles.populate(zeros(N, dtype=float), 'momz')

# Save snapshot
save(particles, initial_conditions)
