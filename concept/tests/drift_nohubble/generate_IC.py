# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create the particle
N = 1
mass = ρ_mbar*boxsize**3/N
particles = Component('test particles', 'matter', N=N, mass=mass)
particles.populate(asarray([0.75])*boxsize, 'posx')
particles.populate(asarray([0.75])*boxsize, 'posy')
particles.populate(asarray([0.75])*boxsize, 'posz')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momx')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momy')
particles.populate(ones(N)*boxsize/(10*units.Gyr)*mass, 'momz')

# Save snapshot
save(particles, initial_conditions)
