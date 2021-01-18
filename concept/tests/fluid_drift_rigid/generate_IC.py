# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from linear import random_uniform
from species import Component
from snapshot import save

# Create the fluid.
# This is a global sine wave along the x-direction,
# moving with the speed "speed" along the x-direction,
# the speed 0 along the y-direction and a random speed
# in the z-direction.
components = []
gridsize = 64
speed = a_begin**2*boxsize/(0.5*units.Gyr)
N = gridsize                   # Number of particles
N_fluidelements = gridsize**3  # Number of fluid elements
Vcell = (boxsize/gridsize)**3
mass_tot = ρ_mbar*boxsize**3
mass_fluid_tot = mass_particles_tot = 0.5*mass_tot
mass_fluid = mass_fluid_tot/N_fluidelements  # Mass of each fluid element
mass_particles = mass_particles_tot/N        # Mass of each particle
component = Component('test fluid', 'matter', gridsize=gridsize)
ϱ = empty([gridsize]*3, dtype=float)
for i in range(gridsize):
    ϱ[i, :, :] = 2 + sin(2*π*i/gridsize)  # Unitless
ϱ /= sum(ϱ)                               # Normalize
ϱ *= mass_fluid_tot/Vcell                 # Apply units
component.populate(ϱ,                                'ϱ'   )
component.populate(ϱ*speed,                          'J', 0)
component.populate(zeros([gridsize]*3, dtype=float), 'J', 1)
component.populate(ϱ*speed*random_uniform(-1, 1),    'J', 2)
components.append(component)

# Create the particles.
# These are N particles strewn uniformly over the x-axis, with a velocity
# only in the x-direction. These should then remain equally spaced
# in the x-direction throughout time, each following a specific fluid element.
# The y-positions are given by a sine (in order to match
# these against the fluid density profile) and the z-positions are random.
component = Component('control particles', 'matter', N=N, mass=mass_particles)
offset = 0.5*boxsize
A = 0.4*boxsize
component.populate(linspace(0, boxsize, N, endpoint=False),     'posx')
component.populate(offset + A*sin([2*π*i/N for i in range(N)]), 'posy')
component.populate(random_uniform(0, boxsize, size=N),          'posz')
component.populate(ones(N)*speed*mass_particles,                'momx')
component.populate(zeros(N, dtype=float),                       'momy')
component.populate(zeros(N, dtype=float),                       'momz')
components.append(component)

# Save snapshot
save(components, initial_conditions)
