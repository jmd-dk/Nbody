# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from linear import random_uniform
from species import Component
from snapshot import save

# Create a global sine wave along the x-direction,
# traversing the box in the x-direction over 10 Gyr.
# The y-velocity is 0 and the z-velocity is random.
gridsize = 64
Vcell = (boxsize/gridsize)**3
speed = boxsize/(10*units.Gyr)
N = gridsize**3
component = Component('test fluid', 'matter', gridsize=gridsize)
ϱ = empty([gridsize]*3)
for i in range(gridsize):
    ϱ[i, :, :] = 2 + np.sin(2*π*i/gridsize)  # Unitless
ϱ /= sum(ϱ)                                  # Normalize
ϱ *= ρ_mbar*gridsize**3                      # Apply units
component.populate(ϱ,                             'ϱ'   )
component.populate(ϱ*speed,                       'J', 0)
component.populate(zeros([gridsize]*3),           'J', 1)
component.populate(ϱ*speed*random_uniform(-1, 1), 'J', 2)

# Save snapshot
save(component, initial_conditions)
