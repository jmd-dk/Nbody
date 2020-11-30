# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create homogeneous matter distribution perturbed
# with a global sine wave along the x-direction.
gridsize = 24
component = Component('matter', 'matter', gridsize=gridsize)
x = (0.5 + arange(gridsize))*boxsize/gridsize
f = gridsize//3//2
y = cos(f*π/boxsize*x + π/4)
ϱ = empty([gridsize]*3, dtype=float)
for i in range(gridsize):
    ϱ[i, :, :] = 2 + y[i]  # Unitless
ϱ /= sum(ϱ)                # Normalize
ϱ *= ρ_mbar*gridsize**3    # Apply units
component.populate(ϱ, 'ϱ')
for index in component.J.multi_indices:
    component.populate(zeros([gridsize]*3, dtype=float), 'J', index)

# Save snapshot
save(component, output_dirs['snapshot'] + '/sine.hdf5')

