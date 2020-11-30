# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from species import Component
from snapshot import save

# Create stationary, homogeneous matter distribution,
# perturbed with global, stationary sine wave along
# the x-direction.
w  = user_params['_w']
ρ0 = user_params['_ρ0']
A  = user_params['_A']
σ  = user_params['_σ']
gridsize = 4*16  # Should be a multiple of 4
component = Component('test fluid', 'matter', gridsize=gridsize, boltzmann_order=2)
ρ = empty([gridsize]*3, dtype=float)
for i in range(gridsize):
    x = boxsize*i/gridsize
    ρ[i, :, :] = ρ0 + A*sin(x/boxsize*2*π)
component.populate(ρ, 'ϱ')
for multi_index in component.J.multi_indices:
    component.populate(zeros([gridsize]*3, dtype=float), 'J', multi_index)
for multi_index in component.ς.multi_indices:
    component.populate(ones([gridsize]*3)*ρ*(1 + w)*σ, 'ς', multi_index)

# Save snapshot
save(component, initial_conditions)
