# This file has to be run in pure Python mode!

# Imports from the CO𝘕CEPT code
from commons import *
from linear import random_uniform
from species import Component
from snapshot import save

# Create a global sine wave along the x-direction:
# ϱ(x, y, z) = ϱ(x) ∝ (1.05 + sin(2*π*x/boxsize)).
# The function ∫_{x1}^{x2}dxϱ(x)
ᔑdxϱ = lambda x1, x2: 1.05*(x2 - x1) + boxsize/π*(cos(π*x1/boxsize)**2 - cos(π*x2/boxsize)**2)
# Function which finds x2 in ∫_{x1}^{x2}dxϱ(x) == mass_unitless
def binary_search(x, mass_unitless, x_lower=None, x_upper=None):
    # Find ᔑdxϱ(x, x_next) == mass_unitless
    if x_lower is None:
        x_lower = x
    if x_upper is None:
        x_upper = boxsize
    x_next = 0.5*(x_lower + x_upper)
    mass_unitless_test = ᔑdxϱ(x, x_next)
    if isclose(mass_unitless_test, mass_unitless, rel_tol=1e-12):
        return x_next
    elif mass_unitless_test < mass_unitless:
        x_lower = x_next
    elif mass_unitless_test > mass_unitless:
        x_upper = x_next
    return binary_search(x, mass_unitless, x_lower=x_lower, x_upper=x_upper)
# Compute positions of particles
Nx = user_params['_size']*20
Ny = Nz = user_params['_size']
N = Nx*Ny*Nz
mass = ρ_mbar*boxsize**3/N
mass_unitless_tot = ᔑdxϱ(0, boxsize)
mass_unitless = mass_unitless_tot/Nx
posx = zeros(N)
posy = zeros(N)
posz = zeros(N)
momx = -ones(N)*mass*boxsize/(60*units.Gyr)
momy =  ones(N)*mass*boxsize/(60*units.Gyr)*random_uniform(-1, 1)
momz =  ones(N)*mass*boxsize/(60*units.Gyr)*random_uniform(-1, 1)
x = 0
count = 0
for i in range(Nx):
    for j in range(Ny):
        y = j/Ny*boxsize
        for k in range(Nz):
            z = k/Nz*boxsize
            posx[count] = x
            posy[count] = y
            posz[count] = z
            # Make the momenta diverge from the point of lowest density
            if isclose(x, 0.75*boxsize, abs_tol=boxsize/Nx):
                momx[count] -= 0.01*mass*boxsize/(100*units.Gyr)
            elif abs(x - 0.75*boxsize) < 0.3*boxsize:
                if x <= 0.75*boxsize:
                    fac = -0.5*exp(-0.07*(x - 0.75*boxsize + 0.1*boxsize)**2)
                else:
                    fac = +0.5*exp(-0.07*(x - 0.75*boxsize - 0.1*boxsize)**2)
                momx[count] += fac*mass*boxsize/(100*units.Gyr)
            count += 1
    if i < Nx - 1:
        x = binary_search(x, mass_unitless)

# Instantiate particles
particles = Component('control particles', 'matter', N=N, mass=mass)
particles.populate(posx, 'posx')
particles.populate(posy, 'posy')
particles.populate(posz, 'posz')
particles.populate(momx, 'momx')
particles.populate(momy, 'momy')
particles.populate(momz, 'momz')

# Save snapshot
save(particles, initial_conditions)
