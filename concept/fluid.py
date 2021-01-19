# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2021 Jeppe Mosgaard Dakin.
#
# CO𝘕CEPT is free software: You can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CO𝘕CEPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CO𝘕CEPT. If not, see https://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from integration import cosmic_time, scale_factor')
cimport('from mesh import diff_domaingrid')

# Function pointer types used in this module
pxd("""
ctypedef double (*func_flux_limiter)(double)
""")



# The Kurganov-Tadmor method
@cython.pheader(
    # Arguments
    component='Component',
    ᔑdt=dict,
    a='double',
    rk_order='int',
    rk_step='int',
    # Locals
    Jᵐ='double[:, :, ::1]',
    Jᵐ_interface='double[::1]',
    Jᵐ_ptr='double*',
    Jᵐˣ='double[:, :, ::1]',
    Jᵐˣ_ptr='double*',
    Jₙ='double[:, :, ::1]',
    Jₙ_interface='double[::1]',
    Jₙ_ptr='double*',
    Jₙˣ='double[:, :, ::1]',
    Jₙˣ_ptr='double*',
    a_passed='double',
    f_interface='double[::1]',
    fluidscalar='FluidScalar',
    flux='double',
    flux_limiter_name=str,
    i='Py_ssize_t',
    index_c='Py_ssize_t',
    index_m='Py_ssize_t',
    index_m2='Py_ssize_t',
    index_p='Py_ssize_t',
    indices_local_start='Py_ssize_t[::1]',
    indices_local_end='Py_ssize_t[::1]',
    j='Py_ssize_t',
    k='Py_ssize_t',
    lr='int',
    m='int',
    n='int',
    shape=tuple,
    soundspeed='double',
    t_flux='double',
    use_ς='bint',
    use_𝒫='bint',
    v_interface='double[::1]',
    view=str,
    viewˣ=str,
    w='double',
    w_eff='double',
    Δ='double',
    Δx='double',
    ϕ=func_flux_limiter,
    ςᵐₙ='double[:, :, ::1]',
    ςᵐₙ_interface='double[::1]',
    ςᵐₙ_ptr='double*',
    ϱ='double[:, :, ::1]',
    ϱ_interface='double[::1]',
    ϱ_ptr='double*',
    ϱˣ='double[:, :, ::1]',
    ϱˣ_ptr='double*',
    𝒫='double[:, :, ::1]',
    𝒫_interface='double[::1]',
    𝒫_ptr='double*',
)
def kurganov_tadmor(component, ᔑdt, a=-1, rk_order=2, rk_step=0):
    """This function will update the non-linear fluid variables of the
    given component by advecting the flux terms in the fluid equations.
    Source terms (e.g. gravity) will not be applied.

    The rk_order determines the order of the Runge-Kutta time
    integration (can be 1 or 2), while rk_step specifies the current
    Runge kutta step (should not be explicitly passed). This function is
    written with rk_order == 2 in mind, and so although Euler
    integration (rk_order == 1) is possible, it is not implemented
    efficiently and so should only be used for testing.
    For rk_order == 2, the strategy is as follows. In the first step,
    the updates to the unstarred grids are computed and added to the
    starred grids, which have been pre-nullified. The unstarred grid
    values are then added to the starred grids. We could also just copy
    unstarred grid values to the starred grids instead of the
    nullification, but building up a grid of updates leads to smaller
    numerical errors. In the second step, the updates to the unstarred
    grids from the starred grids are computed and added to the unstarred
    grids, completing the time step.

    It is assumed that all fluid variable grids have correctly
    populated ghost points. Also, it is assumed that the unstarred 𝒫
    grid contains correctly realised 𝒫 data, regardless of whether or
    not 𝒫 is an actual non-linear variable (that is, whether
    or not the approximation P=wρ is enabled does not matter).

    When estimating the largest local comoving propagation speed,
    we use the global sound speed c*sqrt(w), and so
      v = abs(ẋ) + c*sqrt(w)/a, ẋ = dx/dt = u/a,
      u = a**(-4)*J/(ρ + c⁻²P)
        = a**(3*w_eff - 1)*J/(ϱ + c⁻²𝒫)
    ⇒ v = abs(a**(3*w_eff - 2)*J/(ϱ + c⁻²𝒫)) + c*sqrt(w)/a.
    """
    # There is nothing to be done by this function
    # if no J variable exist.
    if (    component.boltzmann_order == -1
        or (component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate')
    ):
        return
    if rk_order not in (1, 2):
        abort('Only Runge-Kutta orders 1 and 2 are implemented for the Kurganov-Tadmor method')
    a_passed = a
    if a == -1:
        a = universals.a
    # The scale factor passed as a matches the time of the kick
    # operation (source terms), but is half a time step ahead of the
    # drift operation (flux terms, i.e. the job of this function).
    # For Euler integration (rk_order == 0), this is the most fair value
    # of a, in the sense that it is right between the time at which the
    # fluid variables are defined now (according to flux terms), and the
    # time at which they will be defined at the end of this function.
    # For second-order Runge-Kutta integration (rk_order == 1) though,
    # the time step is split up into two half steps. The passed value
    # of a corresponds to the time right between these two half steps.
    # As we want a to be in the middle of each of the half time steps
    # themselves, we have to correct for this.
    if enable_Hubble and rk_order == 2:
        # The current cosmic time with respect to the flux terms is
        # t_flux = universals.t - 0.5*ᔑdt['1']. Right before a dump time
        # however, the time span of the two half time steps making up
        # the full time step are not equal. A more general formula
        # capturing this fact is t_flux = cosmic_time(a) - 0.5*ᔑdt['1'],
        # where we rely on the passed a to correspond to the time of the
        # source terms, which is half a step into the future compared to
        # the flux terms.
        t_flux = cosmic_time(a) - 0.5*ᔑdt['1']
        if rk_step == 0:
            a = scale_factor(t_flux + 0.25*ᔑdt['1'])
        elif rk_step == 1:
            a = scale_factor(t_flux + 0.75*ᔑdt['1'])
    # If this is the first (0) Runge-Kutta step,
    # the starred grids should be nullified.
    if rk_step == 0:
        component.nullify_nonlinear_fluid_gridsˣ()
    # Attribute names of the data in FluidScalars.
    # In the second (1) Runge-Kutta step, the roles of the
    # starred and the unstarred grids should be swapped.
    view  = 'grid_mv'
    viewˣ = 'gridˣ_mv'
    if rk_step == 1:
        view, viewˣ = viewˣ, view
    # Comoving grid spacing
    Δx = boxsize/component.gridsize
    # Parameters specific to the passed component
    w     = component.w    (a=a)
    w_eff = component.w_eff(a=a)
    shape = component.shape
    # The global comoving sound speed. Unless J is non-linear,
    # the Euler equation is not going to be solved, and so no pressure
    # gradient is ever applied, meaning that the sound speed
    # should be 0.
    if component.boltzmann_order > 0:
        soundspeed = light_speed*sqrt(w)/a
    else:
        soundspeed = 0
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding ghost points.
    indices_local_start = asarray([nghosts]*3, dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(shape      , dtype=C2np['Py_ssize_t']) - nghosts
    # Get the flux limiter function for this component
    flux_limiter_name = is_selected(
        component,
        fluid_options['kurganovtadmor']['flux_limiter_select'],
    )
    if flux_limiter_name == 'minmod':
        ϕ = flux_limiter_minmod
    elif flux_limiter_name in {'monotonizedcentral', 'mc'}:
        ϕ = flux_limiter_monotonized_central
    elif flux_limiter_name == 'ospre':
        ϕ = flux_limiter_ospre
    elif flux_limiter_name == 'superbee':
        ϕ = flux_limiter_superbee
    elif flux_limiter_name == 'sweby':
        ϕ = flux_limiter_sweby
    elif flux_limiter_name == 'umist':
        ϕ = flux_limiter_umist
    elif flux_limiter_name == 'vanalbada':
        ϕ = flux_limiter_vanalbada
    elif flux_limiter_name in {'vanleer', 'muscl', 'harmonic'}:
        ϕ = flux_limiter_vanleer
    elif flux_limiter_name == 'koren':
        ϕ = flux_limiter_koren
    else:
        abort(f'Flux limiter "{flux_limiter_name}" not implemented')
    #############################################
    # The flux term of the continuity equation, #
    # ∂ₜϱ = -ᔑa**(3*w_eff - 2)dt ∂ₘJᵐ           #
    #############################################
    masterprint('Computing energy fluxes in the continuity equation ...')
    # Extract density ϱ grids and pointers
    fluidscalar = component.ϱ
    ϱ  = getattr(fluidscalar, view )
    ϱˣ = getattr(fluidscalar, viewˣ)
    ϱ_ptr  = cython.address(ϱ [:, :, :])
    ϱˣ_ptr = cython.address(ϱˣ[:, :, :])
    # Extract pressure 𝒫 grid and pointer, realising it if 𝒫 is linear
    component.realize_if_linear(2, specific_multi_index='trace', a=a, use_gridˣ=𝔹['ˣ' in view])
    𝒫 = getattr(component.𝒫, view)
    𝒫_ptr = cython.address(𝒫[:, :, :])
    # Allocate needed interface arrays
    ϱ_interface  = empty(2, dtype=C2np['double'])
    𝒫_interface  = empty(2, dtype=C2np['double'])
    Jᵐ_interface = empty(2, dtype=C2np['double'])
    v_interface  = empty(2, dtype=C2np['double'])
    f_interface  = empty(2, dtype=C2np['double'])
    # Loop over the elements of J, realising them if J is linear
    for m in range(3):
        component.realize_if_linear(1, specific_multi_index=m, a=a, use_gridˣ=𝔹['ˣ' in view])
        Jᵐ = getattr(component.J[m], view)
        Jᵐ_ptr  = cython.address(Jᵐ[:, :, :])
        # Triple loop over local interfaces [i-½, j, k] for m = 0
        for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0] + 1]):
            for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1] + 1]):
                for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2] + 1]):
                    # Pointer indices into the 3D memory views,
                    # pointing to cell values at
                    # [i-2, j, k] through [i+1, j, k] for m = 0.
                    index_m2 = (ℤ[ℤ[(i - ℤ[2*(m == 0)]) *ℤ[shape[1]]]
                                  + (j - ℤ[2*(m == 1)])]*ℤ[shape[2]]
                                  + (k - ℤ[2*(m == 2)]))
                    index_m  = (ℤ[ℤ[(i - ℤ[1*(m == 0)]) *ℤ[shape[1]]]
                                  + (j - ℤ[1*(m == 1)])]*ℤ[shape[2]]
                                  + (k - ℤ[1*(m == 2)]))
                    index_c  = (ℤ[ℤ[ i                  *ℤ[shape[1]]]
                                   + j                 ]*ℤ[shape[2]]
                                   + k                 )
                    index_p  = (ℤ[ℤ[(i + ℤ[1*(m == 0)]) *ℤ[shape[1]]]
                                  + (j + ℤ[1*(m == 1)])]*ℤ[shape[2]]
                                  + (k + ℤ[1*(m == 2)]))
                    # The left and right interface value of fluid
                    # quantities at interface [i-½, j, k] for m = 0.
                    at_interface(index_m2, index_m, index_c, index_p, ϱ_ptr , ϱ_interface , ϕ)
                    at_interface(index_m2, index_m, index_c, index_p, 𝒫_ptr , 𝒫_interface , ϕ)
                    at_interface(index_m2, index_m, index_c, index_p, Jᵐ_ptr, Jᵐ_interface, ϕ)
                    # The left and right interface value of the
                    # absolute value of the propagation speed
                    # at interface [i-½, j, k] for m = 0.
                    for lr in range(2):
                        v_interface[lr] = (
                            abs(ℝ[a**(3*w_eff - 2)]*Jᵐ_interface[lr]
                                /(ϱ_interface[lr] + ℝ[light_speed**(-2)]*𝒫_interface[lr])
                            ) + soundspeed
                        )
                    # The left and right flux of ϱ through
                    # interface [i-½, j, k] for m = 0, due to the term
                    # -ᔑa**(3*w_eff - 2)∂ₘJᵐ.
                    for lr in range(2):
                        f_interface[lr] = (
                            ℝ[ᔑdt['a**(3*w_eff-2)', component.name]/ᔑdt['1']]
                            *Jᵐ_interface[lr]
                        )
                    # The final, numerical flux of ϱ through
                    # interface [i-½, j, k] for m = 0.
                    flux = kurganov_tadmor_flux(ϱ_interface, f_interface, v_interface)
                    # Update ϱ[i - 1, j, k] and ϱ[i, j, k] due to the
                    # flux through  interface [i-½, j, k] for m = 0.
                    Δ = flux*ℝ[(1 + rk_step)/rk_order*ᔑdt['1']/Δx]
                    ϱˣ_ptr[index_m] -= Δ
                    ϱˣ_ptr[index_c] += Δ
    masterprint('done')
    # Stop here if ϱ is the last non-linear fluid variable
    if component.boltzmann_order < 1:
        finalize_rk_step(component, ᔑdt, a_passed, rk_order, rk_step)
        return
    #####################################################
    # The flux terms of the Euler equation,             #
    # ∂ₜJᵐ = - ᔑa**( 3*w_eff - 2)dt ∂ⁿ(JᵐJₙ/(ϱ + c⁻²𝒫)) #
    #        - ᔑa**(-3*w_eff    )dt ∂ᵐ𝒫                 #
    #        - ᔑa**(-3*w_eff    )dt ∂ⁿςᵐₙ               #
    #####################################################
    # Allocate needed interface arrays
    Jₙ_interface = empty(2, dtype=C2np['double'])
    use_𝒫 = (not (component.w_type == 'constant' and component.w_constant == 0))
    use_ς = (
            component.boltzmann_order > 1
        or (component.boltzmann_order == 1 and component.boltzmann_closure == 'class')
    )
    if use_ς:
        ςᵐₙ_interface = empty(2, dtype=C2np['double'])
    if use_𝒫 and use_ς:
        masterprint('Computing momentum, pressure and shear fluxes in the Euler equation ...')
    elif use_𝒫:
        masterprint('Computing momentum and pressure fluxes in the Euler equation ...')
    elif use_ς:
        masterprint('Computing momentum and shear fluxes in the Euler equation ...')
    else:
        masterprint('Computing momentum fluxes in the Euler equation ...')
    # Loop over the elements of J
    for m in range(3):
        fluidscalar = component.J[m]
        Jᵐ  = getattr(fluidscalar, view )
        Jᵐˣ = getattr(fluidscalar, viewˣ)
        Jᵐ_ptr  = cython.address(Jᵐ [:, :, :])
        Jᵐˣ_ptr = cython.address(Jᵐˣ[:, :, :])
        # Loop over the elements of J
        for n in range(3):
            fluidscalar = component.J[n]
            Jₙ  = getattr(fluidscalar, view )
            Jₙˣ = getattr(fluidscalar, viewˣ)
            Jₙ_ptr  = cython.address(Jₙ [:, :, :])
            Jₙˣ_ptr = cython.address(Jₙˣ[:, :, :])
            with unswitch(2):
                if use_ς:
                    if m <= n:
                        # Realising element of ς if ς is linear
                        component.realize_if_linear(
                            2, specific_multi_index=(m, n), a=a, use_gridˣ=𝔹['ˣ' in view],
                        )
                        ςᵐₙ = getattr(component.ς[m, n], view)
                        ςᵐₙ_ptr = cython.address(ςᵐₙ[:, :, :])
            # Triple loop over local interfaces [i-½, j, k] for n = 0
            for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0] + 1]):
                for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1] + 1]):
                    for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2] + 1]):
                        # Pointer indices into the 3D memory views,
                        # pointing to cell values at
                        # [i-2, j, k] through [i+1, j, k] for n = 0.
                        index_m2 = (ℤ[ℤ[(i - ℤ[2*(n == 0)]) *ℤ[shape[1]]]
                                      + (j - ℤ[2*(n == 1)])]*ℤ[shape[2]]
                                      + (k - ℤ[2*(n == 2)]))
                        index_m  = (ℤ[ℤ[(i - ℤ[1*(n == 0)]) *ℤ[shape[1]]]
                                      + (j - ℤ[1*(n == 1)])]*ℤ[shape[2]]
                                      + (k - ℤ[1*(n == 2)]))
                        index_c  = (ℤ[ℤ[ i                  *ℤ[shape[1]]]
                                       + j                 ]*ℤ[shape[2]]
                                       + k                 )
                        index_p  = (ℤ[ℤ[(i + ℤ[1*(n == 0)]) *ℤ[shape[1]]]
                                      + (j + ℤ[1*(n == 1)])]*ℤ[shape[2]]
                                      + (k + ℤ[1*(n == 2)]))
                        # The left and right interface value of fluid
                        # quantities at interface [i-½, j, k] for n = 0.
                        at_interface(index_m2, index_m, index_c, index_p,
                            ϱ_ptr, ϱ_interface, ϕ)
                        at_interface(index_m2, index_m, index_c, index_p,
                            𝒫_ptr, 𝒫_interface, ϕ)
                        at_interface(index_m2, index_m, index_c, index_p,
                            Jᵐ_ptr, Jᵐ_interface, ϕ)
                        at_interface(index_m2, index_m, index_c, index_p,
                            Jₙ_ptr, Jₙ_interface, ϕ)
                        with unswitch(5):
                            if use_ς:
                                at_interface(index_m2, index_m, index_c, index_p,
                                    ςᵐₙ_ptr, ςᵐₙ_interface, ϕ)
                        # The left and right interface value of the
                        # absolute value of the propagation speed
                        # at interface [i-½, j, k] for n = 0.
                        for lr in range(2):
                            v_interface[lr] = (
                                abs(ℝ[a**(3*w_eff - 2)]*Jₙ_interface[lr]
                                    /(ϱ_interface[lr] + ℝ[light_speed**(-2)]*𝒫_interface[lr])
                                ) + soundspeed
                            )
                        # The left and right flux of Jᵐ through
                        # interface [i-½, j, k] for n = 0,
                        # due to the term
                        # -ᔑa**(3*w_eff - 2)∂ⁿ(JᵐJₙ/(ϱ + c⁻²𝒫)).
                        for lr in range(2):
                            f_interface[lr] = (
                                ℝ[ᔑdt['a**(3*w_eff-2)', component.name]/ᔑdt['1']]
                                *(Jᵐ_interface[lr]*Jₙ_interface[lr]
                                /(ϱ_interface[lr] + ℝ[light_speed**(-2)]*𝒫_interface[lr]))
                            )
                        # The left and right flux of Jᵐ through
                        # interface [i-½, j, k] for n = 0,
                        # due to the term
                        # -ᔑa**(-3*w_eff)∂ᵐ𝒫.
                        with unswitch(3):
                            if use_𝒫 and m == n:
                                for lr in range(2):
                                    f_interface[lr] += (
                                        ℝ[ᔑdt['a**(-3*w_eff)', component.name]/ᔑdt['1']]
                                        *𝒫_interface[lr]
                                    )
                        # The final, numerical flux of Jᵐ through
                        # interface [i-½, j, k] for n = 0,
                        # not including the ςᵐₙ term.
                        flux = kurganov_tadmor_flux(Jᵐ_interface, f_interface, v_interface)
                        # Update Jᵐ[i - 1, j, k] and Jᵐ[i, j, k]
                        # due to the flux through
                        # interface [i-½, j, k] for m = 0.
                        Δ = flux*ℝ[(1 + rk_step)/rk_order*ᔑdt['1']/Δx]
                        Jᵐˣ_ptr[index_m] -= Δ
                        Jᵐˣ_ptr[index_c] += Δ
                        # The flux of Jᵐ and Jₙ through
                        # interface [i-½, j, k] for n = 0,
                        # due to the term
                        # -ᔑa**(-3*w_eff)∂ⁿςᵐₙ.
                        # This is handled separately because we want a
                        # single realisation of ςᵐₙ to be used to update
                        # both Jᵐ and Jₙ. Here we make use of
                        # the symmetry ςᵐₙ = ςⁿₘ.
                        with unswitch(5):
                            if use_ς:
                                with unswitch(3):
                                    if m <= n:
                                        flux = 0
                                        for lr in range(2):
                                            flux += ςᵐₙ_interface[lr]
                                        Δ = flux*ℝ[
                                            0.5*(1 + rk_step)/rk_order
                                            *ᔑdt['a**(-3*w_eff)', component.name]/Δx
                                        ]
                                        # Update Jᵐ[i - 1, j, k] and
                                        # Jᵐ[i, j, k] due to the ςᵐₙ
                                        # flux through interface
                                        # [i-½, j, k] for m = 0.
                                        Jᵐˣ_ptr[index_m] -= Δ
                                        Jᵐˣ_ptr[index_c] += Δ
                                with unswitch(3):
                                    if m < n:
                                        # Update Jₙ[i - 1, j, k] and
                                        # Jₙ[i, j, k] due to the ςᵐₙ
                                        # flux through interface
                                        # [i-½, j, k] for m = 0.
                                        Jₙˣ_ptr[index_m] -= Δ
                                        Jₙˣ_ptr[index_c] += Δ
    masterprint('done')
    # Stop here if J is the last non-linear fluid variable
    if component.boltzmann_order < 2:
        finalize_rk_step(component, ᔑdt, a_passed, rk_order, rk_step)
        return
    # No further non-linear fluid equations implemented. Stop here.
    finalize_rk_step(component, ᔑdt, a_passed, rk_order, rk_step)

# Helper function to the kurganov_tadmor function,
# which given cell-centred values compute the 4 interface values.
@cython.header(
    # Arguments
    index_m2='Py_ssize_t',
    index_m='Py_ssize_t',
    index_c='Py_ssize_t',
    index_p='Py_ssize_t',
    center_ptr='double*',
    interface='double[::1]',
    ϕ=func_flux_limiter,
    # Locals
    center_m2='double',
    center_m='double',
    center_c='double',
    center_p='double',
    r_c='double',
    r_m='double',
    returns='void',
)
def at_interface(index_m2, index_m, index_c, index_p, center_ptr, interface, ϕ):
    """The 4 index variables are pointer indices into center_ptr,
    which point to cell-centred values of some quantity, in the cells
    {i-2, i-1, i, i+1}. The interface array should be of size 2
    and will be used to store the interface values. These interface
    values will be the left and right values at the interface i-½.
    The slope limiter function to use is given by ϕ.
    """
    # Lookup centred values
    center_m2 = center_ptr[index_m2]
    center_m  = center_ptr[index_m ]
    center_c  = center_ptr[index_c ]
    center_p  = center_ptr[index_p ]
    # The three curvature (slope ratio) values
    r_m = slope_ratio(  center_m - center_m2 , ℝ[center_c  - center_m])
    r_c = slope_ratio(ℝ[center_c - center_m ], ℝ[center_p  - center_c])
    # Fill the interface array with the 4 interface values
    interface[0] = center_m +   0.5*ϕ(r_m) *ℝ[center_c - center_m]  # Interface i-½, left
    interface[1] = center_c - ℝ[0.5*ϕ(r_c)]*ℝ[center_p - center_c]  # Interface i-½, right

# Helper function to the at_interface function, which compute the ratio
# of the given numerator and denominator in a numerically stable way.
@cython.header(
    # Arguments
    numerator='double',
    denominator='double',
    # Locals
    returns='double',
)
def slope_ratio(numerator, denominator):
    if abs(numerator) < slope_ratio_ϵ:
        return 0
    if abs(denominator) < slope_ratio_ϵ:
        if numerator > slope_ratio_ϵ:
            return ℝ[1/slope_ratio_ϵ]
        if numerator < ℝ[-slope_ratio_ϵ]:
            return ℝ[-1/slope_ratio_ϵ]
    return numerator/denominator
cython.declare(slope_ratio_ϵ='double')
slope_ratio_ϵ = 1e+2*machine_ϵ

# Helper function to the kurganov_tadmor function,
# which given the 4 interface values of a quantity, its fluxes and
# its speeds compute the total flux through the cell.
@cython.header(
    # Arguments
    q_interface='double[::1]',
    f_interface='double[::1]',
    v_interface='double[::1]',
    # Locals
    flux='double',
    v='double',
    returns='double',
)
def kurganov_tadmor_flux(q_interface, f_interface, v_interface):
    """All arguments should be arrays of size 2 with left and right
    values (in that order) at the i-½ interface. Here, q refers to a
    variable with flux f, while v is the absolute value
    of the propagation speed.
    """
    # Determine the maximum speed at interface i-½
    v = pairmax(v_interface[0], v_interface[1])
    # Compute the numerical flux through interface i-½
    flux = 0.5*((f_interface[1] + f_interface[0])
           - v* (q_interface[1] - q_interface[0])
    )
    # Return the total flux through the cell
    return flux

# Function which should be called at the end of each Runge-Kutta step
@cython.header(
    # Arguments
    component='Component',
    ᔑdt=dict,
    a='double',
    rk_order='int',
    rk_step='int',
)
def finalize_rk_step(component, ᔑdt, a, rk_order, rk_step):
    if rk_step == 0:
        # After the first (0) Runge-Kutta step, the starred non-linear
        # grids contain the updates to the unstarred non-linear grids.
        # Communicate these updates to the ghost points and then add to
        # them the original values.
        component.communicate_nonlinear_fluid_gridsˣ('=')
        component.copy_nonlinear_fluid_grids_to_gridsˣ('+=')
        if rk_order == 1:
            # If doing Euler integration, the time integration is
            # now finished, but the updated fluid grids are the
            # starred ones. Copy these into the unstarred grids.
            component.copy_nonlinear_fluid_gridsˣ_to_grids()
        elif rk_order == 2:
            # Now take the second Runge-Kutta step
            kurganov_tadmor(component, ᔑdt, a, rk_order, rk_step=1)
    elif rk_step == 1:
        # After the second (1) Runge-Kutta step,
        # the unstarred non-linear grids have been updated.
        # Communicate these updates to the ghost points.
        # Note that rk_order == 2 is required to reach this line.
        component.communicate_nonlinear_fluid_grids('=')

# The minmod flux limiter function
@cython.header(r='double', returns='double')
def flux_limiter_minmod(r):
    if r > 1:
        return 1
    if r > 0:
        return r
    return 0
# The monotonized central (aka MC) flux limiter
@cython.header(r='double', returns='double')
def flux_limiter_monotonized_central(r):
    if r > 3:
        return 2
    if r > 1./3.:
        return 0.5*(r + 1)
    if r > 0:
        return 2*r
    return 0
# The ospre flux limiter function
@cython.header(r='double', r2='double', returns='double')
def flux_limiter_ospre(r):
    if r > 0:
        r2 = r**2
        return 1.5*ℝ[r2 + r]/(ℝ[r2 + r] + 1)
    return 0
# The superbee flux limiter function
@cython.header(r='double', returns='double')
def flux_limiter_superbee(r):
    if r > 2:
        return 2
    if r > 1:
        return r
    if r > 0.5:
        return 1
    if r > 0:
        return 2*r
    return 0
# The Sweby flux limiter function
@cython.header(r='double', returns='double')
def flux_limiter_sweby(r):
    if r > β_sweby:
        return β_sweby
    if r > 1:
        return r
    if r > ℝ[1/β_sweby]:
        return 1
    if r > 0:
        return β_sweby*r
    return 0
cython.declare(β_sweby='double')
β_sweby = 1.5
# The UMIST flux limiter function
@cython.header(r='double', β='double', returns='double')
def flux_limiter_umist(r):
    if r > 5:
        return 2
    if r > 1:
        return 0.75 + 0.25*r
    if r > 0.2:
        return 0.25 + 0.75*r
    if r > 0:
        return 2*r
    return 0
# The van Albada flux limiter function
@cython.header(r='double', r2='double', returns='double')
def flux_limiter_vanalbada(r):
    if r > 0:
        r2 = r**2
        return (r2 + r)/(r2 + 1)
    return 0
# The van Leer (aka MUSCL, harmonic) flux limiter function
@cython.header(r='double', returns='double')
def flux_limiter_vanleer(r):
    if r > 0:
        return 2*r/(1 + r)
    return 0
# The Koren flux limiter function
@cython.header(r='double', returns='double')
def flux_limiter_koren(r):
    if r > 2.5:
        return 2
    if r > 0.25:
        return 1./3.*(1 + 2*r)
    if r > 0:
        return 2*r
    return 0

# Function which evolve the fluid variables of a component
# due to internal source terms. This function should be used
# together with the kurganov_tadmor function.
@cython.header(# Arguments
               component='Component',
               ᔑdt=dict,
               a='double',
               # Locals
               i='Py_ssize_t',
               w='double',
               ϱ_ptr='double*',
               𝒫_ptr='double*',
               )
def kurganov_tadmor_internal_sources(component, ᔑdt, a=-1):
    """By "internal sources" is meant source terms which do not arise
    due to interactions, such as the Hubble term in the continuity
    equation for P ≠ wρ.
    The kurganov_tadmor function takes care of all flux terms,
    leaving only the Hubble source term in the continuity equation
    to be applied by this function.
    """
    if a == -1:
        a = universals.a
    # Update ϱ due to its internal source term
    # in the continuity equation,
    # ∂ₜϱ = 3ᔑ(ȧ/a)dt (wϱ - c⁻²𝒫).
    if component.boltzmann_order > -1 and not component.approximations['P=wρ'] and enable_Hubble:
        masterprint('Computing the Hubble term in the continuity equation ...')
        w = component.w(a=a)
        ϱ_ptr = component.ϱ.grid
        𝒫_ptr = component.𝒫.grid
        for i in range(component.size):
            ϱ_ptr[i] += ℝ[3*ᔑdt['ȧ/a']]*(w*ϱ_ptr[i] - ℝ[light_speed**(-2)]*𝒫_ptr[i])
        masterprint('done')

# The MacCormack method
@cython.header(
    # Arguments
    component='Component',
    ᔑdt=dict,
    a_next='double',
    # Locals
    attempt='int',
    i='Py_ssize_t',
    max_vacuum_corrections=list,
    mc_step='int',
    steps='Py_ssize_t[::1]',
)
def maccormack(component, ᔑdt, a_next=-1):
    # There is nothing to be done by this function
    # if no J variable exist.
    if (   component.boltzmann_order == -1
        or component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate'
    ):
        return
    # Maximum allowed number of attempts to correct for
    # negative densities, for the first and second MacCormack step.
    max_vacuum_corrections = is_selected(
        component,
        fluid_options['maccormack']['max_vacuum_corrections_select'],
    )
    for mc_step, max_vacuum_corrections_step in enumerate(max_vacuum_corrections):
        if max_vacuum_corrections_step == 'gridsize':
            max_vacuum_corrections[mc_step] = component.gridsize
    # Step/flux directions for the first MacCormack step
    steps = next(maccormack_steps)
    # The two MacCormack steps
    for mc_step in range(2):
        # Evolve the fluid variables. If this leads to negative
        # densities, attempts are made to correct this.
        for attempt in range(max_vacuum_corrections[mc_step]):
           # Evolve fluid variables. In the first MacCormack step,
           # the variables are re-evolved at each attempt. In the
           # second MacCormack step, the variables should only be
           # evolved once (vacuum correction may still take place
           # multiple times).
           if attempt == 0 or mc_step == 0:
               # Compute starred variables from unstarred variables
               # (first MacCormack step) or vice versa
               # (second MacCormack step).
               maccormack_step(component, ᔑdt, steps, mc_step, a_next)
           # Do vacuum corrections if toggled for this species.
           # If not, check but do not correct for vacuum.
           if 𝔹[is_selected(component, fluid_options['maccormack']['vacuum_corrections_select'])]:
               # Nullify the Δ buffers, so that they are ready to
               # be used by the following vacuum correction sweep.
               component.nullify_Δ()
               # Check and correct for density values heading
               # dangerously fast towards negative values. If every
               # density value is OK, accept this attempt at a
               # MacCormack step as is.
               if not correct_vacuum(component, mc_step):
                   break
           else:
               check_vacuum(component, mc_step)
               break
        else:
            # None of the attempted MacCormack steps were accepted.
            # If this is the second MacCormack step, this means that
            # we have been unable to correct for negative densities.
            if mc_step == 1:
                abort(
                    f'Giving up after {max_vacuum_corrections[mc_step]} failed attempts '
                    f'to remove negative densities in {component.name}'
                )
        # Reverse step direction for the second MacCormack step
        for i in range(3):
           steps[i] *= -1
    # The two MacCormack steps leave all values of all fluid variables
    # with double their actual values. All grid values thus need
    # to be halved. Note that no further communication is needed as we
    # also halve the ghost points.
    component.scale_nonlinear_fluid_grids(0.5)
    # Nullify the starred grid buffers and the Δ buffers,
    # leaving these with no leftover junk.
    component.nullify_nonlinear_fluid_gridsˣ()
    component.nullify_Δ()

# Infinite generator cycling through the 8 triples of
# step/flux directions, used in the maccormack function.
def generate_maccormack_steps():
    steps = []
    for sign in (+1, -1):
        steps.append(sign*asarray((+1, +1, +1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((-1, +1, -1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((-1, -1, +1), dtype=C2np['Py_ssize_t']))
        steps.append(sign*asarray((+1, -1, -1), dtype=C2np['Py_ssize_t']))
    yield from itertools.cycle(steps)
maccormack_steps = generate_maccormack_steps()

# Function which evolve the fluid variables of a component,
# disregarding all source terms.
@cython.header(# Arguments
               component='Component',
               ᔑdt=dict,
               steps='Py_ssize_t[::1]',
               mc_step='int',
               a_next='double',
               # Locals
               J_div='double[:, :, ::1]',
               J_el='double[:, :, ::1]',
               Jˣ_el='double[:, :, ::1]',
               dim_div='int',
               dim_el='int',
               fluidscalar='FluidScalar',
               grid='double*',
               gridˣ='double*',
               indices_local_end='Py_ssize_t[::1]',
               indices_local_start='Py_ssize_t[::1]',
               view=str,
               viewˣ=str,
               Δ='double',
               i='Py_ssize_t',
               j='Py_ssize_t',
               k='Py_ssize_t',
               step_i='Py_ssize_t',
               step_j='Py_ssize_t',
               step_k='Py_ssize_t',
               Δx='double',
               ϱ='double[:, :, ::1]',
               ϱˣ='double[:, :, ::1]',
               𝒫='double[:, :, ::1]',
               )
def maccormack_step(component, ᔑdt, steps, mc_step, a_next=-1):
    """It is assumed that the unstarred and starred grids have
    correctly populated ghost points.
    """
    # There is nothing to be done by this function
    # if no J variable exist.
    if (   component.boltzmann_order == -1
        or component.boltzmann_order == 0 and component.boltzmann_closure == 'truncate'
    ):
        return
    # Comoving grid spacing
    Δx = boxsize/component.gridsize
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding ghost points.
    indices_local_start = asarray([nghosts]*3    , dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(component.shape, dtype=C2np['Py_ssize_t']) - nghosts
    # At the beginning of the first MacCormack step, the starred buffers
    # should contain a copy of the actual (unstarred) data.
    # At the beginning of the second MacCormack step, the unstarred
    # variables should be updated by adding to them the values in the
    # starred buffers.
    for fluidscalar in component.iterate_nonlinear_fluidscalars():
        grid  = fluidscalar.grid
        gridˣ = fluidscalar.gridˣ
        for i in range(component.size):
            with unswitch:
                if mc_step == 0:
                    gridˣ[i] = grid[i]
                else:  # mc_step == 1
                    grid[i] += gridˣ[i]
    # Attribute names of the data in fluidscalars.
    # In the second MacCormack step, the roles of the
    # starred and the unstarred grids should be swapped.
    view  = 'grid_mv'
    viewˣ = 'gridˣ_mv'
    if mc_step == 1:
        view, viewˣ = viewˣ, view
    # The continuity equation (flux terms only).
    # Δϱ = - ᔑa**(3*w_eff - 2)dt ∂ᵢJⁱ  (energy flux).
    masterprint('Computing energy fluxes in the continuity equation ...')
    ϱ  = getattr(component.ϱ, view )
    ϱˣ = getattr(component.ϱ, viewˣ)
    for (dim_div, ), J_div in component.J.iterate(view, multi_indices=True, a_next=a_next):
        step_i = steps[dim_div] if dim_div == 0 else 0
        step_j = steps[dim_div] if dim_div == 1 else 0
        step_k = steps[dim_div] if dim_div == 2 else 0
        for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
            for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                    Δ = ℤ[steps[dim_div]]*(
                          J_div[i + step_i, j + step_j, k + step_k]
                        - J_div[i         , j         , k         ]
                    )
                    ϱˣ[i, j, k] += Δ*ℝ[-ᔑdt['a**(3*w_eff-2)', component.name]/Δx]
    masterprint('done')
    # Stop here if ϱ is the last non-linear fluid variable
    if component.boltzmann_order < 1:
        finalize_maccormack_step(component, mc_step)
        return
    # The Euler equation (flux terms only).
    # ΔJⁱ = -ᔑa**(3*w_eff - 2)dt ∂ʲ(JⁱJⱼ/(ϱ + c⁻²𝒫))  (momentum flux).
    # As the pressure is not evolved by the MacCormack method,
    # we use the unstarred grid in both MacCormack steps. We could of
    # course re-realise the pressure from ϱˣ after the first MacCormack
    # step. It was found that this does not impact the final result,
    # and so is off below.
    masterprint('Computing momentum fluxes in the Euler equation ...')
    if False:
        component.realize_if_linear(2, specific_multi_index='trace', a_next=a_next, use_gridˣ=(mc_step == 1))
        𝒫 = getattr(component.𝒫, view)
    else:
        if mc_step == 0:
            component.realize_if_linear(2, specific_multi_index='trace', a_next=a_next)
        𝒫  = component.𝒫.grid_mv
    for dim_el in range(3):  # Loop over elements of J
        J_el  = getattr(component.J[dim_el], view )
        Jˣ_el = getattr(component.J[dim_el], viewˣ)
        # The momentum flux
        for dim_div in range(3):  # Loop over dimensions in divergence
            J_div = getattr(component.J[dim_div], view)
            step_i = steps[dim_div] if dim_div == 0 else 0
            step_j = steps[dim_div] if dim_div == 1 else 0
            step_k = steps[dim_div] if dim_div == 2 else 0
            for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
                for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
                    for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                        Δ = ℤ[steps[dim_div]]*(
                              J_el [i + step_i, j + step_j, k + step_k]
                             *J_div[i + step_i, j + step_j, k + step_k]
                             /(                       ϱ[i + step_i, j + step_j, k + step_k]
                               + ℝ[light_speed**(-2)]*𝒫[i + step_i, j + step_j, k + step_k]
                               )
                            - J_el [i, j, k]
                             *J_div[i, j, k]
                             /(                       ϱ[i, j, k]
                               + ℝ[light_speed**(-2)]*𝒫[i, j, k]
                               )
                        )
                        Jˣ_el[i, j, k] += Δ*ℝ[-ᔑdt['a**(3*w_eff-2)', component.name]/Δx]
    masterprint('done')
    # Stop here if J is the last non-linear fluid variable
    if component.boltzmann_order < 2:
        finalize_maccormack_step(component, mc_step)
        return
    # No further non-linear fluid equations implemented. Stop here.
    finalize_maccormack_step(component, mc_step)

# Function for doing communication of ghost points of fluid grids
# after each MacCormack step.
@cython.header(component='Component', mc_step='int')
def finalize_maccormack_step(component, mc_step):
    # Populate the ghost points of all fluid variable grids with the
    # updated values. Dependent on whether we are in the end of the
    # first or the second MacCormack step (mc_step), the updated grids
    # are really the starred grids (first MacCormack step) or the
    # unstarred grids (second MacCormack step).
    if mc_step == 0:
        component.communicate_nonlinear_fluid_gridsˣ('=')
    else:  # mc_step == 1
        component.communicate_nonlinear_fluid_grids ('=')

# Function which evolve the fluid variables of a component
# due to internal source terms. This function should be used together
# with the maccormack function.
@cython.header(
    # Arguments
    component='Component',
    ᔑdt=dict,
    a_next='double',
    # Locals
    Jᵢ='FluidScalar',
    Jᵢ_ptr='double*',
    i='Py_ssize_t',
    j='Py_ssize_t',
    multi_index=tuple,
    multi_index_list=list,
    potential='double[:, :, ::1]',
    potential_ptr='double*',
    n='Py_ssize_t',
    source='double[:, :, ::1]',
    source_ptr='double*',
    w='double',
    Δx='double',
    ςᵢⱼ='FluidScalar',
    ςᵢⱼ_ptr='double*',
    ϱ_ptr='double*',
    𝒫='double[:, :, ::1]',
    𝒫_ptr='double*',
)
def maccormack_internal_sources(component, ᔑdt, a_next=-1):
    """By "internal sources" is meant source terms which do not arise
    due to interactions, such as the Hubble term in the continuity
    equation for P ≠ wρ.
    A special kind of such internal source arise when
    component.boltzmann_closure == 'class', in which case one additional
    fluid variable should be realised using CLASS, and then affect its
    lower fluid variable (which will then be the highest dynamic fluid
    variable) through the dynamical fluid equations. The coupling
    between two such fluid variables takes the form of a flux,
    but since one of the variables is not dynamic, here it act just like
    a source term, and should hence be treated as such.
    Because lower fluid variables appear in the source terms of higher
    fluid variables, we need to update the higher fluid variables first.
    """
    # Extract scalar variable fluid grids
    ϱ_ptr = component.ϱ.grid
    𝒫_ptr = component.𝒫.grid
    𝒫     = component.𝒫.grid_mv
    # Physical grid spacing
    Δx = boxsize/component.gridsize
    # If closure of the Boltzmann hierarchy is achieved by continuously
    # realising ς, do this realisation now and update J accordingly.
    # This source term looks like
    # ΔJᵢ = -ᔑa**(-3*w_eff)dt ∂ʲςⁱⱼ.
    if (    component.boltzmann_order > 1
        or (component.boltzmann_order == 1 and component.boltzmann_closure == 'class')):
        masterprint('Computing the shear term in the Euler equation ...')
        # Loop over all distinct ςᵢⱼ and realise them as we go
        for multi_index, ςᵢⱼ in component.ς.iterate(multi_indices=True, a_next=a_next):
            # The potential of the source is
            # -ᔑa**(-3*w_eff)dt ςⁱⱼ.
            # Construct this potential, using the starred ϱ grid
            # as the buffer.
            potential     = component.ϱ.gridˣ_mv
            potential_ptr = component.ϱ.gridˣ
            ςᵢⱼ_ptr = ςᵢⱼ.grid
            for n in range(component.size):
                potential_ptr[n] = ℝ[-ᔑdt['a**(-3*w_eff)', component.name]]*ςᵢⱼ_ptr[n]
            # Loop over elements of J affected by ςᵢⱼ
            for i in set(multi_index):
                Jᵢ = component.J[i]
                Jᵢ_ptr = Jᵢ.grid
                # The index in multi_index other than the chosen i is
                # the dimension of differentiation by the divergence, j.
                multi_index_list = list(multi_index)
                multi_index_list.remove(i)
                j = multi_index_list[0]
                # Differentiate the potential and apply the source term
                source = diff_domaingrid(potential, j, 2, Δx)
                source_ptr = cython.address(source[:, :, :])
                for n in range(component.size):
                    Jᵢ_ptr[n] += source_ptr[n]
        masterprint('done')
    # The pressure term in the Euler equation
    # ΔJⁱ = -ᔑa**(-3*w_eff)dt ∂ⁱ𝒫.
    if (
            component.boltzmann_order > 0
        and not (component.w_type == 'constant' and component.w_constant == 0)
    ):
        masterprint('Computing the pressure term in the Euler equation ...')
        for i in range(3):
            Jᵢ = component.J[i]
            Jᵢ_ptr = Jᵢ.grid
            source = diff_domaingrid(𝒫, i, 2, Δx)
            source_ptr = cython.address(source[:, :, :])
            for n in range(component.size):
                Jᵢ_ptr[n] += ℝ[-ᔑdt['a**(-3*w_eff)', component.name]]*source_ptr[n]
        masterprint('done')
    # Update ϱ due to its internal source term
    # in the continuity equation
    # Δϱ = 3ᔑ(ȧ/a)dt (wϱ - c⁻²𝒫).
    if component.boltzmann_order > -1 and not component.approximations['P=wρ'] and enable_Hubble:
        masterprint('Computing the Hubble term in the continuity equation ...')
        w = component.w()
        for n in range(component.size):
            ϱ_ptr[n] += ℝ[3*ᔑdt['ȧ/a']]*(w*ϱ_ptr[n] - ℝ[light_speed**(-2)]*𝒫_ptr[n])
        masterprint('done')

# Function which checks and warn about vacuum in a fluid component
@cython.header(# Arguments
               component='Component',
               mc_step='int',
               # Locals
               any_vacuum='bint',
               i='Py_ssize_t',
               ϱ='double*',
               )
def check_vacuum(component, mc_step):
    # Grab pointer to the density. After the first MacCormack step,
    # the starred buffers have been updated from the non-starred
    # buffers, and so it is the starred buffers that should be checked
    # for vacuum values. After the second MacCormack step, the unstarred
    # buffers have been updated from the starred buffers, and so we
    # should check the unstarred buffers.
    if component.is_linear(0):
        return
    if mc_step == 0:
        ϱ = component.ϱ.grid
    else:  # mc_step == 1
        ϱ = component.ϱ.gridˣ
    # Check for vacuum
    any_vacuum = False
    for i in range(component.size):
        if ϱ[i] < ρ_vacuum:
            any_vacuum = True
            break
    # Show a warning if any vacuum elements were found
    any_vacuum = reduce(any_vacuum, op=MPI.LOR)
    if any_vacuum:
        masterwarn(f'Vacuum detected in {component.name}')

# Function which checks for imminent vacuum in a fluid component
# and does one sweep of vacuum corrections.
@cython.header(
    # Arguments
    component='Component',
    mc_step='int',
    # Locals
    Jx='double[:, :, ::1]',
    Jx_correction='double',
    Jx_ptr='double*',
    Jxˣ='double[:, :, ::1]',
    Jy='double[:, :, ::1]',
    Jy_correction='double',
    Jy_ptr='double*',
    Jyˣ='double[:, :, ::1]',
    Jz='double[:, :, ::1]',
    Jz_correction='double',
    Jz_ptr='double*',
    Jzˣ='double[:, :, ::1]',
    dist2='Py_ssize_t',
    fac_smoothing='double',
    fac_time='double',
    foresight='double',
    i='Py_ssize_t',
    indices_local_start='Py_ssize_t[::1]',
    indices_local_end='Py_ssize_t[::1]',
    j='Py_ssize_t',
    k='Py_ssize_t',
    m='Py_ssize_t',
    mi='Py_ssize_t',
    mj='Py_ssize_t',
    mk='Py_ssize_t',
    n='Py_ssize_t',
    ni='Py_ssize_t',
    nj='Py_ssize_t',
    nk='Py_ssize_t',
    vacuum_imminent='bint',
    ΔJx='double[:, :, ::1]',
    ΔJx_ptr='double*',
    ΔJy='double[:, :, ::1]',
    ΔJy_ptr='double*',
    ΔJz='double[:, :, ::1]',
    ΔJz_ptr='double*',
    Δϱ='double[:, :, ::1]',
    Δϱ_ptr='double*',
    ϱ='double[:, :, ::1]',
    ϱ_correction='double',
    ϱ_ijk='double',
    ϱ_ptr='double*',
    ϱˣ='double[:, :, ::1]',
    ϱˣ_ijk='double',
    returns='bint',
)
def correct_vacuum(component, mc_step):
    """This function will detect and correct for imminent vacuum in a
    fluid component. If vacuum is found to be imminent, a value of True
    will be returned, otherwise False. The vacuum detection is done
    differently depending on the MacCormack step (the passed mc_step).
    For the first MacCormack step, vacuum is considered imminent if a
    density below the vacuum density, ρ_vacuum, will be reached within
    'foresight' similar time steps. For the second MacCormack step,
    vacuum is considered imminent if the density is below the
    vacuum density. The vacuum correction is done by smoothing all fluid
    variables in the 3x3x3 neighbouring cells surrounding the
    vacuum cell. The smoothing between each pair of cells,
    call them (i, j), is given by
    ϱi += fac_smoothing*(ϱj - ϱi)/r²,
    ϱj += fac_smoothing*(ϱi - ϱj)/r²,
    (similar for other fluid variables)
    where r is the distance between the cells in grid units.
    Whether or not any vacuum corrections were made is returned
    as the return value.
    Experimentally, it has been found that when
    max_vacuum_corrections[0] == 1,
    the following values give good, stable results:
    foresight = 30
    fac_smoothing = 1/(6/1 + 12/2 + 8/3)
    """
    if component.is_linear(0):
        return False
    # In the case of the first MacCormack step, consider vacuum to be
    # imminent if a cell will reach the vacuum density after this many
    # similar time steps. Should be at least 1.
    foresight = is_selected(component, fluid_options['maccormack']['foresight_select'])
    # Amount of smoothing to apply when vacuum is detected.
    # A numerator of 0 implies no smoothing.
    # A numerator of 1 implies that in the most extreme case,
    # a vacuum cell will be replaced with a weighted average of its
    # 26 neighbour cells (all of the original cell will be distributed
    # among these neighbours).
    fac_smoothing = 1./(6 + 12./2. + 8./3.)*is_selected(
        component,
        fluid_options['maccormack']['smoothing_select'],
    )
    # Arrays of start and end indices for the local part of the
    # fluid grids, meaning disregarding ghost points.
    indices_local_start = asarray([nghosts]*3    , dtype=C2np['Py_ssize_t'])
    indices_local_end   = asarray(component.shape, dtype=C2np['Py_ssize_t']) - nghosts
    # Extract memory views and pointers to the fluid variables
    ϱ       = component.ϱ .grid_mv
    ϱ_ptr   = component.ϱ .grid
    ϱˣ      = component.ϱ .gridˣ_mv
    ϱˣ_ptr  = component.ϱ .gridˣ
    Δϱ      = component.ϱ .Δ_mv
    Δϱ_ptr  = component.ϱ .Δ
    Jx      = component.Jx.grid_mv
    Jx_ptr  = component.Jx.grid
    Jxˣ     = component.Jx.gridˣ_mv
    Jxˣ_ptr = component.Jx.gridˣ
    ΔJx     = component.Jx.Δ_mv
    ΔJx_ptr = component.Jx.Δ
    Jy      = component.Jy.grid_mv
    Jy_ptr  = component.Jy.grid
    Jyˣ     = component.Jy.gridˣ_mv
    Jyˣ_ptr = component.Jy.gridˣ
    ΔJy     = component.Jy.Δ_mv
    ΔJy_ptr = component.Jy.Δ
    Jz      = component.Jz.grid_mv
    Jz_ptr  = component.Jz.grid
    Jzˣ     = component.Jz.gridˣ_mv
    Jzˣ_ptr = component.Jz.gridˣ
    ΔJz     = component.Jz.Δ_mv
    ΔJz_ptr = component.Jz.Δ
    # In the case of the second MacCormack step, the role of the
    # starred and the unstarred variables should be swapped.
    if mc_step == 1:
        ϱ     , ϱˣ      = ϱˣ     , ϱ
        ϱ_ptr , ϱˣ_ptr  = ϱˣ_ptr , ϱ_ptr
        Jx    , Jxˣ     = Jxˣ    , Jx
        Jx_ptr, Jxˣ_ptr = Jxˣ_ptr, Jx_ptr
        Jy    , Jyˣ     = Jyˣ    , Jy
        Jy_ptr, Jyˣ_ptr = Jyˣ_ptr, Jy_ptr
        Jz    , Jzˣ     = Jzˣ    , Jz
        Jz_ptr, Jzˣ_ptr = Jzˣ_ptr, Jz_ptr
    # Loop over the local domain and check and compute
    # corrections for imminent vacuum.
    vacuum_imminent = False
    for         i in range(ℤ[indices_local_start[0]], ℤ[indices_local_end[0]]):
        for     j in range(ℤ[indices_local_start[1]], ℤ[indices_local_end[1]]):
            for k in range(ℤ[indices_local_start[2]], ℤ[indices_local_end[2]]):
                # Unstarred and starred density at this point
                ϱ_ijk  = ϱ [i, j, k]
                ϱˣ_ijk = ϱˣ[i, j, k]
                # Check for imminent vacuum.
                # After the first MacCormack step, vacuum is considered
                # to be imminent if a density below the vacuum density,
                # ρ_vacuum, will be reached within foresight similar
                # time steps. That is, vacuum is imminent if
                # ϱ + foresight*dϱ < ρ_vacuum,
                # where dϱ is the change in ϱ from the first MacCormack
                # step, given by dϱ = ½(ϱˣ - ϱ), where the factor ½ is
                # due to ϱˣ really holding double the change,
                # ϱˣ = ϱ + 2*dϱ. Put together, this means that vacuum
                # is imminent if
                # ϱˣ + ϱ*(2/foresight - 1) < 2/foresight*ρ_vacuum.
                # After the second MacCormack step, vacuum is considered
                # to be imminent only if the density is lower than the
                # vacuum density, ρ_vacuum. Because the starred
                # variables hold double their actual values,
                # this corresponds to
                # ϱˣ_ijk < 2*ρ_vacuum.
                if (   (    mc_step == 0
                        and ϱ_ijk*ℝ[2/foresight - 1] + ϱˣ_ijk < ℝ[2/foresight*ρ_vacuum]
                        )
                    or (    mc_step == 1
                        and ϱˣ_ijk < ℝ[2*ρ_vacuum]
                        )
                    ):
                    vacuum_imminent = True
                    # The amount of smoothing to apply depends upon
                    # how far into the future densities below the vacuum
                    # density will be reached.
                    if mc_step == 0:
                        # The number of time steps before densities
                        # become lower than the vacuum density is given
                        # by
                        # ϱ + timesteps*dϱ == ρ_vacuum, dϱ = ½(ϱˣ - ϱ).
                        # --> timesteps = 2*(ϱ - ρ_vacuum)/(ϱ - ϱˣ).
                        fac_time = 0.5*(ϱ_ijk - ϱˣ_ijk)/(ϱ_ijk - ρ_vacuum)
                    else:  # mc_step == 1
                        # The density is already lower
                        # than the vacuum density.
                        fac_time = 1
                    # Loop over all cell pairs (m, n) in the 3x3x3 block
                    # surrounding the vacuum cell and apply smoothing.
                    for m in range(27):
                        # 3D indices of m'th cell
                        mi = i + relative_neighbour_indices_i[m]
                        mj = j + relative_neighbour_indices_j[m]
                        mk = k + relative_neighbour_indices_k[m]
                        for n in range(m + 1, 27):
                            # 3D indices of n'th cell
                            ni = i + relative_neighbour_indices_i[n]
                            nj = j + relative_neighbour_indices_j[n]
                            nk = k + relative_neighbour_indices_k[n]
                            # Distance squared between the two cells,
                            # in grid units (1 ≤ dist2 ≤ 12).
                            dist2 = (ni - mi)**2 + (nj - mj)**2 + (nk - mk)**2
                            # Compute vacuum corrections
                            ϱ_correction = (ϱ[ni, nj, nk] - ℝ[ϱ[mi, mj, mk]])*ℝ[
                                fac_smoothing*fac_time]*ℝ[1/dist2]
                            Jx_correction = (Jx[ni, nj, nk] - ℝ[Jx[mi, mj, mk]])*ℝ[
                                fac_smoothing*fac_time]*ℝ[1/dist2]
                            Jy_correction = (Jy[ni, nj, nk] - ℝ[Jy[mi, mj, mk]])*ℝ[
                                fac_smoothing*fac_time]*ℝ[1/dist2]
                            Jz_correction = (Jz[ni, nj, nk] - ℝ[Jz[mi, mj, mk]])*ℝ[
                                fac_smoothing*fac_time]*ℝ[1/dist2]
                            # Store vacuum corrections
                            Δϱ [mi, mj, mk] += ϱ_correction
                            ΔJx[mi, mj, mk] += Jx_correction
                            ΔJy[mi, mj, mk] += Jy_correction
                            ΔJz[mi, mj, mk] += Jz_correction
                            Δϱ [ni, nj, nk] -= ϱ_correction
                            ΔJx[ni, nj, nk] -= Jx_correction
                            ΔJy[ni, nj, nk] -= Jy_correction
                            ΔJz[ni, nj, nk] -= Jz_correction
    # If vacuum is imminent on any process, consider it to be
    # imminent on every process.
    vacuum_imminent = allreduce(vacuum_imminent, op=MPI.LOR)
    if vacuum_imminent:
        # Communicate contributions to local vacuum corrections
        # residing on other processes.
        component.communicate_fluid_Δ('+=')
        # Local Δ buffers now store final values.
        # Populate ghost points of Δ buffers.
        component.communicate_fluid_Δ('=')
        # Apply vacuum corrections.
        # Note that no further communication is needed as we also apply
        # vacuum corrections to the ghost points.
        for i in range(component.size):
            ϱ_ptr [i] += Δϱ_ptr [i]
        for i in range(component.size):
            Jx_ptr[i] += ΔJx_ptr[i]
        for i in range(component.size):
            Jy_ptr[i] += ΔJy_ptr[i]
        for i in range(component.size):
            Jz_ptr[i] += ΔJz_ptr[i]
    # The return value should indicate whether or not
    # vacuum corrections have been carried out.
    return vacuum_imminent
# Relative 1D indices to the 27 neighbours of a cell (itself included).
# These are thus effectively mappings from 1D indices to 3D indices.
cython.declare(
    relative_neighbour_indices_i_mv='Py_ssize_t[::1]',
    relative_neighbour_indices_j_mv='Py_ssize_t[::1]',
    relative_neighbour_indices_k_mv='Py_ssize_t[::1]',
    relative_neighbour_indices_i='Py_ssize_t*',
    relative_neighbour_indices_j='Py_ssize_t*',
    relative_neighbour_indices_k='Py_ssize_t*',
)
relative_neighbour_indices = asarray(
    [(i, j, k) for i in range(-1, 2) for j in range(-1, 2) for k in range(-1, 2)],
    dtype=C2np['Py_ssize_t'],
)
relative_neighbour_indices_i_mv = asarray(relative_neighbour_indices[:, 0]).copy()
relative_neighbour_indices_j_mv = asarray(relative_neighbour_indices[:, 1]).copy()
relative_neighbour_indices_k_mv = asarray(relative_neighbour_indices[:, 2]).copy()
relative_neighbour_indices_i = cython.address(relative_neighbour_indices_i_mv[:])
relative_neighbour_indices_j = cython.address(relative_neighbour_indices_j_mv[:])
relative_neighbour_indices_k = cython.address(relative_neighbour_indices_k_mv[:])
