# This file is part of CO𝘕CEPT, the cosmological 𝘕-body code in Python.
# Copyright © 2015–2020 Jeppe Mosgaard Dakin.
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
# along with CO𝘕CEPT. If not, see http://www.gnu.org/licenses/
#
# The author of CO𝘕CEPT can be contacted at dakin(at)phys.au.dk
# The latest version of CO𝘕CEPT is available at
# https://github.com/jmd-dk/concept/



# Import everything from the commons module.
# In the .pyx file, Cython declared variables will also get cimported.
from commons import *

# Cython imports
cimport('from communication import get_buffer, smart_mpi')

# Function pointer types used in this module
pxd('ctypedef double (*func_d_dd) (double, double)')



# Class used for 1D, cubic interpolation.
# GSL is used when in compiled mode, while SciPy is used in
# pure Python mode. The behaviour is identical.
@cython.cclass
class Spline:
    # Initialization method
    @cython.header(
        # Arguments
        x='double[::1]',
        y='double[::1]',
        name=str,
        logx='bint',
        logy='bint',
        # Locals
        i='Py_ssize_t',
    )
    def __init__(self, x, y, name='', *, logx=False, logy=False):
        # Here x and y = y(x) are the tabulated data.
        # The values in x must be in increasing order.
        # If logx (logy) is True, the log will be taken of the x (y)
        # data before it is splined.
        #
        # The triple quoted string below serves as the type declaration
        # for the data attributes of the Spline type.
        # It will get picked up by the pyxpp script
        # and indluded in the .pxd file.
        """
        double[::1] x
        double[::1] y
        str name
        bint logx
        bint logy
        double xmin
        double xmax
        double abs_tol_min
        double abs_tol_max
        gsl_interp_accel* acc
        gsl_spline* spline
        """
        # The name is only for clearer error messages
        self.name = name
        # The size of the data
        if x.shape[0] != y.shape[0]:
            abort(
                f'Spline "{self.name}": '
                f'A Spline object cannot be tabulated using arrays of different lengths. '
                f'Arrays of lengths {x.shape[0]} and {y.shape[0]} were passed.'
            )
        if x.shape[0] < 3:
            abort(
                f'Spline "{self.name}": '
                f'Too few tabulated values ({x.shape[0]}) were given for '
                f'cubic spline interpolation. At least 3 is needed.'
            )
        # Take the log, if requested.
        # Note that a copy of the input data is used,
        # we do not mutate the input data.
        self.logx, self.logy = logx, logy
        if self.logx:
            for i in range(x.shape[0]):
                if x[i] <= 0:
                    self.logx = False
                    warn(
                        f'Spline "{self.name}": '
                        f'Could not take log of spline x data as it contains non-positive values'
                    )
                    break
        if self.logx:
            x = np.log(x)
        if self.logy:
            for i in range(y.shape[0]):
                if y[i] <= 0:
                    self.logy = False
                    warn(
                        f'Spline "{self.name}": '
                        'Could not take log of spline y data as it contains non-positive values'
                    )
                    break
        if self.logy:
            y = np.log(y)
        # Check that the passed x values are strictly increasing and
        # remove doppelgänger points. Note that this does not mutate the
        # passed x and y arrays, and so the new x and y refer to
        # different underlying data.
        x, y = remove_doppelgängers(x, y)
        # Store a copy of the non-logged, doppelgänger free data
        self.x = np.exp(x) if self.logx else asarray(x).copy()
        self.y = np.exp(y) if self.logy else asarray(y).copy()
        # Store meta data
        self.xmin = x[0]
        self.xmax = x[x.shape[0] - 1]
        abs_tol = 1e-9*(self.xmax - self.xmin) + machine_ϵ
        self.abs_tol_min = abs_tol + 0.5*(x[1] - self.xmin)
        self.abs_tol_max = abs_tol + 0.5*(self.xmax - x[x.shape[0] - 2])
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            # Initialize the spline.
            # Here we simply overwrite the spline attribute.
            # The boundary condition type is set to 'natural'
            # as to match the default of GSL.
            self.spline = scipy.interpolate.CubicSpline(
                asarray(x).copy(), asarray(y).copy(), bc_type='natural')
        else:
            # Allocate an interpolation accelerator
            # and a (cubic) spline object.
            self.acc = gsl_interp_accel_alloc()
            self.spline = gsl_spline_alloc(gsl_interp_cspline, x.shape[0])
            # Initialize the spline
            gsl_spline_init(self.spline, cython.address(x[:]), cython.address(y[:]), x.shape[0])

    # Method for doing spline evaluation
    @cython.pheader(
        # Arguments
        x_in='double',
        # Locals
        x='double',
        y='double',
        returns='double',
    )
    def eval(self, x_in):
        x = log(x_in) if self.logx else x_in
        # Check that x is within the interpolation interval
        x = self.in_interval(x, 'interpolate to')
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            y = float(self.spline(x))
        else:
            y = gsl_spline_eval(self.spline, x, self.acc)
        # Undo the log
        if self.logy:
            y = exp(y)
        return y

    # Method for doing spline derivative evaluation
    @cython.pheader(
        # Arguments
        x_in='double',
        # Locals
        x='double',
        ẏ='double',
        returns='double',
    )
    def eval_deriv(self, x_in):
        x = log(x_in) if self.logx else x_in
        # Check that x is within the interpolation interval
        x = self.in_interval(x, 'differentiate at')
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            ẏ = self.spline(x, 1)
        else:
            ẏ = gsl_spline_eval_deriv(self.spline, x, self.acc)
        # Undo the log
        if self.logx and self.logy:
            # ∂ₓy(x) = y(x)/x*∂ₗₙ₍ₓ₎ln(y(x))
            ẏ *= self.eval(x_in)/x_in
        elif self.logx:
            # ∂ₓy(x) = x⁻¹*∂ₗₙ₍ₓ₎y(x)
            ẏ /= x_in
        elif self.logy:
            # ∂ₓy(x) = y(x)*∂ₓln(y(x))
            ẏ *= self.eval(x_in)
        return ẏ

    # Method for computing the definite integral over some
    # interval [a, b] of the splined function.
    @cython.pheader(
        # Arguments
        a='double',
        b='double',
        # Locals
        sign_flip='bint',
        ᔑ='double',
        returns='double',
    )
    def integrate(self, a, b):
        if self.logx or self.logy:
            abort(f'Spline "{self.name}": Spline integration not possible for logged data')
        # Check that the supplied limits are
        # within the interpolation interval.
        a = self.in_interval(a, 'integrate from')
        b = self.in_interval(b, 'integrate to')
        # The function gsl_spline_eval_integ fails for a > b.
        # Take care of this manually by switching a and b and note
        # the sign change.
        sign_flip = False
        if a > b:
            sign_flip = True
            a, b = b, a
        # Use SciPy in pure Python and GSL when compiled
        if not cython.compiled:
            ᔑ = self.spline.integrate(a, b)
        else:
            ᔑ = gsl_spline_eval_integ(self.spline, a, b, self.acc)
        # Remember the sign change for a > b
        return -ᔑ if sign_flip else ᔑ

    # Method for checking whether a given number
    # is within the tabulated interval.
    @cython.header(# Arguments
                   x='double',
                   action=str,
                   # Locals
                   returns='double',
                   )
    def in_interval(self, x, action='interpolate to'):
        # Check that the supplied x is within the
        # interpolation interval. If it is just barely outside of it,
        # move it to the boundary.
        if x < self.xmin:
            if x > self.xmin - self.abs_tol_min:
                x = self.xmin
            else:
                abort(
                    f'Spline "{self.name}": '
                    f'Could not {action} {x} because it is outside the tabulated interval '
                    f'[{self.xmin}, {self.xmax}]'
                )
        elif x > self.xmax:
            if x < self.xmax + self.abs_tol_max:
                x = self.xmax
            else:
                abort(
                    f'Spline "{self.name}": '
                    f'Could not {action} {x} because it is outside the tabulated interval '
                    f'[{self.xmin}, {self.xmax}]'
                )
        return x

    # This method is automaticlly called when a Spline instance
    # is garbage collected. All manually allocated memory is freed.
    def __dealloc__(self):
        # Free the accelerator and the spline object
        gsl_spline_free(self.spline)
        gsl_interp_accel_free(self.acc)



# Function for cleaning up arrays of points possibly containing
# duplicate points.
@cython.pheader(
    # Arguments
    xs='object',  # double[::1] or tuple or list of double[::1]
    y='double[::1]',
    rel_tol='double',
    copy='bint',
    # Locals
    accepted_indices='Py_ssize_t[::1]',
    i='Py_ssize_t',
    index='Py_ssize_t',
    j='Py_ssize_t',
    multiple_x_passed='bint',
    size='Py_ssize_t',
    sizes=tuple,
    x='double[::1]',
    x_arrays=list,
    x_cleaned='double[::1]',
    x_cleaned_arrays=list,
    x_prev='double',
    xdiff='double',
    xdiff_prev='double',
    y_cleaned='double[::1]',
    returns=tuple,
)
def remove_doppelgängers(xs, y, rel_tol=1e-1, copy=False):
    """Given arrays of x and y values, this function checks for
    doppelgängers in the x values, meaning consecutive x values that are
    exactly or very nearly equal. New arrays with doppelgängers
    removed to single points will be returned.
    You may have as many x arrays as you like.
    The relative tolerance used in determining doppelgängers is given
    by rel_tol.
    For this function to work, the x values must be in increasing order.
    The passed x and y will not be modified. Note that this function
    reuses the same buffer for all returned arrays, meaning that if you
    call this function multiple times, the returned arrays will point to
    the same underlying data. If you want the returned arrays to be
    freshly allocated, use copy=True.
    """
    global accepted_indices_ptr, accepted_indices_size
    # Pack xs into a list of x arrays
    if isinstance(xs, (tuple, list)):
        x_arrays = list(xs)
        multiple_x_passed = True
    else:
        x_arrays = [xs]
        multiple_x_passed = False
    # All the x arrays should have the same size,
    # and the y array must be at least as large.
    sizes = tuple({x.shape[0] for x in x_arrays})
    if len(sizes) > 1:
        abort(
            f'The x arrays passed to remove_doppelgängers() do not all have '
            f'the same size {sizes}'
        )
    size = sizes[0]
    if y.shape[0] < size:
        abort(
            f'The y array passed to remove_doppelgängers() must have at least the same number '
            f'of elements as that of the passed x array(s) ({size}), '
            f'but it only has {y.shape[0]} elements'
        )
    y = y[:size]
    # Fetch buffers for storing the cleaned up versions of the arrays
    x_cleaned_arrays = [get_buffer(size, f'remove_doppelgängers (x_cleaned {i})')
        for i in range(len(x_arrays))
    ]
    y_cleaned = get_buffer(size, 'remove_doppelgängers (y_cleaned)')
    # No cleanup is required if a single
    # or no points exist in the passed data.
    if size < 2:
        for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
            x_cleaned[:] = x
        y_cleaned[:] = y
        if multiple_x_passed:
            if copy:
                return ([asarray(x_cleaned).copy() for x_cleaned in x_cleaned_arrays],
                    asarray(y_cleaned).copy())
            else:
                return [asarray(x_cleaned) for x_cleaned in x_cleaned_arrays], asarray(y_cleaned)
        else:
            if copy:
                return asarray(x_cleaned).copy(), asarray(y_cleaned).copy()
            else:
                return asarray(x_cleaned), asarray(y_cleaned)
    # Check that the x values are in increasing order
    for x in x_arrays:
        x_prev = x[0]
        for i in range(1, size):
            if ℝ[x[i]] < x_prev:
                abort(
                    'The values in (one of) the x array(s) passed to remove_doppelgängers() '
                    'are not in increasing order'
                )
            x_prev = ℝ[x[i]]
    # Loop from right to left, checking for doppelgängers
    # using the first passed x array.
    if accepted_indices_size < size:
        accepted_indices_ptr = realloc(accepted_indices_ptr, size*sizeof('Py_ssize_t'))
        accepted_indices_size = size
    accepted_indices = cast(accepted_indices_ptr, 'Py_ssize_t[:size]')
    for index in range(size - 1, size - 3, -1):
        accepted_indices[index] = index
    x = x_arrays[0]
    x_prev = x[index]
    xdiff_prev = x[size - 1] - x_prev
    index -= 1
    for i in range(size - 3, -1, -1):
        xdiff = x_prev - x[i]
        if xdiff > rel_tol*xdiff_prev:
            # Accept this point
            accepted_indices[index] = i
            index -= 1
            x_prev = x[i]
            xdiff_prev = xdiff
    # Copy accepted points to the cleaned arrays
    accepted_indices = accepted_indices[index+1 :]
    size = accepted_indices.shape[0]
    for j, (x, x_cleaned) in enumerate(zip(x_arrays, x_cleaned_arrays)):
        for i in range(size):
            x_cleaned[i] = x[accepted_indices[i]]
        x_cleaned_arrays[j] = x_cleaned[:size]
    for i in range(size):
        y_cleaned[i] = y[accepted_indices[i]]
    y_cleaned = y_cleaned[:size]
    # Always include the first point
    for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
        x_cleaned[0] = x[0]
    y_cleaned[0] = y[0]
    # Let x and y point to the cleaned up versions of these arrays
    for i, x_cleaned in enumerate(x_cleaned_arrays):
        x_arrays[i] = x_cleaned
    y = y_cleaned
    # Loop from left to right, checking for doppelgängers
    # using the first passed x array.
    for index in range(2):
        accepted_indices[index] = index
    x = x_arrays[0]
    x_prev = x[1]
    xdiff_prev = x_prev - x[0]
    index += 1
    for i in range(2, x.shape[0]):
        xdiff =  ℝ[x[i]] - x_prev
        if xdiff > rel_tol*xdiff_prev:
            # Accept this point
            accepted_indices[index] = i
            index += 1
            x_prev = ℝ[x[i]]
            xdiff_prev = xdiff
    # Copy accepted points to the cleaned arrays
    size = index
    accepted_indices = accepted_indices[:size]
    for j, x_cleaned in enumerate(x_cleaned_arrays):
        for i in range(size):
            x_cleaned[i] = x_cleaned[accepted_indices[i]]
        x_cleaned_arrays[j] = x_cleaned[:size]
    for i in range(size):
        y_cleaned[i] = y_cleaned[accepted_indices[i]]
    y_cleaned = y_cleaned[:size]
    # Always include the last point
    for x, x_cleaned in zip(x_arrays, x_cleaned_arrays):
        x_cleaned[ℤ[size - 1]] = x[ℤ[y.shape[0] - 1]]
    y_cleaned[ℤ[size - 1]] = y[ℤ[y.shape[0] - 1]]
    # Return the cleaned arrays
    if multiple_x_passed:
        if copy:
            return ([asarray(x_cleaned).copy() for x_cleaned in x_cleaned_arrays],
                asarray(y_cleaned).copy())
        else:
            return [asarray(x_cleaned) for x_cleaned in x_cleaned_arrays], asarray(y_cleaned)
    else:
        if copy:
            return asarray(x_cleaned_arrays[0]).copy(), asarray(y_cleaned).copy()
        else:
            return asarray(x_cleaned_arrays[0]), asarray(y_cleaned)
# Pointer used by the remove_doppelgängers function
cython.declare(accepted_indices_ptr='Py_ssize_t*', accepted_indices_size='Py_ssize_t')
accepted_indices_size = 3
accepted_indices_ptr = malloc(accepted_indices_size*sizeof('Py_ssize_t'))

# This function implements the Hubble parameter H(a)=ȧ/a,
# The Hubble parameter is only ever written here. Every time the Hubble
# parameter is used in the code, this function should be called.
@cython.header(# Arguments
               a='double',
               # Locals
               ΩΛ='double',
               returns='double',
               )
def hubble(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    # When using CLASS for the background computation, simply look up
    # H(a) in the tabulated data. The largest tabulated a is 1, but it
    # may happen that this function gets called with an a that is
    # sightly larger. In this case, compute H directly via the simple
    # Friedmann equation, just as when CLASS is not being used for the
    # background computation.
    if enable_class_background and a <= 1:
        if spline_a_H is None:
            abort('The function H(a) has not been tabulated. Have you called init_time?')
        return spline_a_H.eval(a)
    # CLASS not enabled. Assume that the universe is flat
    # and consists purely of matter and dark energy.
    ΩΛ = 1 - Ωm
    return H0*sqrt(Ωm/(a**3 + machine_ϵ) + ΩΛ)

# Function returning the proper time differentiated scale factor,
# given the value of the scale factor.
@cython.header(# Argumetns
               a='double',
               # Locals
               returns='double',
               )
def ȧ(a):
    return a*hubble(a)

# Function returning the proper time differentiated scale factor,
# given the value of the scale factor.
# This function is intended to be used by the rkf45 function, which is
# why it takes in both t and a as arguments, even though t is not used.
@cython.header(# Argumetns
               t='double',
               a='double',
               # Locals
               returns='double',
               )
def ȧ_rkf(t, a):
    return ȧ(a)

# Function returning the proper time differentiated Hubble parameter
@cython.header(# Arguments
               a='double',
               # Locals
               ΩΛ='double',
               returns='double',
               )
def Ḣ(a=-1):
    if not enable_Hubble:
        return 0
    if a == -1:
        a = universals.a
    # When using CLASS for the background computation, simply compute
    # Ḣ(a) from the tabulated data. The largest tabulated a is 1, but it
    # may happen that this function gets called with an a that is
    # sightly larger. In this case, compute Ḣ directly via the
    # derivative of the simple Friedmann equation, just as when CLASS is
    # not being used for the background computation.
    # We have Ḣ = dH/dt = ȧ*dH/da.
    if enable_class_background and a <= 1:
        if spline_a_H is None:
            abort('The function H(a) has not been tabulated. Have you called init_time?')
        return ȧ(a)*spline_a_H.eval_deriv(a)
    # CLASS not enabled. Assume that the universe is flat
    # and consists purely of matter and dark energy.
    ΩΛ = 1 - Ωm
    return ȧ(a)*(-1.5*H0*Ωm/(sqrt(a**5*(Ωm + a**3*ΩΛ)) + machine_ϵ))

# Function returning the second proper time derivative of the factor,
# given the value of the scale factor.
@cython.header(# Argumetns
               a='double',
               # Locals
               returns='double',
               )
def ä(a):
    return a*(hubble(a)**2 + Ḣ(a))

# Function for computing the scale factor a at a given cosmic time t
@cython.header(# Arguments
               t='double',
               # Locals
               returns='double',
               )
def scale_factor(t=-1):
    if not enable_Hubble:
        return 1
    if t == -1:
        t = universals.t
    # When using CLASS for the background computation,
    # simply look up a(t) in the tabulated data.
    if enable_class_background:
        if spline_t_a is None:
            abort('The function a(t) has not been tabulated. Have you called init_time?')
        return spline_t_a.eval(t)
    # Not using CLASS.
    # Integrate the Friedmann equation from the beginning of time
    # to the requested time.
    return rkf45(ȧ_rkf, machine_ϵ, machine_ϵ, t, abs_tol=1e-9, rel_tol=1e-9)

# Function for computing the cosmic time t at some given scale factor a
@cython.pheader(# Arguments
                a='double',
                a_lower='double',
                t_lower='double',
                t_upper='double',
                # Locals
                a_test='double',
                a_test_prev='double',
                abs_tol='double',
                rel_tol='double',
                t='double',
                t_max='double',
                t_min='double',
                returns='double',
                )
def cosmic_time(a=-1, a_lower=machine_ϵ, t_lower=machine_ϵ, t_upper=-1):
    """Given lower and upper bounds on the cosmic time, t_lower and
    t_upper, and the scale factor at time t_lower, a_lower,
    this function finds the future time at which the scale
    factor will have the value a.
    """
    global t_max_ever
    # This function only works when Hubble expansion is enabled
    if not enable_Hubble:
        abort('The cosmic_time function was called. '
              'A mapping from a to t is only meaningful when Hubble expansion is enabled.')
    if a == -1:
        a = universals.a
    # When using CLASS for the background computation,
    # simply look up t(a) in the tabulated data.
    if enable_class_background:
        if spline_a_t is None:
            abort('The function t(a) has not been tabulated. Have you called init_time?')
        return spline_a_t.eval(a)
    # Not using CLASS
    if t_upper == -1:
        # If no explicit t_upper is passed, use t_max_ever
        t_upper = t_max_ever
    elif t_upper > t_max_ever:
        # If passed t_upper exceeds t_max_ever,
        # set t_max_ever to this larger value.
        t_max_ever = t_upper
    # Tolerances
    abs_tol = 1e-9
    rel_tol = 1e-9
    # Saves copies of extreme t values
    t_min, t_max = t_lower, t_upper
    # Compute the cosmic time at which the scale factor had the value a,
    # using a binary search.
    a_test = a_test_prev = t = -1
    while (    not isclose(a_test,  a,       0, ℝ[2*machine_ϵ])
           and not isclose(t_upper, t_lower, 0, ℝ[2*machine_ϵ])):
        t = 0.5*(t_upper + t_lower)
        a_test = rkf45(ȧ_rkf, a_lower, t_min, t, abs_tol, rel_tol)
        if a_test == a_test_prev:
            if not isclose(a_test, a):
                if isclose(t, t_max):
                    # Integration stopped at t == t_max.
                    # Break out so that this function is called
                    # recursively, this time with a highter t_upper.
                    break
                else:
                    # Integration halted for whatever reason
                    abort('Integration of scale factor a(t) halted')
            break
        a_test_prev = a_test
        if a_test > a:
            t_upper = t
        else:
            t_lower = t
    # If the result is equal to t_max, it means that the t_upper
    # argument was too small! Call recursively with double t_upper.
    if isclose(t, t_max):
        return cosmic_time(a, a_test, t_max, 2*t_max)
    return t
# Initialize t_max_ever, a cosmic time later than what will
# ever be reached (if exceeded, it dynamically grows).
cython.declare(t_max_ever='double')
t_max_ever = 20*units.Gyr

# Function for updating the scale factor
@cython.header(# Arguments
               a='double',
               t='double',
               Δt='double',
               returns='double',
               )
def expand(a, t, Δt):
    """Integrates the Friedmann equation from t to t + Δt,
    where the scale factor at time t is given by a. Returns a(t + Δt).
    """
    return rkf45(ȧ_rkf, a, t, t + Δt, abs_tol=1e-9, rel_tol=1e-9, save_intermediate=True)

# Function for solving ODEs of the form ḟ(t, f)
@cython.header(# Arguments
               ḟ=func_d_dd,
               f_start='double',
               t_start='double',
               t_end='double',
               abs_tol='double',
               rel_tol='double',
               save_intermediate='bint',
               # Locals
               error='double',
               f='double',
               f4='double',
               f5='double',
               h='double',
               h_max='double',
               i='Py_ssize_t',
               k1='double',
               k2='double',
               k3='double',
               k4='double',
               k5='double',
               k6='double',
               tolerance='double',
               Δt='double',
               returns='double',
               )
def rkf45(ḟ, f_start, t_start, t_end, abs_tol, rel_tol, save_intermediate=False):
    """ḟ(t, f) is the derivative of f with respect to t. Initial values
    are given by f_start and t_start. ḟ will be integrated from t_start
    to t_end. That is, the returned value is f(t_end). The absolute and
    relative accuracies are given by abs_tol, rel_tol.
    If save_intermediate is True, intermediate values optained during
    the integration will be kept in the global variables t_tab, f_tab.
    """
    global alloc_tab, f_tab, f_tab_mv, integrand_tab, integrand_tab_mv
    global size_tab, t_tab, t_tab_mv
    # The maximum and minimum step size
    Δt = t_end - t_start
    h_min = ℝ[10*machine_ϵ]
    h_max = 0.01*Δt + h_min
    # Initial values
    h = h_max*rel_tol
    i = 0
    f = f_start
    t = t_start
    if Δt == 0:
        return f
    # Drive the method
    while t_end - t >  ℝ[1e+3*machine_ϵ]:
        # The embedded Runge-Kutta-Fehlberg 4(5) step
        k1 = h*ḟ(t             , f)
        k2 = h*ḟ(t + ℝ[1/4  ]*h, f + ℝ[1/4      ]*k1)
        k3 = h*ḟ(t + ℝ[3/8  ]*h, f + ℝ[3/32     ]*k1 + ℝ[9/32      ]*k2)
        k4 = h*ḟ(t + ℝ[12/13]*h, f + ℝ[1932/2197]*k1 + ℝ[-7200/2197]*k2 + ℝ[7296/2197 ]*k3)
        k5 = h*ḟ(t +          h, f + ℝ[439/216  ]*k1 + ℝ[-8        ]*k2 + ℝ[3680/513  ]*k3 + ℝ[-845/4104  ]*k4)
        k6 = h*ḟ(t + ℝ[1/2  ]*h, f + ℝ[-8/27    ]*k1 + ℝ[2         ]*k2 + ℝ[-3544/2565]*k3 + ℝ[1859/4104  ]*k4 + ℝ[-11/40]*k5)
        f5 =                     f + ℝ[16/135   ]*k1                    + ℝ[6656/12825]*k3 + ℝ[28561/56430]*k4 + ℝ[-9/50 ]*k5 + ℝ[2/55]*k6
        f4 =                     f + ℝ[25/216   ]*k1                    + ℝ[1408/2565 ]*k3 + ℝ[2197/4104  ]*k4 + ℝ[-1/5  ]*k5
        # The error estimate
        error = abs(f5 - f4) + machine_ϵ
        # The local tolerance
        tolerance = (rel_tol*abs(f5) + abs_tol)*sqrt(h/Δt) + ℝ[2*machine_ϵ]
        if error < tolerance:
            # Step accepted
            t += h
            f = f5
            # Save intermediate t and f values
            if save_intermediate:
                t_tab[i] = t
                f_tab[i] = f
                i += 1
                # If necessary, t_tab and f_tab get resized (doubled)
                if i == alloc_tab:
                    alloc_tab *= 2
                    t_tab = realloc(t_tab, alloc_tab*sizeof('double'))
                    f_tab = realloc(f_tab, alloc_tab*sizeof('double'))
                    integrand_tab = realloc(integrand_tab, alloc_tab*sizeof('double'))
                    t_tab_mv = cast(t_tab, 'double[:alloc_tab]')
                    f_tab_mv = cast(f_tab, 'double[:alloc_tab]')
                    integrand_tab_mv = cast(integrand_tab, 'double[:alloc_tab]')
        # Updating step size
        h *= 0.95*(tolerance/error)**0.25
        if h > h_max:
            h = h_max
        elif h < h_min:
            h = h_min
        if t + h > t_end:
            h = t_end - t
    if save_intermediate:
        size_tab = i
    return f
# Allocate t_tab, f_tab and integrand_tab at import time.
# t_tab and f_tab are used to store intermediate values of t and f
# in the Runge-Kutta-Fehlberg method. integrand_tab stores the
# associated values of the integrand in ᔑ_t^(t + Δt) integrand dt.
cython.declare(alloc_tab='Py_ssize_t',
               f_tab='double*',
               f_tab_mv='double[::1]',
               integrand_tab='double*',
               integrand_tab_mv='double[::1]',
               size_tab='Py_ssize_t',
               t_tab='double*',
               t_tab_mv='double[::1]'
               )
alloc_tab = 100
size_tab = 0
t_tab = malloc(alloc_tab*sizeof('double'))
f_tab = malloc(alloc_tab*sizeof('double'))
integrand_tab = malloc(alloc_tab*sizeof('double'))
t_tab_mv = cast(t_tab, 'double[:alloc_tab]')
f_tab_mv = cast(f_tab, 'double[:alloc_tab]')
integrand_tab_mv = cast(integrand_tab, 'double[:alloc_tab]')

# Function for calculating integrals of the sort
# ᔑ_t^(t + Δt) integrand(a) dt.
@cython.header(
    # Arguments
    key=object,  # str or tuple
    t_ini='double',
    Δt='double',
    all_components=list,
    # Locals
    a='double',
    a_tab_spline='double[::1]',
    component='Component',
    component_0='Component',
    component_1='Component',
    component_name=str,
    component_names=list,
    components=list,
    i='Py_ssize_t',
    integrand=str,
    integrand_tab_spline='double[::1]',
    size='Py_ssize_t',
    spline='Spline',
    t='double',
    t_tab_spline='double[::1]',
    w_eff='double',
    w_eff_0='double',
    w_eff_1='double',
    returns='double',
)
def scalefactor_integral(key, t_ini, Δt, all_components):
    """This function returns the integral
    ᔑ_t^(t + Δt) integrand(a) dt.
    The integrand is passed as the key argument, which may be a string
    (e.g. 'a**(-1)') or a tuple in the format (string, component.name),
    (string, component_0.name, component_1.name) etc., where again the
    first string is really the integrand. The tuple form is used when
    the integrand is component specific, e.g. 'a**(-3*w_eff)'.
    When the CLASS background is disabled it is important that the
    expand function expand(a, t, Δt) has been called prior to calling
    this function, as expand generates the values needed in
    the integration. You can call this function multiple times (and with
    different integrands) without calling expand in between.
    """
    # Extract the integrand from the passed key
    components = []
    if 𝔹[isinstance(key, str)]:
        # Global integrand
        integrand, component = key, None
    else:
        # Component integrand
        integrand, *component_names = key
        for component_name in component_names:
            for component in all_components:
                if component.name == component_name:
                    components.append(component)
                    break
    # When using the CLASS background, a(t) is already tabulated
    # throughout time. Here we simply construct Spline objects over the
    # given integrand and ask for the integral.
    if enable_class_background:
        spline = spline_t_integrands.get(key)
        if spline is not None:
            return spline.integrate(t_ini, t_ini + Δt)
    # At this point, either enable_class_background is False,
    # or this is the first time this function has been called with
    # the given key. In both cases, we now need to tabulate
    # the integrand.
    if enable_class_background:
        for component in components:
            if component is not None and component.w_eff_type != 'constant':
                a_tab_spline = component.w_eff_spline.x
                t_tab_spline = asarray([spline_a_t.eval(a) for a in a_tab_spline])
                break
        else:
            a_tab_spline = spline_a_t.x
            t_tab_spline = spline_a_t.y
        integrand_tab_spline = empty(t_tab_spline.shape[0], dtype=C2np['double'])
        size = t_tab_spline.shape[0]
    else:
        # We do not use the CLASS background. If expand() has been
        # called as it should, t_tab_mv stores the tabulated values of t
        # while f_tab_mv stores the tabulated values of a.
        a_tab_spline = f_tab_mv
        t_tab_spline = t_tab_mv
        integrand_tab_spline = integrand_tab_mv
        size = size_tab
    # Do the tabulation
    for i in range(size):
        a = a_tab_spline[i]
        t = t_tab_spline[i]
        with unswitch:
            if 𝔹[isinstance(key, str)]:
                # Global integrands
                with unswitch:
                    if integrand == '1' or integrand == '':
                        integrand_tab_spline[i] = 1
                    elif integrand == 'a**2':
                        integrand_tab_spline[i] = a**2
                    elif integrand == 'a**(-1)':
                        integrand_tab_spline[i] = 1/a
                    elif integrand == 'a**(-2)':
                        integrand_tab_spline[i] = 1/a**2
                    elif integrand == 'ȧ/a':
                        integrand_tab_spline[i] = hubble(a)
                    elif master:
                        abort(
                            f'The scalefactor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            elif ℤ[len(key)] == 2:
                # Single-component integrands
                with unswitch:
                    if integrand == 'a**(-3*w_eff)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff)
                    elif integrand == 'a**(-3*(1+w_eff))':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*(1 + w_eff))
                    elif integrand == 'a**(-3*w_eff-1)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff - 1)
                    elif integrand == 'a**(3*w_eff-2)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(3*w_eff - 2)
                    elif integrand == 'a**(2-3*w_eff)':
                        w_eff = component.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(2 - 3*w_eff)
                    elif integrand == 'a**(-3*w_eff)*Γ/H':
                        with unswitch:
                            if enable_Hubble:
                                w_eff = component.w_eff(t=t, a=a)
                                integrand_tab_spline[i] = a**(-3*w_eff)*component.Γ(a)/hubble(a)
                            else:
                                integrand_tab_spline[i] = component.Γ(a)
                    elif master:
                        abort(
                            f'The scalefactor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            elif ℤ[len(key)] == 3:
                # Two-component integrands
                component_0, component_1 = components
                with unswitch:
                    if integrand == 'a**(-3*w_eff₀-3*w_eff₁-1)':
                        w_eff_0 = component_0.w_eff(t=t, a=a)
                        w_eff_1 = component_1.w_eff(t=t, a=a)
                        integrand_tab_spline[i] = a**(-3*w_eff_0 - 3*w_eff_1 - 1)
                    elif master:
                        abort(
                            f'The scalefactor integral with "{integrand}" as the integrand '
                            f'is not implemented'
                        )
            else:
                abort(f'scalefactor_integral(): Invalid lenth ({len(key)}) of key {key}')
    # Do the integration
    spline = Spline(t_tab_spline[:size], integrand_tab_spline[:size], integrand)
    if enable_class_background:
        spline_t_integrands[key] = spline
    return spline.integrate(t_ini, t_ini + Δt)
# Global dict of Spline objects defined by scalefactor_integral
cython.declare(spline_t_integrands=dict)
spline_t_integrands = {}

# Function which sets the value of universals.a and universals.t
# based on the user parameters a_begin and t_begin together with the
# cosmology if enable_Hubble is True. The functions t(a), a(t) and H(a)
# will also be tabulated and stored in the module namespace in the
# form of spline_a_t, spline_t_a and spline_a_H.
@cython.pheader(
    # Arguments
    reinitialize='bint',
    # Locals
    H_values='double[::1]',
    a_begin_correct='double',
    a_values='double[::1]',
    background=dict,
    t_values='double[::1]',
    t_begin_correct='double',
)
def init_time(reinitialize=False):
    global time_initialized, spline_a_t, spline_t_a, spline_a_H
    if time_initialized and not reinitialize:
        return
    time_initialized = True
    if enable_Hubble:
        # Hubble expansion enabled.
        # If CLASS should be used to compute the evolution of
        # the background throughout time, run CLASS now.
        if enable_class_background:
            # Ideally we would call CLASS via compute_cosmo from the
            # linear module, as this would preserve all results for any
            # future needs. However, the linear module imports functions
            # from this module (integration), so importing from
            # linear would create a cyclic import. Instead we do the
            # call ourselves via the "raw" call_class function from the
            # commons module.
            # Note that only the master have access to the results
            # from the CLASS computation.
            cosmo = call_class(class_call_reason='in order to set the cosmic clock')
            background = cosmo.get_background()
            # What we need to store is the cosmic time t and the Hubble
            # parametert H, both as functions of the scale factor a.
            # Since the time stepping is done in t, we furthermore want
            # the scale factor a as a function of cosmic time t.
            # We do this by defining global Spline objects.
            a_values = smart_mpi(1/(background['z'] + 1), 0, mpifun='bcast')
            t_values = smart_mpi(background['proper time [Gyr]']*units.Gyr, 1, mpifun='bcast')
            H_values = smart_mpi(
                background['H [1/Mpc]']*(light_speed/units.Mpc), 2, mpifun='bcast')
            spline_a_t = Spline(a_values, t_values, 't(a)', logx=True, logy=True)
            spline_t_a = Spline(t_values, a_values, 'a(t)', logx=True, logy=True)
            spline_a_H = Spline(a_values, H_values, 'H(a)', logx=True, logy=True)
        # A specification of initial scale factor or
        # cosmic time is needed.
        if 'a_begin' in user_params_keys_raw:
            # a_begin specified
            if 't_begin' in user_params_keys_raw:
                # t_begin also specified
                masterwarn('Ignoring t_begin = {}*{} becuase enable_Hubble is True '
                           'and a_begin is specified'.format(t_begin, unit_time))
            a_begin_correct = a_begin
            t_begin_correct = cosmic_time(a_begin_correct)
        elif 't_begin' in user_params_keys_raw:
            # a_begin not specified, t_begin specified
            if t_begin == 0:
                abort(
                    'You have specified t_begin = 0 while having Hubble expansion enabled. '
                    'Please specify some finite starting time or disable Hubble expansion.'
                )
            t_begin_correct = t_begin
            a_begin_correct = scale_factor(t_begin_correct)
        else:
            # Neither a_begin nor t_begin is specified.
            # One or the other is needed when enable_Hubble is True.
            abort('No initial scale factor (a_begin) or initial cosmic time (t_begin) specified. '
                  'A specification of one or the other is needed when enable_Hubble is True.')
    else:
        # Hubble expansion disabled
        t_begin_correct = t_begin
        # Values of the scale factor (and therefore a_begin)
        # are meaningless. Set a_begin to unity,
        # effectively ignoring its existence.
        a_begin_correct = 1.0
        if 'a_begin' in user_params_keys_raw:
            masterwarn('Ignoring a_begin = {} because enable_Hubble is False'.format(a_begin))
    # Now t_begin_correct and a_begin_correct are defined and store
    # the actual values of the initial time and scale factor.
    # Assign these correct values to the corresponding
    # universal variables.
    universals.t_begin = t_begin_correct
    universals.a_begin = a_begin_correct
    universals.z_begin = 1/a_begin_correct - 1
    # Initiate the current universal time and scale factor
    universals.t = t_begin_correct
    universals.a = a_begin_correct
cython.declare(time_initialized='bint')
time_initialized = False



# Global Spline objects defined by init_time
cython.declare(spline_a_t='Spline', spline_t_a='Spline', spline_a_H='Spline')
spline_a_t = None
spline_t_a = None
spline_a_H = None
