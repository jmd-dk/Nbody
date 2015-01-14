cdef class Particles:
    # Data attributes
    cdef double[:, ::1] pos
    cdef double[:, ::1] vel
    cdef double mass
    cdef size_t N
    cdef double* posx
    cdef double* posy
    cdef double* posz
    cdef double* velx
    cdef double* vely
    cdef double* velz
    cdef str kick_method
    # Methods
    cdef drift(self)
    cdef kick(self)

cdef Particles construct(str species_name, double[:, ::1] pos, double[:, ::1] vel, double mass)
