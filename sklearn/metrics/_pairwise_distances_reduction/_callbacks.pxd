from ...utils._typedefs cimport DTYPE_t, ITYPE_t

##############################################
# IMPORTANT: Each callback must define its own explicit struct that is
# implicitly passed through `callback_args`.

# Generic typedef for use in _engines.{pyx, pxd}
ctypedef void (*CALLBACK)(
    ITYPE_t,
    ITYPE_t,
    ITYPE_t,
    ITYPE_t,
    ITYPE_t,
    ITYPE_t,
    ITYPE_t,
    DTYPE_t,
    DTYPE_t *,
    ITYPE_t *,
    void *,
) nogil

##############################################

cdef void argkmin_callback(
        ITYPE_t X_start,
        ITYPE_t X_end,
        ITYPE_t Y_start,
        ITYPE_t Y_end,
        ITYPE_t thread_num,
        ITYPE_t i,
        ITYPE_t j,
        DTYPE_t val,
        DTYPE_t * heaps_r_distances,
        ITYPE_t * heaps_indices,
        void * callback_args,
) nogil

cdef struct ARGKMIN_CALLBACK_ARGS:
    ITYPE_t k

##############################################
