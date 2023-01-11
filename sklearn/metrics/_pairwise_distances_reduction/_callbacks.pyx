from ...utils._typedefs cimport DTYPE_t, ITYPE_t
from ...utils._heap cimport heap_push

cdef inline void argkmin_callback(
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
) nogil:
    cdef ITYPE_t k = (<ARGKMIN_CALLBACK_ARGS *> callback_args).k
    heap_push(
        values=heaps_r_distances + i * k,
        indices=heaps_indices + i * k,
        size=k,
        val=val,
        val_idx=j + Y_start,
    )
