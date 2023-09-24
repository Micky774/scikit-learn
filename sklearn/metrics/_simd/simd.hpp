#ifndef SIMD_HPP
#define SIMD_HPP
#include <hwy/base.h>

namespace manhattan{

    HWY_DLLEXPORT float simd_manhattan_dist_f32(
        const float* x,
        const float* y,
        const size_t size
    );
    HWY_DLLEXPORT double simd_manhattan_dist_f64(
        const double* x,
        const double* y,
        const size_t size
    );
}

namespace hgbt
{
    struct hist_struct{
        double sum_gradients;
        double sum_hessians;
        unsigned int count;
    };

    struct hist_struct_alt{
        double* sum_gradients;
        double* sum_hessians;
        unsigned int* count;
    };
    typedef struct hist_struct_alt hist_struct_alt;

    HWY_DLLEXPORT float simd_build_histogram_root(
        const float* x,
        const float* y,
        const size_t size
    );

} // namespace hgbt

#endif /*SIMD_HPP*/
