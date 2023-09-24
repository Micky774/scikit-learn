// #define HWY_TARGETS HWY_AVX2
#include <cmath>
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd.cpp"
#include "hwy/foreach_target.h"
#include "hwy/highway.h"
HWY_BEFORE_NAMESPACE();  // at file scope

namespace manhattan {

    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;


        template <typename Type>
        inline Type manhattan_dist(
            const Type* x,
            const Type* y,
            const size_t size
        ) {
            const hn::ScalableTag<Type> d;
            using batch_type = decltype(hn::Zero(d));

            batch_type simd_sum_1 = hn::Zero(d);
            batch_type simd_x_1;
            batch_type simd_y_1;

            batch_type simd_sum_2 = hn::Zero(d);
            batch_type simd_x_2;
            batch_type simd_y_2;

            size_t lane_size = hn::Lanes(d);
            size_t loop_iter = lane_size * 2;
            size_t vec_size = size - size % loop_iter;
            size_t vec_remainder_size = size - size % lane_size;

            for (size_t i = 0; i < vec_size; i += loop_iter) {
                simd_x_1 = hn::LoadU(d, x + i);
                simd_y_1 = hn::LoadU(d, y + i);
                simd_sum_1 += hn::AbsDiff(simd_x_1, simd_y_1);

                simd_x_2 = hn::LoadU(d, x + i + lane_size);
                simd_y_2 = hn::LoadU(d, y + i + lane_size);
                simd_sum_2 += hn::AbsDiff(simd_x_2, simd_y_2);
            }
            for (size_t i = vec_size; i < vec_remainder_size; i += loop_iter) {
                simd_x_1 = hn::LoadU(d, x + i);
                simd_y_1 = hn::LoadU(d, y + i);
                simd_sum_1 += hn::AbsDiff(simd_x_1, simd_y_1);
            }
            simd_sum_1 += simd_sum_2;
            Type scalar_sum = hn::ReduceSum(d, simd_sum_1);
            for (size_t i = vec_remainder_size; i < size; i += 1) {
                scalar_sum += fabs(x[i] - y[i]);
            }
            return scalar_sum;
        }
        float manhattan_dist_f32(
            const float* x,
            const float* y,
            const size_t size
        ) {
            return manhattan_dist<float>(x, y, size);
        }
        double manhattan_dist_f64(
            const double* x,
            const double* y,
            const size_t size
        ) {
            return manhattan_dist<double>(x, y, size);
        }
    }
}

namespace hgbt {

    namespace HWY_NAMESPACE {
        namespace hn = hwy::HWY_NAMESPACE;

        void _build_histogram_root(
            const uint32_t n_samples,
            const uint32_t* binned_features,
            const float* all_gradients,
            const float* all_hessians,
            hist_struct_alt out
        ){
            uint32_t i = 0;

            const hn::ScalableTag<uint8_t> d_uint8;
            const hn::ScalableTag<uint32_t> d_uint32;
            const hn::ScalableTag<float> d_float;
            using batch_uint32 = decltype(hn::Zero(d_uint32));
            using batch_float = decltype(hn::Zero(d_float));

            uint32_t vec_size = hn::Lanes(d_float);
            uint32_t unrolled_upper = (n_samples / vec_size) * vec_size;

            uint32_t bin_idx;
            batch_uint32 bins;

            batch_float simd_sum_gradients;
            batch_float simd_all_gradients;
            batch_float simd_sum_hessians;
            batch_float simd_all_hessians;
            batch_uint32 simd_count;

            batch_uint32 ones = Set(d_uint32, 1);

            for(int i = 0; i < unrolled_upper; i+=vec_size){
                bins = hn::Load(d_uint32, binned_features);

                //TODO: I believe that GatherIndex requires indices of type in32_t for
                // an output simd vector of type double (tag d_float), however we initially
                // expect bins to be of type uint8_t (uint32_t right now for convenience)
                simd_sum_gradients = hn::GatherIndex(d_float, out.sum_gradients, bins);
                simd_all_gradients = hn::Load(d_float, &all_gradients[i]);
                simd_sum_gradients += simd_all_gradients;

                simd_sum_hessians = hn::GatherIndex(d_float, out.sum_hessians, bins);
                simd_all_hessians = hn::Load(d_float, &all_hessians[i]);
                simd_all_hessians += simd_all_hessians;

                simd_count = hn::GatherIndex(d_uint32, out.count, bins);
                simd_count += ones;
            }

            for(int i = unrolled_upper; i < n_samples; i+=1){
                bin_idx = binned_features[i];
                out.sum_gradients[bin_idx] += all_gradients[i];
                out.sum_hessians[bin_idx] += all_hessians[i];
                out.count[bin_idx] += 1;
            }
        }
    }
}
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
#include "simd.hpp"

namespace manhattan {

    HWY_EXPORT(manhattan_dist_f32);
    HWY_EXPORT(manhattan_dist_f64);

    HWY_DLLEXPORT float simd_manhattan_dist_f32(
        const float* x,
        const float* y,
        const size_t size
    ){
        return HWY_DYNAMIC_DISPATCH(manhattan_dist_f32)(x,  y, size);
    }
    HWY_DLLEXPORT double simd_manhattan_dist_f64(
        const double* x,
        const double* y,
        const size_t size
    ){
        return HWY_DYNAMIC_DISPATCH(manhattan_dist_f64)(x,  y, size);
    }
}

namespace hgbt{
    HWY_EXPORT(_build_histogram_root);

    HWY_DLLEXPORT void simd_build_histogram_root(
        const uint32_t n_samples,
        const uint32_t* binned_features,
        const float* all_gradients,
        const float* all_hessians,
        hist_struct_alt out
    ){
        return HWY_DYNAMIC_DISPATCH(_build_histogram_root)(
            n_samples,
            binned_features,
            all_gradients,
            all_hessians,
            out
        );
    }

}
#endif  // HWY_ONCE
