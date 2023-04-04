#ifdef __SSE3__
    #define HAS_SIMD 1
    #include <x86intrin.h>

    typedef __m128d simd_float64_t;
    typedef __m128 simd_float32_t;

    inline simd_float32_t abs_ps(simd_float32_t x) {
        const simd_float32_t sign_mask = _mm_set1_ps(-0.f); // -0.f = 1 << 31
        return _mm_andnot_ps(sign_mask, x);
    }

    inline simd_float64_t abs_pd(simd_float64_t x) {
        const simd_float64_t sign_mask = _mm_set1_pd(-0.f); // -0.f = 1 << 63
        return _mm_andnot_pd(sign_mask, x);
    }

    float simd_manhattan32(const float* x, const float* y, ssize_t t) {
        simd_float32_t simd_x_1, simd_x_2;
        simd_float32_t simd_y_1, simd_y_2;
        ssize_t loop_width = 8; // Two SIMD registers can hold eight floats
        ssize_t remainder = t % loop_width;
        ssize_t n_iter = t - remainder;
        ssize_t idx;

        simd_float32_t sum_1 = _mm_setzero_ps();
        simd_float32_t sum_2 = _mm_setzero_ps();
        for(idx = 0; idx < n_iter; idx += loop_width) {
            simd_x_1 = _mm_set_ps(x[idx], x[idx + 1], x[idx + 2], x[idx + 3]);
            simd_y_1 = _mm_set_ps(y[idx], y[idx + 1], y[idx + 2], y[idx + 3]);
            sum_1 += abs_ps(simd_x_1 - simd_y_1);

            simd_x_2 = _mm_set_ps(x[idx + 4], x[idx + 5], x[idx + 6], x[idx + 7]);
            simd_y_2 = _mm_set_ps(y[idx + 4], y[idx + 5], y[idx + 6], y[idx + 7]);
            sum_2 += abs_ps(simd_x_2 - simd_y_2);
        }

        sum_1 += sum_2;
        simd_float32_t hsum = _mm_hadd_ps(sum_1, sum_1);
        hsum = _mm_hadd_ps(hsum, hsum);
        float output_sum = *(float*)&hsum;
        for(idx = n_iter; idx < t; idx++){
            output_sum += fabs(x[idx] - y[idx]);
        }

        return output_sum;
    }

    double simd_manhattan(const double* x, const double* y, ssize_t t) {
        simd_float64_t simd_x_1, simd_x_2;
        simd_float64_t simd_y_1, simd_y_2;
        ssize_t loop_width = 4; // Two SIMD registers can hold four doubles
        ssize_t remainder = t % loop_width;
        ssize_t n_iter = t - remainder;
        ssize_t idx;

        simd_float64_t sum_1 = _mm_setzero_pd();
        simd_float64_t sum_2 = _mm_setzero_pd();
        for(idx = 0; idx < n_iter; idx += loop_width) {
            simd_x_1 = _mm_set_pd(x[idx], x[idx + 1]);
            simd_y_1 = _mm_set_pd(y[idx], y[idx + 1]);
            sum_1 += abs_pd(simd_x_1 - simd_y_1);

            simd_x_2 = _mm_set_pd(x[idx + 2], x[idx + 3]);
            simd_y_2 = _mm_set_pd(y[idx + 2], y[idx + 3]);
            sum_2 += abs_pd(simd_x_2 - simd_y_2);
        }
        sum_1 += sum_2;
        simd_float64_t hsum = _mm_hadd_pd(sum_1, sum_1);
        double output_sum = *(double*)&hsum;
        for(idx = n_iter; idx < t; idx++){
            output_sum += fabs(x[idx] - y[idx]);
        }

        return output_sum;
    }
#else
    #define HAS_SIMD 0

    double simd_manhattan(double* x, double* y, ssize_t t) {return -1.f;}
    float simd_manhattan32(float* x, float* y, ssize_t t) {return -1.;}
#endif
