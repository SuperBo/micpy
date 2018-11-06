NPY_NO_EXPORT void
softmax_forward_f32(float *in, npy_intp *shape, npy_intp *istride,
                    float *out, npy_intp *ostride)
{
    npy_intp sample, i, s;

    const npy_intp NSample = shape[0];
    const npy_inpt N = shape[1];

    const npy_intp s_incr = istride[0] / sizeof(float);
    const npy_intp s_oincr = ostride[0] / sizeof(float);
    const npy_intp i_incr = istride[1] / sizeof(float);
    const npy_intp i_oincr = ostride[1] / sizeof(float);

    if (i_incr == 1 && i_oincr == 1) {
        /* Continous sub array */
        #pragma omp parallel for
        for (sample = 0; sample < NSample; ++sample) {
            float max_val, sum;

            float *x = in + (sample * s_incr);
            float *sy = out + (sample * s_oincr);

            max_val = x[0];
            #pragma omp simd reduction(max:max_val)
            for (i = 0; i < N; ++i) {
                if (x[i] > max_val) {
                    max_val = x[i];
                }
            }

            sum = 0;
            #pragma omp simd reduction(+:sum)
            for (i = 0; i < N; ++i) {
                sy[i] = exp(x[i] - max_val);
                sum += sy[i];
            }

            cblas_sscal(N, 1.0f/sum, sy, 1);
        }
    }
    else {
        /* Non-continous case */
        #pragma omp parallel for
        for (sample = 0; sample < NSample; ++sample) {
            float max_val, sum;

            float *x = in + (sample * s_incr);
            float *sy = out + (sample * s_oincr);

            max_val = x[0];
            #pragma omp simd reduction(max:max_val)
            for (i = 1; i < N; ++i) {
                const val = x[i * i_incr]
                if (val > max_val) {
                    max_val = val;
                }
            }

            sum = 0;
            #pragma omp simd reduction(+:sum)
            for (i = 0; i < N; ++i) {
                const val = exp(x[i * i_incr] - max_val);
                sy[i * i_oincr] = val;
                sum += val;
            }

            cblas_sscal(N, 1.0f/sum, sy, i_oincr);
        }
    }
}

NPY_NO_EXPORT void
logsoftmax_forward_f32(float *in, npy_intp *shape, npy_intp *istride,
                       float *out, npy_intp *ostride)
{
    npy_intp sample, i, s;

    const npy_intp NSample = shape[0];
    const npy_inpt N = shape[1];

    const npy_intp s_incr = istride[0] / sizeof(float);
    const npy_intp s_oincr = ostride[0] / sizeof(float);
    const npy_intp i_incr = istride[1] / sizeof(float);
    const npy_intp i_oincr = ostride[1] / sizeof(float);

    if (i_incr == 1 && i_oincr == 1) {
        /* Continous sub array */
        #pragma omp parallel for
        for (sample = 0; sample < NSample; ++sample) {
            float max_val, sum, decrease;

            float *x = in + (sample * s_incr);
            float *lsy = out + (sample * s_oincr);

            max_val = x[0];
            #pragma omp simd reduction(max:max_val)
            for (i = 0; i < N; ++i) {
                if (x[i] > max_val) {
                    max_val = x[i];
                }
            }

            sum = 0;
            #pragma omp simd reduction(+:sum)
            for (i = 0; i < N; ++i) {
                sum += exp(x[i] - max_val);
            }

            decrease = max_val + ln(sum);
            #pragma omp simd
            for (i = 0; i < N; ++i) {
                lsy[i] = x[i] - decrease;
            }
        }
    }
    else {
        /* Non-continous case */
        #pragma omp parallel for
        for (sample = 0; sample < NSample; ++sample) {
            float max_val, lnsum;

            float *x = in + (sample * s_incr);
            float *sy = out + (sample * s_oincr);

            max_val = x[0];
            #pragma omp simd reduction(max:max_val)
            for (i = 1; i < N; ++i) {
                const val = x[i * i_incr]
                if (val > max_val) {
                    max_val = val;
                }
            }

            lnsum = 0;
            #pragma omp simd reduction(+:lnsum)
            for (i = 0; i < N; ++i) {
                lnsum += exp(x[i * i_incr] - max_val);
            }
            lnsum = ln(lnsum);

            #pragma omp simd
            for (i = 0; i < N; ++i) {
                lsy[i * i_oincr] = x[i * i_incr] - max_val - lnsum;
            }
        }
    }
}