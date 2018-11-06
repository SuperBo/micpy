NPY_NO_EXPORT void
cross_forward_f32(float *data, npy_intp *shape, npy_intp *stride,
                  int *label, npy_intp *lstride,
                  float *coeff, float *entropy)
{
    npy_intp sample, i;

    const npy_intp NSample = shape[0];
    const npy_intp NLabel = shape[1];

    const npy_intp s_incr = stride[0] / sizeof(float);
    const npy_intp s_iincr = stride[1] / sizeof(float);
    const npy_intp t_incr = lstride[0] / sizeof(float);

    float sum = 0.0f;

    for (sample = 0; sample < NSample; ++sample) {
        float *x = data + (s_incr * sample);
        int *t = label + (t_incr * sample);

        const val = (*t < 0 || *t >NLabel) ? x[0] : x[s_iincr * (*t)];

        sum -= val;
    }

    if (coeff != NULL && *coeff != 1.0f) {
        *entropy = sum / *coeff;
    }
    else {
        *entropy = sum;
    }
}

NPY_NO_EXPORT void
cross_backward_f32(float *data, npy_intp *shape, npy_intp *stride,
                   int *label, npy_intp *lstride,
                   float *ydata, npy_intp *ystride
                   float *coeff)
{
    npy_intp sample, i;

    const npy_intp NSample = shape[0];
    const npy_intp NLabel = shape[1];

    const npy_intp s_incr = stride[0] / sizeof(float);
    const npy_intp s_iincr = stride[1] / sizeof(float);
    const npy_intp t_incr = lstride[0] / sizeof(float);

    float sum = 0.0f;

    for (sample = 0; sample < NSample; ++sample) {
        float *x = data + (s_incr * sample);
        float *y = ydata + (s_incr * sample);
        int *t = label + (t_incr * sample);

        if (*t < 0 || *t >NLabel) {
            for (i = 0; i < NLabel; ++i) {
                y[i*s_iincr] = 0.0f;
            }
        }
        else {
            y[s_iincr * (*t)] -= 1.0f;
        }
    }

    if (coeff != NULL && *coeff != 1.0f) {
        cblas_sscal(NSample * NLabel, *coeff, ydata, 1);
    }
}