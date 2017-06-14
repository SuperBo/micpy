/*
 * This module provides a BLAS optimized matrix multiply,
 * inner product and dot for numpy arrays
 */

#define NPY_NO_DEPRECATED_API NPY_API_VERSION

#include <Python.h>
#include <assert.h>


#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#define NO_IMPORT_ARRAY
#include <numpy/arrayobject.h>

#pragma omp declare target
#include "npy_cblas.h"
#pragma omp end declare target

#define _MICARRAYMODULE
#include "mpyndarraytypes.h"
#include "arraytypes.h"
#include "common.h"
#include "mpymem_overlap.h"
#include "convert.h"
#include "creators.h"
#include "scalar.h"

/* These might be faster without the dereferencing of obj
   going on inside -- of course an optimizing compiler should
   inline the constants inside a for loop making it a moot point
*/

#define Array_GETPTR1(data, strides, i) ((void *)((char *) data + \
                                         (i)*strides[0]))

#define Array_GETPTR2(data, strides, i, j) ((void *)((char *) data + \
                                            (i)*strides[0] + \
                                            (j)*strides[1]))

#define Array_GETPTR3(data, strides, i, j, k) ((void *)((char *) data + \
                                            (i)*strides[0] + \
                                            (j)*strides[1] + \
                                            (k)*strides[2]))

#define Array_GETPTR4(data, strides, i, j, k, l) ((void *)((char *) data + \
                                            (i)*strides[0] + \
                                            (j)*strides[1] + \
                                            (k)*strides[2] + \
                                            (l)*strides[3]))

/*
 * Helper: call appropriate BLAS dot function for typenum.
 * Strides are NumPy strides.
 */
static void
blas_dot(int device, int typenum, npy_intp n,
         void *a, npy_intp stridea, void *b, npy_intp strideb, void *res)
{
    switch (typenum) {
        case NPY_DOUBLE:
            DOUBLE_dot(a, stridea, b, strideb, res, n, device);
            break;
        case NPY_FLOAT:
            FLOAT_dot(a, stridea, b, strideb, res, n, device);
            break;
        case NPY_CDOUBLE:
            CDOUBLE_dot(a, stridea, b, strideb, res, n, device);
            break;
        case NPY_CFLOAT:
            CFLOAT_dot(a, stridea, b, strideb, res, n, device);
            break;
    }
}

#pragma omp declare target

static const double oneD[2] = {1.0, 0.0}, zeroD[2] = {0.0, 0.0};
static const float oneF[2] = {1.0, 0.0}, zeroF[2] = {0.0, 0.0};

#pragma omp end declare target


/*
 * Helper: dispatch to appropriate cblas_?gemm for typenum.
 */
static void
gemm(int typenum, enum CBLAS_ORDER order,
     enum CBLAS_TRANSPOSE transA, enum CBLAS_TRANSPOSE transB,
     int m, int n, int k,
     PyMicArrayObject *A, int lda, PyMicArrayObject *B, int ldb, PyMicArrayObject *R)
{
    const void *Adata = PyMicArray_DATA(A), *Bdata = PyMicArray_DATA(B);
    void *Rdata = PyMicArray_DATA(R);
    int ldc = PyMicArray_DIM(R, 1) > 1 ? PyMicArray_DIM(R, 1) : 1;

    int device = PyMicArray_DEVICE(A);

#pragma omp target device(device) map(to: typenum, order, transA, transB, m, n, k, \
                                    Adata, lda, Bdata, ldb, Rdata, ldc)
    switch (typenum) {
        case NPY_DOUBLE:
            cblas_dgemm(order, transA, transB, m, n, k, 1.,
                        Adata, lda, Bdata, ldb, 0., Rdata, ldc);
            break;
        case NPY_FLOAT:
            cblas_sgemm(order, transA, transB, m, n, k, 1.f,
                        Adata, lda, Bdata, ldb, 0.f, Rdata, ldc);
            break;
        case NPY_CDOUBLE:
            cblas_zgemm(order, transA, transB, m, n, k, oneD,
                        Adata, lda, Bdata, ldb, zeroD, Rdata, ldc);
            break;
        case NPY_CFLOAT:
            cblas_cgemm(order, transA, transB, m, n, k, oneF,
                        Adata, lda, Bdata, ldb, zeroF, Rdata, ldc);
            break;
    }
}


/*
 * Helper: dispatch to appropriate cblas_?gemv for typenum.
 */
static void
gemv(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     PyMicArrayObject *A, int lda, PyMicArrayObject *X, int incX,
     PyMicArrayObject *R)
{
    const void *Adata = PyMicArray_DATA(A), *Xdata = PyMicArray_DATA(X);
    void *Rdata = PyMicArray_DATA(R);

    int m = PyMicArray_DIM(A, 0), n = PyMicArray_DIM(A, 1);
    int device = PyMicArray_DEVICE(A);

#pragma omp target device(device) map(to: typenum, order, trans, m, n, \
                                    Adata, lda, Xdata, incX, Rdata)
    switch (typenum) {
        case NPY_DOUBLE:
            cblas_dgemv(order, trans, m, n, 1., Adata, lda, Xdata, incX,
                        0., Rdata, 1);
            break;
        case NPY_FLOAT:
            cblas_sgemv(order, trans, m, n, 1.f, Adata, lda, Xdata, incX,
                        0.f, Rdata, 1);
            break;
        case NPY_CDOUBLE:
            cblas_zgemv(order, trans, m, n, oneD, Adata, lda, Xdata, incX,
                        zeroD, Rdata, 1);
            break;
        case NPY_CFLOAT:
            cblas_cgemv(order, trans, m, n, oneF, Adata, lda, Xdata, incX,
                        zeroF, Rdata, 1);
            break;
    }
}


/*
 * Helper: dispatch to appropriate cblas_?syrk for typenum.
 */
static void
syrk(int typenum, enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans,
     int n, int k,
     PyMicArrayObject *A, int lda, PyMicArrayObject *R)
{
    const void *Adata = PyMicArray_DATA(A);
    void *Rdata = PyMicArray_DATA(R);
    npy_intp *Rstrides = PyMicArray_STRIDES(R);
    int ldc = PyMicArray_DIM(R, 1) > 1 ? PyMicArray_DIM(R, 1) : 1;
    int device = PyMicArray_DEVICE(A);

    npy_intp i;
    npy_intp j;

#pragma omp target device(device) map(to: typenum, order, trans, n, k, \
                                  Adata, lda, ldc, Rdata, Rstrides[0:2])
    switch (typenum) {
        case NPY_DOUBLE:
            cblas_dsyrk(order, CblasUpper, trans, n, k, 1.,
                        Adata, lda, 0., Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_double*)Array_GETPTR2(Rdata, Rstrides, j, i)) =
                            *((npy_double*)Array_GETPTR2(Rdata, Rstrides, i, j));
                }
            }
            break;
        case NPY_FLOAT:
            cblas_ssyrk(order, CblasUpper, trans, n, k, 1.f,
                        Adata, lda, 0.f, Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_float*)Array_GETPTR2(Rdata, Rstrides, j, i)) =
                            *((npy_float*)Array_GETPTR2(Rdata, Rstrides, i, j));
                }
            }
            break;
        case NPY_CDOUBLE:
            cblas_zsyrk(order, CblasUpper, trans, n, k, oneD,
                        Adata, lda, zeroD, Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_cdouble*)Array_GETPTR2(Rdata, Rstrides, j, i)) =
                            *((npy_cdouble*)Array_GETPTR2(Rdata, Rstrides, i, j));
                }
            }
            break;
        case NPY_CFLOAT:
            cblas_csyrk(order, CblasUpper, trans, n, k, oneF,
                        Adata, lda, zeroF, Rdata, ldc);

            for (i = 0; i < n; i++) {
                for (j = i + 1; j < n; j++) {
                    *((npy_cfloat*)Array_GETPTR2(Rdata, Rstrides, j, i)) =
                            *((npy_cfloat*)Array_GETPTR2(Rdata, Rstrides, i, j));
                }
            }
            break;
    }
}


typedef enum {_scalar, _column, _row, _matrix} MatrixShape;


static MatrixShape
_select_matrix_shape(PyMicArrayObject *array)
{
    switch (PyMicArray_NDIM(array)) {
        case 0:
            return _scalar;
        case 1:
            if (PyMicArray_DIM(array, 0) > 1)
                return _column;
            return _scalar;
        case 2:
            if (PyMicArray_DIM(array, 0) > 1) {
                if (PyMicArray_DIM(array, 1) == 1)
                    return _column;
                else
                    return _matrix;
            }
            if (PyMicArray_DIM(array, 1) == 1)
                return _scalar;
            return _row;
    }
    return _matrix;
}


/*
 * This also makes sure that the data segment is aligned with
 * an itemsize address as well by returning one if not true.
 */
static int
_bad_strides(PyMicArrayObject *ap)
{
    int itemsize = PyMicArray_ITEMSIZE(ap);
    int i, N=PyMicArray_NDIM(ap);
    npy_intp *strides = PyMicArray_STRIDES(ap);

    if (((npy_intp)(PyMicArray_DATA(ap)) % itemsize) != 0) {
        return 1;
    }
    for (i = 0; i < N; i++) {
        if ((strides[i] < 0) || (strides[i] % itemsize) != 0) {
            return 1;
        }
    }

    return 0;
}

/*
 * dot(a,b)
 * Returns the dot product of a and b for arrays of floating point types.
 * Like the generic numpy equivalent the product sum is over
 * the last dimension of a and the second-to-last dimension of b.
 * NB: The first argument is not conjugated.;
 *
 * This is for use by PyArray_MatrixProduct2. It is assumed on entry that
 * the arrays ap1 and ap2 have a common data type given by typenum that is
 * float, double, cfloat, or cdouble and have dimension <= 2. The
 * __array_ufunc__ nonsense is also assumed to have been taken care of.
 */
NPY_NO_EXPORT PyObject *
cblas_matrixproduct(int typenum, PyMicArrayObject *ap1, PyMicArrayObject *ap2,
                    PyMicArrayObject *out)
{
    PyMicArrayObject *result = NULL, *out_buf = NULL;
    int j, lda, ldb;
    npy_intp l;
    int nd;
    npy_intp ap1stride = 0;
    npy_intp dimensions[NPY_MAXDIMS];
    npy_intp numbytes;
    double prior1, prior2;
    PyTypeObject *subtype;
    MatrixShape ap1shape, ap2shape;
    void *tmpdata;
    int device = PyMicArray_DEVICE(ap1); // Assume on the same device

    if (_bad_strides(ap1)) {
            PyObject *op1 = PyMicArray_NewCopy(ap1, NPY_ANYORDER);

            Py_DECREF(ap1);
            ap1 = (PyMicArrayObject *)op1;
            if (ap1 == NULL) {
                goto fail;
            }
    }
    if (_bad_strides(ap2)) {
            PyObject *op2 = PyMicArray_NewCopy(ap2, NPY_ANYORDER);

            Py_DECREF(ap2);
            ap2 = (PyMicArrayObject *)op2;
            if (ap2 == NULL) {
                goto fail;
            }
    }
    ap1shape = _select_matrix_shape(ap1);
    ap2shape = _select_matrix_shape(ap2);

    if (ap1shape == _scalar || ap2shape == _scalar) {
        PyMicArrayObject *oap1, *oap2;
        oap1 = ap1; oap2 = ap2;
        /* One of ap1 or ap2 is a scalar */
        if (ap1shape == _scalar) {
            /* Make ap2 the scalar */
            PyMicArrayObject *t = ap1;
            ap1 = ap2;
            ap2 = t;
            ap1shape = ap2shape;
            ap2shape = _scalar;
        }

        if (ap1shape == _row) {
            ap1stride = PyMicArray_STRIDE(ap1, 1);
        }
        else if (PyMicArray_NDIM(ap1) > 0) {
            ap1stride = PyMicArray_STRIDE(ap1, 0);
        }

        if (PyMicArray_NDIM(ap1) == 0 || PyMicArray_NDIM(ap2) == 0) {
            npy_intp *thisdims;
            if (PyMicArray_NDIM(ap1) == 0) {
                nd = PyMicArray_NDIM(ap2);
                thisdims = PyMicArray_DIMS(ap2);
            }
            else {
                nd = PyMicArray_NDIM(ap1);
                thisdims = PyMicArray_DIMS(ap1);
            }
            l = 1;
            for (j = 0; j < nd; j++) {
                dimensions[j] = thisdims[j];
                l *= dimensions[j];
            }
        }
        else {
            l = PyMicArray_DIM(oap1, PyMicArray_NDIM(oap1) - 1);

            if (PyMicArray_DIM(oap2, 0) != l) {
                dot_alignment_error(oap1, PyMicArray_NDIM(oap1) - 1, oap2, 0);
                goto fail;
            }
            nd = PyMicArray_NDIM(ap1) + PyMicArray_NDIM(ap2) - 2;
            /*
             * nd = 0 or 1 or 2. If nd == 0 do nothing ...
             */
            if (nd == 1) {
                /*
                 * Either PyArray_NDIM(ap1) is 1 dim or PyArray_NDIM(ap2) is
                 * 1 dim and the other is 2 dim
                 */
                dimensions[0] = (PyMicArray_NDIM(oap1) == 2) ?
                                PyMicArray_DIM(oap1, 0) : PyMicArray_DIM(oap2, 1);
                l = dimensions[0];
                /*
                 * Fix it so that dot(shape=(N,1), shape=(1,))
                 * and dot(shape=(1,), shape=(1,N)) both return
                 * an (N,) array (but use the fast scalar code)
                 */
            }
            else if (nd == 2) {
                dimensions[0] = PyMicArray_DIM(oap1, 0);
                dimensions[1] = PyMicArray_DIM(oap2, 1);
                /*
                 * We need to make sure that dot(shape=(1,1), shape=(1,N))
                 * and dot(shape=(N,1),shape=(1,1)) uses
                 * scalar multiplication appropriately
                 */
                if (ap1shape == _row) {
                    l = dimensions[1];
                }
                else {
                    l = dimensions[0];
                }
            }

            /* Check if the summation dimension is 0-sized */
            if (PyMicArray_DIM(oap1, PyMicArray_NDIM(oap1) - 1) == 0) {
                l = 0;
            }
        }
    }
    else {
        /*
         * (PyArray_NDIM(ap1) <= 2 && PyArray_NDIM(ap2) <= 2)
         * Both ap1 and ap2 are vectors or matrices
         */
        l = PyMicArray_DIM(ap1, PyMicArray_NDIM(ap1) - 1);

        if (PyMicArray_DIM(ap2, 0) != l) {
            dot_alignment_error(ap1, PyMicArray_NDIM(ap1) - 1, ap2, 0);
            goto fail;
        }
        nd = PyMicArray_NDIM(ap1) + PyMicArray_NDIM(ap2) - 2;

        if (nd == 1) {
            dimensions[0] = (PyMicArray_NDIM(ap1) == 2) ?
                            PyMicArray_DIM(ap1, 0) : PyMicArray_DIM(ap2, 1);
        }
        else if (nd == 2) {
            dimensions[0] = PyMicArray_DIM(ap1, 0);
            dimensions[1] = PyMicArray_DIM(ap2, 1);
        }
    }

    /* Choose which subtype to return */
    if (Py_TYPE(ap1) != Py_TYPE(ap2)) {
        prior2 = PyArray_GetPriority((PyObject *)ap2, 0.0);
        prior1 = PyArray_GetPriority((PyObject *)ap1, 0.0);
        subtype = (prior2 > prior1 ? Py_TYPE(ap2) : Py_TYPE(ap1));
    }
    else {
        prior1 = prior2 = 0.0;
        subtype = Py_TYPE(ap1);
    }

    if (out != NULL) {
        int d;

        /* verify that out is usable */
        if (Py_TYPE(out) != subtype ||
            PyMicArray_NDIM(out) != nd ||
            PyMicArray_TYPE(out) != typenum ||
            !PyMicArray_ISCARRAY(out)) {

            PyErr_SetString(PyExc_ValueError,
                "output array is not acceptable "
                "(must have the right type, nr dimensions, and be a C-Array)");
            goto fail;
        }
        for (d = 0; d < nd; ++d) {
            if (dimensions[d] != PyMicArray_DIM(out, d)) {
                PyErr_SetString(PyExc_ValueError,
                    "output array has wrong dimensions");
                goto fail;
            }
        }

        /* check for memory overlap */
        if (!(solve_may_share_memory(out, ap1, 1) == 0 &&
              solve_may_share_memory(out, ap2, 1) == 0)) {
            /* allocate temporary output array */
            out_buf = (PyMicArrayObject *)PyMicArray_NewLikeArray(device, (PyArrayObject *)out,
                                                                  NPY_CORDER, NULL, 0);
            if (out_buf == NULL) {
                goto fail;
            }

            /* set copy-back */
            /* TODO: check whether SetUpdateIfCopyBase work normally */
            Py_INCREF(out);
            if (PyMicArray_SetUpdateIfCopyBase(out_buf, out) < 0) {
                Py_DECREF(out);
                goto fail;
            }
        }
        else {
            Py_INCREF(out);
            out_buf = out;
        }
        Py_INCREF(out);
        result = out;
    }
    else {
        PyObject *tmp = (PyObject *)(prior2 > prior1 ? ap2 : ap1);

        out_buf = (PyMicArrayObject *)PyMicArray_New(device, subtype, nd, dimensions,
                                               typenum, NULL, NULL, 0, 0, tmp);
        if (out_buf == NULL) {
            goto fail;
        }

        Py_INCREF(out_buf);
        result = out_buf;
    }

    numbytes = PyMicArray_NBYTES(out_buf);
    target_memset(PyMicArray_DATA(out_buf), 0, numbytes, device);
    if (numbytes == 0 || l == 0) {
            Py_DECREF(ap1);
            Py_DECREF(ap2);
            return PyMicArray_Return(out_buf);
    }

    /* Prepare for offloading */
    void *ap1data = PyMicArray_DATA(ap1);
    void *ap2data = PyMicArray_DATA(ap2);
    void *outdata = PyMicArray_DATA(out_buf);
    int ap1ndim = PyMicArray_NDIM(ap1);
    int ap2ndim = PyMicArray_NDIM(ap2);
    int outndim = PyMicArray_NDIM(out_buf);
    npy_intp *ap1dims = PyMicArray_DIMS(ap1);
    npy_intp *ap2dims = PyMicArray_DIMS(ap2);
    npy_intp *outdims = PyMicArray_DIMS(out_buf);
    npy_intp *ap1strides_ptr = PyMicArray_STRIDES(ap1);
    npy_intp *ap2strides_ptr = PyMicArray_STRIDES(ap2);
    npy_intp *outstrides_ptr = PyMicArray_STRIDES(out_buf);

    if (ap2shape == _scalar) {
        /*
         * Multiplication by a scalar -- Level 1 BLAS
         * if ap1shape is a matrix and we are not contiguous, then we can't
         * just blast through the entire array using a single striding factor
         */
        NPY_BEGIN_ALLOW_THREADS;

        if (typenum == NPY_DOUBLE) {
            if (l == 1) {
                #pragma omp target device(device) map(to: outdata, ap1data, ap2data)
                *((double *)outdata) = *((double *)ap2data) * *((double *)ap1data);
            }
            else if (ap1shape != _matrix) {
                #pragma omp target device(device) map(to: l, ap2data, ap1data, \
                                                        ap1stride, outdata)
                cblas_daxpy(l,
                            *((double *)ap2data),
                            (double *)ap1data,
                            ap1stride/sizeof(double),
                            (double *)outdata, 1);
            }
            else {
                int maxind, oind, i, a1s, outs;
                char *ptr, *optr;
                //double val;
                npy_intp niter, incptr, incoptr;

                maxind = (PyMicArray_DIM(ap1, 0) >= PyMicArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyMicArray_DATA(ap1);
                optr = PyMicArray_DATA(out_buf);
                l = PyMicArray_DIM(ap1, maxind);
                //val = *((double *)PyMicArray_DATA(ap2));
                a1s = PyMicArray_STRIDE(ap1, maxind) / sizeof(double);
                outs = PyMicArray_STRIDE(out_buf, maxind) / sizeof(double);

                niter = PyMicArray_DIM(ap1, oind);
                incptr = PyMicArray_STRIDE(ap1, oind);
                incoptr = PyMicArray_STRIDE(out_buf, oind);

                #pragma omp target device(device) \
                                     map(to: l, ptr, a1s, optr, outs, ap2data,\
                                             niter, incptr, incoptr)
                for (i = 0; i < niter; i++) {
                    cblas_daxpy(l, *((double *)ap2data), (double *)ptr, a1s,
                                (double *)optr, outs);
                    ptr += incptr;
                    optr += incoptr;
                }
            }
        }
        else if (typenum == NPY_CDOUBLE) {
            if (l == 1) {
                npy_cdouble *ptr1, *ptr2, *res;

                #pragma omp target device(device) map(to: outdata, ap1data, ap2data)
                {
                    ptr1 = (npy_cdouble *)ap2data;
                    ptr2 = (npy_cdouble *)ap1data;
                    res = (npy_cdouble *)outdata;
                    res->real = ptr1->real * ptr2->real - ptr1->imag * ptr2->imag;
                    res->imag = ptr1->real * ptr2->imag + ptr1->imag * ptr2->real;
                }
            }
            else if (ap1shape != _matrix) {
                #pragma omp target device(device) map(to: l, ap1stride, \
                                                    outdata, ap1data, ap2data)
                cblas_zaxpy(l,
                            (double *)ap2data,
                            (double *)ap1data,
                            ap1stride/sizeof(npy_cdouble),
                            (double *)outdata, 1);
            }
            else {
                int maxind, oind, i, a1s, outs;
                char *ptr, *optr;
                double *pval;
                npy_intp niter, incptr, incoptr;

                maxind = (PyMicArray_DIM(ap1, 0) >= PyMicArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyMicArray_DATA(ap1);
                optr = PyMicArray_DATA(out_buf);
                l = PyMicArray_DIM(ap1, maxind);
                pval = (double *)PyMicArray_DATA(ap2);
                a1s = PyMicArray_STRIDE(ap1, maxind) / sizeof(npy_cdouble);
                outs = PyMicArray_STRIDE(out_buf, maxind) / sizeof(npy_cdouble);

                niter = PyMicArray_DIM(ap1, oind);
                incptr = PyMicArray_STRIDE(ap1, oind);
                incoptr = PyMicArray_STRIDE(out_buf, oind);

                #pragma omp target device(device) map(to: l, pval, ptr, a1s, \
                                                      optr, outs, \
                                                      niter, incptr, incoptr)
                for (i = 0; i < niter; i++) {
                    cblas_zaxpy(l, pval, (double *)ptr, a1s,
                                (double *)optr, outs);
                    ptr += incptr;
                    optr += incoptr;
                }
            }
        }
        else if (typenum == NPY_FLOAT) {
            if (l == 1) {
                #pragma omp target device(device) map(to: outdata, ap1data, ap2data)
                *((float *)outdata) = *((float *)ap2data) * *((float *)ap1data);
            }
            else if (ap1shape != _matrix) {
                #pragma omp target device(device) map(to: l, ap1stride, \
                                                      outdata, ap1data, ap2data)
                cblas_saxpy(l,
                            *((float *)ap2data),
                            (float *)ap1data,
                            ap1stride/sizeof(float),
                            (float *)outdata, 1);
            }
            else {
                int maxind, oind, i, a1s, outs;
                char *ptr, *optr;
                //float val;
                npy_intp niter, incptr, incoptr;

                maxind = (PyMicArray_DIM(ap1, 0) >= PyMicArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyMicArray_DATA(ap1);
                optr = PyMicArray_DATA(out_buf);
                l = PyMicArray_DIM(ap1, maxind);
                //val = *((float *)PyMicArray_DATA(ap2));
                a1s = PyMicArray_STRIDE(ap1, maxind) / sizeof(float);
                outs = PyMicArray_STRIDE(out_buf, maxind) / sizeof(float);

                niter = PyMicArray_DIM(ap1, oind);
                incptr = PyMicArray_STRIDE(ap1, oind);
                incoptr = PyMicArray_STRIDE(out_buf, oind);

                #pragma omp target device(device) map(to: l, ptr, a1s, \
                                                      optr, outs, ap2data, \
                                                      niter, incptr, incoptr)
                for (i = 0; i < niter; i++) {
                    cblas_saxpy(l, *((float *)ap2data), (float *)ptr, a1s,
                                (float *)optr, outs);
                    ptr += incptr;
                    optr += incoptr;
                }
            }
        }
        else if (typenum == NPY_CFLOAT) {
            if (l == 1) {
                npy_cfloat *ptr1, *ptr2, *res;
                
                #pragma omp target device(device) map(to: outdata, ap1data, ap2data)
                {
                    ptr1 = (npy_cfloat *)PyMicArray_DATA(ap2);
                    ptr2 = (npy_cfloat *)PyMicArray_DATA(ap1);
                    res = (npy_cfloat *)PyMicArray_DATA(out_buf);
                    res->real = ptr1->real * ptr2->real - ptr1->imag * ptr2->imag;
                    res->imag = ptr1->real * ptr2->imag + ptr1->imag * ptr2->real;
                }
            }
            else if (ap1shape != _matrix) {
                #pragma omp target device(device) map(to: l, ap1stride, \
                                                      outdata, ap1data, ap2data)
                cblas_caxpy(l,
                            (float *)ap2data,
                            (float *)ap1data,
                            ap1stride/sizeof(npy_cfloat),
                            (float *)outdata, 1);
            }
            else {
                int maxind, oind, i, a1s, outs;
                char *ptr, *optr;
                float *pval;
                npy_intp niter, incptr, incoptr;

                maxind = (PyMicArray_DIM(ap1, 0) >= PyMicArray_DIM(ap1, 1) ? 0 : 1);
                oind = 1 - maxind;
                ptr = PyMicArray_DATA(ap1);
                optr = PyMicArray_DATA(out_buf);
                l = PyMicArray_DIM(ap1, maxind);
                pval = (float *)PyMicArray_DATA(ap2);
                a1s = PyMicArray_STRIDE(ap1, maxind) / sizeof(npy_cfloat);
                outs = PyMicArray_STRIDE(out_buf, maxind) / sizeof(npy_cfloat);

                niter = PyMicArray_DIM(ap1, oind);
                incptr = PyMicArray_STRIDE(ap1, oind);
                incoptr = PyMicArray_STRIDE(out_buf, oind);

                #pragma omp target device(device) map(to: l, ptr, a1s, \
                                                      optr, outs, pval, \
                                                      niter, incptr, incoptr)
                for (i = 0; i < niter; i++) {
                    cblas_caxpy(l, pval, (float *)ptr, a1s,
                                (float *)optr, outs);
                    ptr += incptr;
                    optr += incoptr;
                }
            }
        }
        /*End offload section */
        NPY_END_ALLOW_THREADS;
    }
    else if ((ap2shape == _column) && (ap1shape != _matrix)) {
        NPY_BEGIN_ALLOW_THREADS;

        /* Dot product between two vectors -- Level 1 BLAS */
        blas_dot(device, typenum, l,
                 PyMicArray_DATA(ap1), PyMicArray_STRIDE(ap1, (ap1shape == _row)),
                 PyMicArray_DATA(ap2), PyMicArray_STRIDE(ap2, 0),
                 PyMicArray_DATA(out_buf));
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape == _matrix && ap2shape != _matrix) {
        /* Matrix vector multiplication -- Level 2 BLAS */
        /* lda must be MAX(M,1) */
        enum CBLAS_ORDER Order;
        int ap2s;

        if (!PyMicArray_ISONESEGMENT(ap1)) {
            PyObject *new;
            new = PyMicArray_Copy(ap1);
            Py_DECREF(ap1);
            ap1 = (PyMicArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyMicArray_ISCONTIGUOUS(ap1)) {
            Order = CblasRowMajor;
            lda = (PyMicArray_DIM(ap1, 1) > 1 ? PyMicArray_DIM(ap1, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyMicArray_DIM(ap1, 0) > 1 ? PyMicArray_DIM(ap1, 0) : 1);
        }
        ap2s = PyMicArray_STRIDE(ap2, 0) / PyMicArray_ITEMSIZE(ap2);
        gemv(typenum, Order, CblasNoTrans, ap1, lda, ap2, ap2s, out_buf);
        NPY_END_ALLOW_THREADS;
    }
    else if (ap1shape != _matrix && ap2shape == _matrix) {
        /* Vector matrix multiplication -- Level 2 BLAS */
        enum CBLAS_ORDER Order;
        int ap1s;

        if (!PyMicArray_ISONESEGMENT(ap2)) {
            PyObject *new;
            new = PyMicArray_Copy(ap2);
            Py_DECREF(ap2);
            ap2 = (PyMicArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        NPY_BEGIN_ALLOW_THREADS
        if (PyMicArray_ISCONTIGUOUS(ap2)) {
            Order = CblasRowMajor;
            lda = (PyMicArray_DIM(ap2, 1) > 1 ? PyMicArray_DIM(ap2, 1) : 1);
        }
        else {
            Order = CblasColMajor;
            lda = (PyMicArray_DIM(ap2, 0) > 1 ? PyMicArray_DIM(ap2, 0) : 1);
        }
        if (ap1shape == _row) {
            ap1s = PyMicArray_STRIDE(ap1, 1) / PyMicArray_ITEMSIZE(ap1);
        }
        else {
            ap1s = PyMicArray_STRIDE(ap1, 0) / PyMicArray_ITEMSIZE(ap1);
        }
        gemv(typenum, Order, CblasTrans, ap2, lda, ap1, ap1s, out_buf);
        NPY_END_ALLOW_THREADS;
    }
    else {
        /*
         * (PyArray_NDIM(ap1) == 2 && PyArray_NDIM(ap2) == 2)
         * Matrix matrix multiplication -- Level 3 BLAS
         *  L x M  multiplied by M x N
         */
        enum CBLAS_ORDER Order;
        enum CBLAS_TRANSPOSE Trans1, Trans2;
        int M, N, L;

        /* Optimization possible: */
        /*
         * We may be able to handle single-segment arrays here
         * using appropriate values of Order, Trans1, and Trans2.
         */
        if (!PyMicArray_IS_C_CONTIGUOUS(ap2) && !PyMicArray_IS_F_CONTIGUOUS(ap2)) {
            PyObject *new = PyMicArray_Copy(ap2);

            Py_DECREF(ap2);
            ap2 = (PyMicArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }
        if (!PyMicArray_IS_C_CONTIGUOUS(ap1) && !PyMicArray_IS_F_CONTIGUOUS(ap1)) {
            PyObject *new = PyMicArray_Copy(ap1);

            Py_DECREF(ap1);
            ap1 = (PyMicArrayObject *)new;
            if (new == NULL) {
                goto fail;
            }
        }

        NPY_BEGIN_ALLOW_THREADS;

        Order = CblasRowMajor;
        Trans1 = CblasNoTrans;
        Trans2 = CblasNoTrans;
        L = PyMicArray_DIM(ap1, 0);
        N = PyMicArray_DIM(ap2, 1);
        M = PyMicArray_DIM(ap2, 0);
        lda = (PyMicArray_DIM(ap1, 1) > 1 ? PyMicArray_DIM(ap1, 1) : 1);
        ldb = (PyMicArray_DIM(ap2, 1) > 1 ? PyMicArray_DIM(ap2, 1) : 1);

        /*
         * Avoid temporary copies for arrays in Fortran order
         */
        if (PyMicArray_IS_F_CONTIGUOUS(ap1)) {
            Trans1 = CblasTrans;
            lda = (PyMicArray_DIM(ap1, 0) > 1 ? PyMicArray_DIM(ap1, 0) : 1);
        }
        if (PyMicArray_IS_F_CONTIGUOUS(ap2)) {
            Trans2 = CblasTrans;
            ldb = (PyMicArray_DIM(ap2, 0) > 1 ? PyMicArray_DIM(ap2, 0) : 1);
        }

        /*
         * Use syrk if we have a case of a matrix times its transpose.
         * Otherwise, use gemm for all other cases.
         */
        if (
            (PyMicArray_BYTES(ap1) == PyMicArray_BYTES(ap2)) &&
            (PyMicArray_DIM(ap1, 0) == PyMicArray_DIM(ap2, 1)) &&
            (PyMicArray_DIM(ap1, 1) == PyMicArray_DIM(ap2, 0)) &&
            (PyMicArray_STRIDE(ap1, 0) == PyMicArray_STRIDE(ap2, 1)) &&
            (PyMicArray_STRIDE(ap1, 1) == PyMicArray_STRIDE(ap2, 0)) &&
            ((Trans1 == CblasTrans) ^ (Trans2 == CblasTrans)) &&
            ((Trans1 == CblasNoTrans) ^ (Trans2 == CblasNoTrans))
        ) {
            if (Trans1 == CblasNoTrans) {
                syrk(typenum, Order, Trans1, N, M, ap1, lda, out_buf);
            }
            else {
                syrk(typenum, Order, Trans1, N, M, ap2, ldb, out_buf);
            }
        }
        else {
            gemm(typenum, Order, Trans1, Trans2, L, N, M, ap1, lda, ap2, ldb,
                 out_buf);
        }
        NPY_END_ALLOW_THREADS;
    }


    Py_DECREF(ap1);
    Py_DECREF(ap2);

    /* Trigger possible copyback into `result` */
    Py_DECREF(out_buf);

    return PyMicArray_Return(result);

fail:
    Py_XDECREF(ap1);
    Py_XDECREF(ap2);
    Py_XDECREF(out_buf);
    Py_XDECREF(result);
    return NULL;
}
