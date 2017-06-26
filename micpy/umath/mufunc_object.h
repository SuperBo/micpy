#ifndef _MUFUNCOBJECT_H_
#define _MUFUNCOBJECT_H_

#include <numpy/npy_math.h>
#include <numpy/npy_common.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef NPY_BEGIN_THREADS_NDITER
#ifdef NPY_ALLOW_THREADS
#define NPY_BEGIN_THREADS_NDITER(iter) \
        do { \
            if (!NpyIter_IterationNeedsAPI(iter)) { \
                NPY_BEGIN_THREADS_THRESHOLDED(NpyIter_GetIterSize(iter)); \
            } \
        } while(0)

#else
#define NPY_BEGIN_THREADS_NDITER(iter)
#endif
#endif

/*
 * Given the operands for calling a ufunc, should determine the
 * calculation input and output data types and return an inner loop function.
 * This function should validate that the casting rule is being followed,
 * and fail if it is not.
 *
 * For backwards compatibility, the regular type resolution function does not
 * support auxiliary data with object semantics. The type resolution call
 * which returns a masked generic function returns a standard NpyAuxData
 * object, for which the NPY_AUXDATA_FREE and NPY_AUXDATA_CLONE macros
 * work.
 *
 * ufunc:             The ufunc object.
 * casting:           The 'casting' parameter provided to the ufunc.
 * operands:          An array of length (ufunc->nin + ufunc->nout),
 *                    with the output parameters possibly NULL.
 * type_tup:          Either NULL, or the type_tup passed to the ufunc.
 * out_dtypes:        An array which should be populated with new
 *                    references to (ufunc->nin + ufunc->nout) new
 *                    dtypes, one for each input and output. These
 *                    dtypes should all be in native-endian format.
 *
 * Should return 0 on success, -1 on failure (with exception set),
 * or -2 if Py_NotImplemented should be returned.
 */
typedef int (PyMUFunc_TypeResolutionFunc)(
                                struct _tagPyUFuncObject *ufunc,
                                NPY_CASTING casting,
                                PyMicArrayObject **operands,
                                PyObject *type_tup,
                                PyArray_Descr **out_dtypes);

#define MUFUNC_PYVALS_NAME "MUFUNC_PYVALS"

#define MUFUNC_CHECK_ERROR(arg) \
        do {if ((((arg)->obj & UFUNC_OBJ_NEEDS_API) && PyErr_Occurred()) || \
            ((arg)->errormask && \
             PyMUFunc_checkfperr((arg)->errormask, \
                                (arg)->errobj, \
                                &(arg)->first))) \
                goto fail;} while (0)

#ifdef _MICARRAY_UMATHMODULE
extern PyTypeObject PyMUFunc_Type;

NPY_VISIBILITY_HIDDEN extern PyObject * mpy_um_str_out;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_um_str_subok;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_um_str_pyvals_name;
NPY_VISIBILITY_HIDDEN extern PyObject * mpy_um_str_array_wrap;

NPY_NO_EXPORT PyObject *
PyMUFunc_FromFuncAndData(PyUFuncGenericFunction *func, void **data,
                        char *types, int ntypes,
                        int nin, int nout, int identity,
                        const char *name, const char *doc, int unused);

NPY_NO_EXPORT PyObject *
PyMUFunc_FromFuncAndDataAndSignature(PyUFuncGenericFunction *func, void **data,
                                     char *types, int ntypes,
                                     int nin, int nout, int identity,
                                     const char *name, const char *doc,
                                     int unused, const char *signature);


#endif


#ifdef __cplusplus
}
#endif
#endif /* !Py_UFUNCOBJECT_H */
