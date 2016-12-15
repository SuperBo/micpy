#include <mathimf.h>
#include <pymic_kernel.h>
#include <mkl.h>

/* Data types, needs to match _data_type_map in _misc.py */
#define DTYPE_INT32     0
#define DTYPE_INT64     1
#define DTYPE_FLOAT32   2
#define DTYPE_FLOAT64   3
#define DTYPE_COMPLEX   4
#define DTYPE_UINT64    5

#define _max_(x, y) (x < y) ? (y) : (x)

#define _min_(x, y) (x < y) ? (x) : (y)
