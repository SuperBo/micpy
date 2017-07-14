#ifndef _MPY_ARRAY_TEMP_AVOID_H_
#define _MPY_ARRAY_TEMP_AVOID_H_

NPY_NO_EXPORT int
can_elide_temp_unary(PyMicArrayObject * m1);

NPY_NO_EXPORT int
try_binary_elide(PyMicArrayObject * m1, PyObject * m2,
                 PyObject * (inplace_op)(PyMicArrayObject * m1, PyObject * m2),
                 PyObject ** res, int commutative);

#endif
