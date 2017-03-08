#ifndef _MPY_PRIVATE_CONVERSION_UTILS_H_
#define _MPY_PRIVATE_CONVERSION_UTILS_H_

int PyMicArray_GeneralConverter(PyObject *object, PyObject **address);
int PyMicArray_Converter(PyObject *object, PyObject **address);
int PyMicArray_DeviceConverter(PyObject *object, int *device);
int PyMicArray_OutputConverter(PyObject *object, PyMicArrayObject **address);
int PyMicArray_ConvertMultiAxis(PyObject *axis_in, int ndim, npy_bool *out_axis_flags);

#endif
