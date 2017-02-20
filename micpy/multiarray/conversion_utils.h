#ifndef _MPY_PRIVATE_CONVERSION_UTILS_H_
#define _MPY_PRIVATE_CONVERSION_UTILS_H_

int PyMicArray_GeneralConverter(PyObject *object, PyObject **address);
int PyMicArray_Converter(PyObject *object, PyObject **address);
int PyMicArray_DeviceConverter(PyObject *object, int *device);

#endif
