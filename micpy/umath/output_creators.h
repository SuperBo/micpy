#ifndef _MPY_MUFUNC_OUTPUT_CREATION_H
#define _MPY_MUFUNC_OUTPUT_CREATION_H

int PyMUFunc_GetCommonDevice(int nop, PyMicArrayObject **op);

PyMicArrayObject *
PyMUFunc_CreateArrayBroadcast(int nop, PyMicArrayObject **arrs, PyArray_Descr *dtype);
#endif
