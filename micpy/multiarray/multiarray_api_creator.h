#define PyMicArray_API _PyMicArray_CAPI

#if ((PY_MAJOR_VERSION < 3 && PY_VERSION_HEX >= 0x02070000 ) \
        || PY_VERSION_HEX >= 0x03010000)
#define PyMicArray_API_USE_CAPSULE
#endif

#define init_PyMicArray_API() \
    static void *PyMicArray_API[] = {\
        (void *) &PyMicArray_New,\
        (void *) &PyMicArray_NewFromDescr,\
        (void *) &PyMicArray_NewLikeArray,\
        (void *) &PyMicArray_Empty,\
        (void *) &PyMicArray_Zeros,\
        NULL\
    }
