#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "structmember.h"

#define NPY_NO_DEPRECATED_API NPY_API_VERSION
#define NO_IMPORT_ARRAY
#define PY_ARRAY_UNIQUE_SYMBOL MICPY_ARRAY_API
#include <numpy/ndarraytypes.h>
#include <numpy/arrayobject.h>
#include <numpy/npy_common.h>
//#include "npy_config.h"

#include "common.h"
#include "alloc.h"
#include <assert.h>

#define NBUCKETS 1024 /* number of buckets for data*/
#define NBUCKETS_DIM 16 /* number of buckets for dimensions/strides */
#define NCACHE 7 /* number of cache entries per bucket */
/* this structure fits neatly into a cacheline */
typedef struct {
    npy_uintp available; /* number of cached pointers */
    void * ptrs[NCACHE];
} cache_bucket;
static cache_bucket datacache[NMAXDEVICES*NBUCKETS];
static cache_bucket dimcache[NBUCKETS_DIM];

/*
 * very simplistic small memory block cache to avoid more expensive libc
 * allocations
 * base function for data cache with 1 byte buckets and dimension cache with
 * sizeof(npy_intp) byte buckets
 */
static NPY_INLINE void *
_mpy_alloc_cache(int dev, npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t, int))
{
    assert((dev >= 0 && dev < NDEVICES) &&
           ((esz == 1 && cache == datacache) ||
            (esz == sizeof(npy_intp) && cache == dimcache)));
    if (nelem < msz) {
        int i = dev*msz + nelem;
        if (cache[i].available > 0) {
            return cache[i].ptrs[--(cache[i].available)];
        }
    }
    return alloc(nelem * esz, dev);
}

static NPY_INLINE void *
_npy_alloc_cache(npy_uintp nelem, npy_uintp esz, npy_uint msz,
                 cache_bucket * cache, void * (*alloc)(size_t))
{
    assert((esz == 1 && cache == datacache) ||
           (esz == sizeof(npy_intp) && cache == dimcache));
    if (nelem < msz) {
        if (cache[nelem].available > 0) {
            return cache[nelem].ptrs[--(cache[nelem].available)];
        }
    }
    return alloc(nelem * esz);
}

/*
 * return pointer p to cache, nelem is number of elements of the cache bucket
 * size (1 or sizeof(npy_intp)) of the block pointed too
 */
static NPY_INLINE void
_mpy_free_cache(int dev, void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *, int))
{
    assert(dev >= 0 && dev < NDEVICES);
    if (p != NULL && nelem < msz) {
        int i = dev*msz + nelem;
        if (cache[i].available < NCACHE) {
            cache[i].ptrs[cache[i].available++] = p;
            return;
        }
    }
    dealloc(p, dev);
}
static NPY_INLINE void
_npy_free_cache(void * p, npy_uintp nelem, npy_uint msz,
                cache_bucket * cache, void (*dealloc)(void *))
{
    if (p != NULL && nelem < msz) {
        if (cache[nelem].available < NCACHE) {
            cache[nelem].ptrs[cache[nelem].available++] = p;
            return;
        }
    }
    dealloc(p);
}

/*
 * array data cache, sz is number of bytes to allocate
 */
NPY_NO_EXPORT void *
mpy_alloc_cache(npy_uintp sz, int device)
{
    return _mpy_alloc_cache(device, sz, 1, NBUCKETS, datacache, &PyDataMemMic_NEW);
}

/* zero initialized data, sz is number of bytes to allocate */
NPY_NO_EXPORT void *
mpy_alloc_cache_zero(npy_uintp sz, int device)
{
    void * p;
    if (sz < NBUCKETS) {
        p = _mpy_alloc_cache(device, sz, 1, NBUCKETS, datacache, &PyDataMemMic_NEW);
        if (p) {
            #pragma omp target device(device) map(to:p,sz)
            memset(p, 0, sz);
        }
        return p;
    }
    Py_BEGIN_ALLOW_THREADS
    p = PyDataMemMic_NEW_ZEROED(sz, 1, device);
    Py_END_ALLOW_THREADS
    return p;
}

NPY_NO_EXPORT void
mpy_free_cache(void * p, npy_uintp sz, int device)
{
    _mpy_free_cache(device, p, sz, NBUCKETS, datacache, &PyDataMemMic_FREE);
}

/*
 * dimension/stride cache, uses a different allocator and is always a multiple
 * of npy_intp
 */
NPY_NO_EXPORT void *
mpy_alloc_cache_dim(npy_uintp sz)
{
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }
    return _npy_alloc_cache(sz, sizeof(npy_intp), NBUCKETS_DIM, dimcache,
                            &PyArray_malloc);
}

NPY_NO_EXPORT void
mpy_free_cache_dim(void * p, npy_uintp sz)
{
    /* dims + strides */
    if (NPY_UNLIKELY(sz < 2)) {
        sz = 2;
    }
    _npy_free_cache(p, sz, NBUCKETS_DIM, dimcache,
                    &PyArray_free);
}

/*NUMPY_API
 * Allocates memory for array data.
 */
NPY_NO_EXPORT void *
PyDataMemMic_NEW(size_t size, int device)
{
    void *result;

    result = omp_target_alloc(size, device);

    return result;
}

/*NUMPY_API
 * Allocates zeroed memory for array data on given device.
 */
NPY_NO_EXPORT void *
PyDataMemMic_NEW_ZEROED(size_t size, size_t elsize, int device)
{
    void *result;

    #pragma omp target device(device) map(from:result)
    result = calloc(size, elsize);

    return result;
}

/*NUMPY_API
 * Free memory for array data on given device.
 */
NPY_NO_EXPORT void
PyDataMemMic_FREE(void *ptr, int device)
{
    omp_target_free(ptr, device);
}

/*NUMPY_API
 * Reallocate/resize memory for array data on given divice.
 */
NPY_NO_EXPORT void *
PyDataMemMic_RENEW(void *ptr, size_t size, int device)
{
    void *result;

    Py_BEGIN_ALLOW_THREADS
    #pragma omp target device(device) map(from:result)
    result = realloc(ptr, size);

    Py_END_ALLOW_THREADS

    return result;
}
