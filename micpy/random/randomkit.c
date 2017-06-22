/* MPY Random kit 1.0 */

/* static char const rcsid[] =
  "@(#) $Jeannot: randomkit.c,v 1.28 2005/07/21 22:14:09 js Exp $"; */
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <assert.h>
#include <mkl_vsl.h>

#ifdef _WIN32
/*
 * Windows
 * XXX: we have to use this ugly defined(__GNUC__) because it is not easy to
 * detect the compiler used in distutils itself
 */
#if (defined(__GNUC__) && defined(NPY_NEEDS_MINGW_TIME_WORKAROUND))

/*
 * FIXME: ideally, we should set this to the real version of MSVCRT. We need
 * something higher than 0x601 to enable _ftime64 and co
 */
#define __MSVCRT_VERSION__ 0x0700
#include <time.h>
#include <sys/timeb.h>

/*
 * mingw msvcr lib import wrongly export _ftime, which does not exist in the
 * actual msvc runtime for version >= 8; we make it an alias to _ftime64, which
 * is available in those versions of the runtime
 */
#define _FTIME(x) _ftime64((x))
#else
#include <time.h>
#include <sys/timeb.h>
#define _FTIME(x) _ftime((x))
#endif

#ifndef RK_NO_WINCRYPT
/* Windows crypto */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0400
#endif
#include <windows.h>
#include <wincrypt.h>
#endif

#else
/* Unix */
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#endif

/*
 * Do not move this include. randomkit.h must be included
 * after windows timeb.h is included.
 */
#include "randomkit.h"

#define BRNG VSL_BRNG_MT2203

#ifndef RK_DEV_URANDOM
#define RK_DEV_URANDOM "/dev/urandom"
#endif

#ifndef RK_DEV_RANDOM
#define RK_DEV_RANDOM "/dev/random"
#endif

char *rk_strerror[RK_ERR_MAX] =
{
    "no error",
    "random device unvavailable"
};

void
rk_init(rk_state *state, int ndevice)
{
    int i;
    VSLStreamStatePtr stream;

    state->num_device = ndevice;
    for (i = 0; i < ndevice; ++i) {
        state->rng_streams[i] = NULL;
    }

    //rk_randomseed(state);
}

void
rk_clean(rk_state *state)
{
    int i;
    VSLStreamStatePtr stream;

    for (i = 0; i < state->num_device; ++i) {
        stream = state->rng_streams[i];
        #pragma omp target device(i) map(to: stream)
        vslDeleteStream(&stream);
        state->rng_streams[i] = NULL;
    }
}

/* static functions */
static unsigned long rk_hash(unsigned long key);

void
rk_seed(unsigned long seed, rk_state *state)
{
    int pos, i;
    seed &= 0xffffffffUL;
    VSLStreamStatePtr stream;

    for (i = 0; i < state->num_device; ++i) {
        stream = state->rng_streams[i];
        #pragma omp target device(i) map(to:seed) map(tofrom: stream)
        {
            if (stream != NULL) {
                vslDeleteStream(&stream);
            }
            vslNewStream(&stream, BRNG, seed);
        }
        state->rng_streams[i] = stream;
    }
}

/* Thomas Wang 32 bits integer hash function */
static unsigned long
rk_hash(unsigned long key)
{
    key += ~(key << 15);
    key ^=  (key >> 10);
    key +=  (key << 3);
    key ^=  (key >> 6);
    key += ~(key << 11);
    key ^=  (key >> 16);
    return key;
}

rk_error
rk_randomseed(rk_state *state)
{
#ifndef _WIN32
    struct timeval tv;
#else
    struct _timeb  tv;
#endif
    int i;

    unsigned long buffer;

    if (rk_devfill(&buffer, sizeof(unsigned long), 0) == RK_NOERR) {
        /* ensures non-zero key */
        buffer |= 0x80000000UL;
        buffer &= 0xffffffffUL;
        rk_seed(buffer, state);
        return RK_NOERR;
    }

#ifndef _WIN32
    gettimeofday(&tv, NULL);
    rk_seed(rk_hash(getpid()) ^ rk_hash(tv.tv_sec) ^ rk_hash(tv.tv_usec)
            ^ rk_hash(clock()), state);
#else
    _FTIME(&tv);
    rk_seed(rk_hash(tv.time) ^ rk_hash(tv.millitm) ^ rk_hash(clock()), state);
#endif

    return RK_ENODEV;
}

/*void
rk_fill(void *buffer, size_t size, rk_state *state)
{

}
*/

rk_error
rk_devfill(void *buffer, size_t size, int strong)
{
#ifndef _WIN32
    FILE *rfile;
    int done;

    if (strong) {
        rfile = fopen(RK_DEV_RANDOM, "rb");
    }
    else {
        rfile = fopen(RK_DEV_URANDOM, "rb");
    }
    if (rfile == NULL) {
        return RK_ENODEV;
    }
    done = fread(buffer, size, 1, rfile);
    fclose(rfile);
    if (done) {
        return RK_NOERR;
    }
#else

#ifndef RK_NO_WINCRYPT
    HCRYPTPROV hCryptProv;
    BOOL done;

    if (!CryptAcquireContext(&hCryptProv, NULL, NULL, PROV_RSA_FULL,
            CRYPT_VERIFYCONTEXT) || !hCryptProv) {
        return RK_ENODEV;
    }
    done = CryptGenRandom(hCryptProv, size, (unsigned char *)buffer);
    CryptReleaseContext(hCryptProv, 0);
    if (done) {
        return RK_NOERR;
    }
#endif

#endif
    return RK_ENODEV;
}