/* Random kit 1.3 */

/*
 * Copyright (c) 2003-2005, Jean-Sebastien Roy (js@jeannot.org)
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the
 * "Software"), to deal in the Software without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Software, and to
 * permit persons to whom the Software is furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 * OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

/* @(#) $Jeannot: randomkit.h,v 1.24 2005/07/21 22:14:09 js Exp $ */

/*
 * Typical use:
 *
 * {
 *  rk_state state;
 *  unsigned long seed = 1, random_value;
 *
 *  rk_seed(seed, &state); // Initialize the RNG
 *  ...
 *  random_value = rk_random(&state); // Generate random values in [0..RK_MAX]
 * }
 *
 * Instead of rk_seed, you can use rk_randomseed which will get a random seed
 * from /dev/urandom (or the clock, if /dev/urandom is unavailable):
 *
 * {
 *  rk_state state;
 *  unsigned long random_value;
 *
 *  rk_randomseed(&state); // Initialize the RNG with a random seed
 *  ...
 *  random_value = rk_random(&state); // Generate random values in [0..RK_MAX]
 * }
 */

/*
 * Useful macro:
 *  RK_DEV_RANDOM: the device used for random seeding.
 *                 defaults to "/dev/urandom"
 */

#ifndef _MPY_RANDOMKIT_
#define _MPY_RANDOMKIT_

#include <stddef.h>

#define RK_STATE_LEN 624

typedef struct rk_state_
{
    int num_device;
    void *rng_streams[NMAXDEVICES];
}
rk_state;

typedef enum {
    RK_NOERR = 0, /* no error */
    RK_ENODEV = 1, /* no RK_DEV_RANDOM device */
    RK_ERR_MAX = 2
} rk_error;

/* error strings */
extern char *rk_strerror[RK_ERR_MAX];

/* Maximum generated random value */
#define RK_MAX 0xFFFFFFFFUL

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Initialize the RNG state using the random seed.
 */
void rk_init(rk_state *state, int ndevice);

void rk_clean(rk_state *state);


/*
 * Initialize the RNG state using the given seed.
 */
void rk_seed(unsigned long seed, rk_state *state);

/*
 * Initialize the RNG state using a random seed.
 * Uses /dev/random or, when unavailable, the clock (see randomkit.c).
 * Returns RK_NOERR when no errors occurs.
 * Returns RK_ENODEV when the use of RK_DEV_RANDOM failed (for example because
 * there is no such device). In this case, the RNG was initialized using the
 * clock.
 */
rk_error rk_randomseed(rk_state *state);


/*
 * fill the buffer with randombytes from the random device
 * Returns RK_ENODEV if the device is unavailable, or RK_NOERR if it is
 * On Unix, if strong is defined, RK_DEV_RANDOM is used. If not, RK_DEV_URANDOM
 * is used instead. This parameter has no effect on Windows.
 * Warning: on most unixes RK_DEV_RANDOM will wait for enough entropy to answer
 * which can take a very long time on quiet systems.
 */
rk_error rk_devfill(void *buffer, size_t size, int strong);

#ifdef __cplusplus
}
#endif

#endif /* _RANDOMKIT_ */
