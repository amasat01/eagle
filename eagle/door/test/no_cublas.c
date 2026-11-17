/* SPDX-License-Identifier: Apache-2.0 */

/*
 * LD_PRELOAD shim for the missing-cuBLAS row of gemm_test: dlopen refuses any cuBLAS library and passes every other
 * request on, so the door sees a machine without cuBLAS while the CUDA runtime still loads.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>

void* dlopen(const char* name, int flags);
void* dlopen(const char* name, int flags) {
    static void* (*next)(const char*, int);
    if (next == NULL) *(void**)&next = dlsym(RTLD_NEXT, "dlopen");
    if (name != NULL && strstr(name, "libcublas") != NULL) return NULL;
    return next(name, flags);
}
