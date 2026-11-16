// SPDX-License-Identifier: Apache-2.0

// What the device pair tests share: the PASS/FAIL tally, the error-reason probe, the oracle-file reader, the digest
// parser and the device-tensor wrapper.
#pragma once
#include "eagle/door/eg_door.h"
#include "plugin/dlpack.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int failures = 0;

static void check(int ok, const char* what) {
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok) ++failures;
}

static int reason_has(eg_ctx* ctx, const char* part) {
    char buf[1024];
    size_t len = 0;
    if (eg_error(ctx, buf, sizeof(buf), &len) != EG_OK) return 0;
    if (strstr(buf, part) != NULL) return 1;
    printf("      reason: %s\n", buf);
    return 0;
}

static int read_file(const char* dir, const char* file, void* out, size_t size) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s", dir, file);
    FILE* f = fopen(path, "rb");
    if (f == NULL) return 0;
    const size_t got = fread(out, 1, size, f);
    fclose(f);
    return got == size;
}

static int parse_digest(const char* hex, uint8_t* out) {
    if (strlen(hex) != 2 * EG_DIGEST_BYTES) return 0;
    for (int i = 0; i < EG_DIGEST_BYTES; ++i) {
        unsigned b = 0;
        if (sscanf(hex + 2 * i, "%2x", &b) != 1) return 0;
        out[i] = (uint8_t)b;
    }
    return 1;
}

struct tensor {
    struct DLManagedTensorVersioned managed;
    int64_t shape[1];
};

/* A one-dimensional device tensor over @p data: n elements of the DLPack type (code, bits). */
static inline void tensor_fill(struct tensor* t, int64_t n, void* data, uint8_t code, uint8_t bits) {
    memset(t, 0, sizeof(*t));
    t->shape[0] = n;
    t->managed.version.major = DLPACK_MAJOR_VERSION;
    t->managed.dl_tensor.data = data;
    t->managed.dl_tensor.device.device_type = kDLCUDA;
    t->managed.dl_tensor.ndim = 1;
    t->managed.dl_tensor.dtype.code = code;
    t->managed.dl_tensor.dtype.bits = bits;
    t->managed.dl_tensor.dtype.lanes = 1;
    t->managed.dl_tensor.shape = t->shape;
}

static inline eg_plane* wrap(eg_ctx* ctx, struct tensor* t, int64_t n, void* data, uint8_t code, uint8_t bits) {
    tensor_fill(t, n, data, code, bits);
    eg_plane* p = NULL;
    return eg_plane_wrap(ctx, &t->managed, &p) == EG_OK ? p : NULL;
}

/* A runner over one plan: per-step reverse, step events, a stop with no event refused. */
static inline struct eg_runner_desc runner_desc(eg_plan* const* plan, int64_t max_steps) {
    struct eg_runner_desc d;
    memset(&d, 0, sizeof(d));
    d.struct_size = sizeof(d);
    d.n_plans = 1;
    d.plans = plan;
    d.max_steps = max_steps;
    d.steps_reverse = EG_STEPS_REVERSE_PER_STEP;
    d.events = EG_EVENTS_STEP;
    d.on_no_event = EG_NO_EVENT_REFUSE;
    return d;
}
