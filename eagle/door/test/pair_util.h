// SPDX-License-Identifier: Apache-2.0

// What the two device pair tests (runner_pair_test.c, active_pair_test.c) share: the PASS/FAIL tally, the
// error-reason probe, the oracle-file reader and the digest parser.
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
