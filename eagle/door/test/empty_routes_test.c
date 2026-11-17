/* SPDX-License-Identifier: Apache-2.0 */

/* A host-only door built with an EMPTY EG_GEMM_ROUTE_ROWS (test/empty_routes/eagle_door_rows.h): the host context is made, as a door whose
 * row header gives no gemm routes must make it. Usage: empty_routes_test */
#include "eagle/door/eg_door.h"

#include <stdio.h>
#include <string.h>

int main(void) {
    struct eg_ctx_settings s;
    memset(&s, 0, sizeof(s));
    s.struct_size = sizeof(s);
    s.abi_major = EG_DOOR_VERSION_MAJOR;
    eg_ctx* ctx = NULL;
    const int ok = eg_ctx_new(&s, &ctx) == EG_OK && ctx != NULL;
    printf("%s  a host context is made by a door whose row header has no gemm route rows\n", ok ? "PASS" : "FAIL");
    if (ctx != NULL) eg_ctx_free(ctx);
    return !ok;
}
