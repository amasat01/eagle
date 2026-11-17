/* SPDX-License-Identifier: Apache-2.0 */

/* The oracle of the dynamic SEGMENTED group "dg" of the fixture libraries, shared by the host and the device door tests: SGN samples. */
#pragma once

#include <stdint.h>
#include <string.h>

#define SGN 200

static inline int64_t seg_done(const unsigned char* mask) {
    int64_t done = 0;
    for (int i = 0; i < SGN; ++i) done += mask[i] != 0;
    return done;
}

/* The dynamic group's arrangement and sort schedule as the door states them: a stable sort of the slots by (live, class of the slot at
 * the last launch) when none was made or when more live samples than 5% of the live ones were left stale, the branches over their runs,
 * then the stale live slots stepped by their class. Returns the launches that sorted in @p sorts (up to 16). */
static inline int64_t dg_model(double* k, const double* limit, unsigned char* mask, double* pos, int64_t max_steps, int64_t* sorts, int* n_sorts) {
    static int order[SGN], cls[SGN], stale[SGN];
    int64_t launches = 0, stale_now = 0;
    int sorted = 0, end[2] = { 0, 0 };
    *n_sorts = 0;
    for (int i = 0; i < SGN; ++i) order[i] = i, cls[i] = 0, stale[i] = 0;
    while (launches < max_steps && seg_done(mask) != SGN) {
        int64_t live = 0;
        for (int s = 0; s < SGN; ++s) live += !mask[order[s]];
        if (!sorted || (double)stale_now > 0.05 * (double)(live > 1 ? live : 1)) {
            int next[SGN], at = 0;
            for (int c = 0; c < 3; ++c) {
                for (int s = 0; s < SGN; ++s)
                    if ((mask[order[s]] ? 2 : cls[s]) == c) next[at++] = order[s];
                if (c < 2) end[c] = at;    /* the runs of the sort: slots [0, end[0]) are class 0, [end[0], end[1]) class 1 */
            }
            memcpy(order, next, sizeof(order));
            sorted = 1;
            if (*n_sorts < 16) sorts[(*n_sorts)++] = launches + 1;
        }
        for (int s = 0; s < SGN; ++s) {    /* the slots of the runs the sort just made (a kept sort's runs are the same slots) */
            const int j = s < end[0] ? 0 : s < end[1] ? 1 : -1;
            const int i = order[s];
            if (j < 0 || mask[i]) continue;
            const int c = k[i] >= 4.0;
            cls[s] = c;
            stale[s] = c != j;
            pos[i] = (double)s;
            if (c == j) {
                k[i] += 0.125 * (j + 1);
                if (k[i] >= limit[i]) mask[i] = 1;
            }
        }
        stale_now = 0;
        for (int s = 0; s < SGN; ++s)
            if (stale[s] && !mask[order[s]] && s < end[1]) {
                const int i = order[s];
                k[i] += 0.125 * ((k[i] >= 4.0) + 1);
                if (k[i] >= limit[i]) mask[i] = 1;
                ++stale_now;
            }
        ++launches;
    }
    return launches;
}

