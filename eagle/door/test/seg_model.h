/* SPDX-License-Identifier: Apache-2.0 */

/* The oracle of the dynamic SEGMENTED group "dg" of the fixture libraries, shared by the host and the device door tests: SGN samples. */
#pragma once

#include <stdint.h>
#include <string.h>

#define SGN 200

/* The start of sample i of the dynamic group: the multiples of 0.02 below 4 in a scattered order (37 is coprime to 200), so the samples that
 * cross the class edge k = 4 together sit apart in sample order and a re-sort visibly moves them. */
#define DG_K0(i) (0.02 * (((i) * 37) % SGN))

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


/* The same group as the DEVICE runs it. The host carries the arrangement between launches and sorts that; the device sorts the planes in
 * sample order again at every launch, by a key it keeps frozen: the class of each sample at the last refresh, taken from the class plane
 * (sample order) at the first launch and whenever more live samples than 5% of the live ones were left stale. Both leave the caller's planes in
 * sample order, so the two arrangements differ only in the slot a sample held: pos[i] is the slot of sample i in the last launch that ran it,
 * and with a frozen key it changes only at a refresh. Returns the launches run; @p refreshes holds the launches that refreshed the key. */
static inline int64_t dg_model_device(double* k, const double* limit, unsigned char* mask, double* pos, int64_t max_steps, int64_t* refreshes,
                                      int* n_refreshes) {
    int key[SGN], cls[SGN], stale[SGN], sorted = 0;
    int64_t launches = 0, stale_now = 0;
    *n_refreshes = 0;
    for (int i = 0; i < SGN; ++i) key[i] = cls[i] = stale[i] = 0;
    while (launches < max_steps && seg_done(mask) != SGN) {
        int64_t live = 0;
        for (int i = 0; i < SGN; ++i) live += !mask[i];
        if (!sorted || (double)stale_now > 0.05 * (double)(live > 1 ? live : 1)) {
            memcpy(key, cls, sizeof(key));
            sorted = 1;
            if (*n_refreshes < 16) refreshes[(*n_refreshes)++] = launches + 1;
        }
        int order[SGN], at = 0, end[2] = { 0, 0 };
        for (int c = 0; c < 3; ++c) {
            for (int i = 0; i < SGN; ++i)
                if ((mask[i] ? 2 : key[i]) == c) order[at++] = i;
            if (c < 2) end[c] = at;
        }
        for (int s = 0; s < end[1]; ++s) {    /* the live samples, by run */
            const int i = order[s], j = s < end[0] ? 0 : 1, c = k[i] >= 4.0;
            cls[i] = c;
            stale[i] = c != j;
            pos[i] = (double)s;
            if (c == j) {
                k[i] += 0.125 * (j + 1);
                if (k[i] >= limit[i]) mask[i] = 1;
            }
        }
        stale_now = 0;
        for (int s = 0; s < end[1]; ++s) {    /* the stale samples that are still live, stepped by their class */
            const int i = order[s];
            if (stale[i] && !mask[i]) {
                k[i] += 0.125 * ((k[i] >= 4.0) + 1);
                if (k[i] >= limit[i]) mask[i] = 1;
                ++stale_now;
            }
        }
        ++launches;
    }
    return launches;
}

/* The AUTO group "sga" / "dsga": n1 samples of class 1 (limit l1: two steps a launch) after SGN - n1 of class 0 (limit 2 + i % 4: one step a launch). */
static inline void auto_fill(double* k, double* limit, double* how, unsigned char* mask, int n1, int l1) {
    for (int i = 0; i < SGN; ++i) {
        k[i] = 0.0;
        how[i] = 0.0;
        mask[i] = 0;
        limit[i] = i < SGN - n1 ? (double)(2 + i % 4) : (double)l1;
    }
}

/* What a run of @p seg segmented steps, then fused ones to step @p total, leaves in how[i]: the units add 1 per segmented step and 1000 per fused step, and
 * sample i finishes at its limit (class 0) or half of it (class 1). */
static inline double auto_how(const double* limit, int i, int n1, int seg, int total) {
    const int steps = i < SGN - n1 ? (int)limit[i] : ((int)limit[i] + 1) / 2;
    const int ran = steps < total ? steps : total, in_seg = ran < seg ? ran : seg;
    return in_seg + 1000.0 * (ran - in_seg);
}
