// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0
//
// Sanitizer non-vacuity canary (host). Each mode commits ONE defect that a
// -fsanitize=address,undefined build must catch; the CI step asserts the run
// exits non-zero AND prints the sanitizer banner. A sanitizer job whose canary
// stays quiet is not sanitizing anything.
//
//   canary heap-oob    read one past a heap array        -> AddressSanitizer: heap-buffer-overflow
//   canary leak        drop the only pointer to a block  -> LeakSanitizer: detected memory leaks
//   canary uninit      load a bool from fresh heap memory-> runtime error: load of value ... not a valid value for type 'bool'
//   canary shift-ub    shift by >= the type width        -> runtime error: shift exponent ... is too large
//
// `uninit` is caught through ASan's malloc fill pattern (0xbe) read as a bool;
// real uninitialised-read coverage is valgrind --track-origins (tests/sanitize/).
//
// Needs ASAN_OPTIONS=detect_leaks=1 for `leak` and UBSAN_OPTIONS=halt_on_error=1
// for a non-zero exit from the two UBSan modes.

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace {

void* volatile g_sink;

int heapOob() {
    int* p = static_cast<int*>(std::malloc(4 * sizeof(int)));
    std::memset(p, 0, 4 * sizeof(int));
    g_sink = p;
    volatile int v = static_cast<volatile int*>(g_sink)[4]; // one past the end
    std::free(p);
    return v;
}

int leak() {
    g_sink = std::malloc(64);
    g_sink = nullptr; // the only reference is gone
    return 0;
}

int uninit() {
    bool* p = static_cast<bool*>(std::malloc(16));
    g_sink = p;
    volatile bool b = static_cast<volatile bool*>(g_sink)[0]; // never written
    int r = b ? 1 : 0;
    std::free(p);
    return r;
}

int shiftUb() {
    volatile int n = 40;
    volatile int one = 1;
    return one << n;
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s heap-oob|leak|uninit|shift-ub\n", argv[0]);
        return 64;
    }
    const char* m = argv[1];
    int r = 0;
    if      (!std::strcmp(m, "heap-oob")) r = heapOob();
    else if (!std::strcmp(m, "leak"))     r = leak();
    else if (!std::strcmp(m, "uninit"))   r = uninit();
    else if (!std::strcmp(m, "shift-ub")) r = shiftUb();
    else { std::fprintf(stderr, "unknown mode '%s'\n", m); return 64; }
    std::printf("canary %s: ran to completion (r=%d)\n", m, r);
    return 0;
}
