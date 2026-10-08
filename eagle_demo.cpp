// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// EAGLE C++-mode smoke demo: dispatch a small write loop through the
// OpenMP+SIMD host launcher and verify the result.
#include <cstdio>
#include <vector>

#include <eagle/eagle.h>

int main()
{
    constexpr eagle::idx_t N = 64;
    std::vector<int> buf(N, 0);

    eagle::cpu::Host::launch(N, [&](const auto& i) {
        buf[i.global()] = static_cast<int>(i.global());
    });

    for (eagle::idx_t i = 0; i < N; ++i)
        if (buf[i] != static_cast<int>(i)) {
            std::printf("eagle_demo (C++): FAIL at %ld\n", static_cast<long>(i));
            return 1;
        }

    std::printf("eagle_demo (C++): OK\n");
    return 0;
}
