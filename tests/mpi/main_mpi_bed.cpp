// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

// `eagle_mpi_bed`'s entry point — the ONE thing this binary needs that
// GTest::gtest_main cannot give it: `MPI_Init` before any row runs, and a
// per-rank XML report.
//
// EVERY RANK RUNS THE WHOLE SUITE. That is the point of the bed: a row
// asserts what its own rank computed AND that the other rank agrees, so both ranks
// must reach every row or the collective inside it deadlocks. There is no
// "rank 0 tests, rank 1 idles" mode here.
//
// THE XML IS NAMED PER RANK, AND THE RENAME HAPPENS BEFORE `InitGoogleTest`.
// Two ranks share a working directory, so one `--gtest_output=xml:eagle_mpi_bed.xml`
// would have both ranks racing to write the same file — and the gate
// (tests/mpi/check_mpi_bed.sh) reads BOTH, comparing rank 1's verdict against
// rank 0's. The rewrite must precede `InitGoogleTest` because that call ends in
// `PostFlagParsingInit()`, which installs the XML listener from the output flag AS
// IT STANDS THEN: assigning `GTEST_FLAG(output)` afterwards is silently ignored
// (measured — both ranks wrote one un-suffixed file). So the rank is read first and
// the `--gtest_output=` argument is rewritten IN ARGV, which also keeps a
// caller-supplied path honoured rather than overridden.

#include <mpi.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wrestrict"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <string>

namespace {

// Insert `_rank<r>` before the extension of an `xml:<path>` / `json:<path>` gtest
// output spec. A spec with no extension (or a directory spec, which gtest spells
// with a trailing '/') gets the suffix appended to the stem it does have.
std::string rankOutputSpec(const std::string& spec, int rank)
{
    const std::string suffix = "_rank" + std::to_string(rank);
    const std::size_t colon = spec.find(':');
    if (colon == std::string::npos) return spec;   // bare "xml" — gtest's default name
    const std::string fmt  = spec.substr(0, colon + 1);
    const std::string path = spec.substr(colon + 1);
    if (path.empty() || path.back() == '/') return spec + "eagle_mpi_bed" + suffix;
    const std::size_t slash = path.find_last_of('/');
    const std::size_t dot   = path.find_last_of('.');
    if (dot == std::string::npos || (slash != std::string::npos && dot < slash))
        return fmt + path + suffix;
    return fmt + path.substr(0, dot) + suffix + path.substr(dot);
}

}  // namespace

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // Rewrite `--gtest_output=<spec>` in place. The replacement string is static so
    // the pointer parked in argv outlives InitGoogleTest's parse.
    static std::string rewritten;
    const std::string kFlag = "--gtest_output=";
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind(kFlag, 0) != 0) continue;
        rewritten = kFlag + rankOutputSpec(a.substr(kFlag.size()), rank);
        argv[i] = const_cast<char*>(rewritten.c_str());
        break;
    }

    ::testing::InitGoogleTest(&argc, argv);

    const int rc = RUN_ALL_TESTS();

    // A rank that failed must not leave its peers blocked in a collective that the
    // failing row never reached; `--gtest_break_on_failure` is deliberately NOT
    // used by the bed's gate for the same reason. Finalize normally and let the
    // launcher propagate the non-zero status.
    MPI_Finalize();
    return rc;
}
