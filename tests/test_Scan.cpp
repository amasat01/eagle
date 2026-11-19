// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "eagle/eagle.h"

#include "TestBase.h"

namespace eagle_tests {
namespace ScanTest {

using eagle::idx_t;
using eagle::filtering::Scanner;
using ArrT = eagle::util::observable::Array<idx_t>;
template<bool Inclusive>
using TheScannerT
    = Scanner<idx_t, aether::SumOp<idx_t>, Inclusive>;
using SliceT = eagle::util::Slice<ArrT>;

/** @brief Scan test suite */
template<bool Inclusive_, idx_t TSIZE_ = TEST_SIZE>
class ScanTest : public Test {
public:
    using Self                      = ScanTest;
    using ScannerT                  = TheScannerT<Inclusive_>;
    static constexpr idx_t TSIZE    = TSIZE_;
    static constexpr bool Inclusive = Inclusive_;

    /** @brief Clamp the fixture size under EAGLE_TEST_MINIMAL=1 so the large
     *  scan variants stay tractable under valgrind. */
    static idx_t effectiveSize()
    {
        return isMinimalMode()
            ? std::min<idx_t>(TSIZE_, static_cast<idx_t>(10'000))
            : TSIZE_;
    }

    /* Data members */
    ScannerT scanner_;
    ArrT flags_;
    ArrT validation_;
    idx_t scansize = 0;

    /** @brief Constructor */
    ScanTest()
        : scanner_{ effectiveSize() }
        , flags_{ effectiveSize(), 0 }
        , validation_{ effectiveSize() }
    {
        /* randomly fill arr with 0 and 1 */
        idx_t val                             = 0;
        ArrT::GRef aref                       = scanner_.hostRef().inputs();
        ArrT::GRef vref                       = validation_.hostRef();
        ArrT::GRef fref                       = flags_.hostRef();
        for (idx_t i = 0; i < scanner_.size(); i++) {
            fref(i) = randBool();
            aref(i) = fref(i);
            if constexpr (Inclusive) {
                if (fref(i))
                    val++;
                vref(i) = val;
            } else {
                vref(i) = val;
                if (fref(i))
                    val++;
            }
        }
        scansize = val;
    }

    /* Run the validation */
    void validate()
    {
        /* Validate the scan size */
        typename ScannerT::GRef sref = scanner_.hostRef();
        ASSERT_EQ(sref.numElements(), scansize) << "Wrong Scanned Size";
        /* validate the scan results */
        ArrT::GRef rref = sref.results();
        ArrT::GRef vref = validation_.hostRef();
        for (idx_t i = 0; i < scanner_.size(); i++) {
            ASSERT_EQ(vref(i), rref(i)) << "at index " << i;
        }
    }

    /** @brief Validate over the given slice */
    void validateSlice(const SliceT::GRef slice)
    {
        /* validate the edited slice */
        ASSERT_EQ(slice.size(), scansize) << "Wrong slice size" << std::endl;
        for (idx_t i = 0; i < slice.size(); i++) {
            eagle::SampleIndex idx = eagle::SampleIndex::make(i);
            ASSERT_EQ(slice[idx], 1)
                << "Not true slice flag at index i = " << i;
        }
    }
};

/* Test types */
using SmallInclusive = ScanTest<true>;
using LargeInclusive = ScanTest<true, 1000000>;
using SmallExclusive = ScanTest<false>;
using LargeExclusive = ScanTest<false, 1000000>;

TEST_F(SmallInclusive, Host)
{
    SliceT slice(flags_);
    scanner_.hostRun();
    eagle::util::slice::OMPscatter<SliceT, ScannerT>(
        slice.hostRef(), scanner_.hostRef());
    this->validate();
    this->validateSlice(slice.hostRef());
}

TEST_F(LargeInclusive, Host)
{
    SliceT slice(flags_);
    scanner_.hostRun();
    eagle::util::slice::OMPscatter<SliceT, ScannerT>(
        slice.hostRef(), scanner_.hostRef());
    this->validate();
    this->validateSlice(slice.hostRef());
}

TEST_F(SmallExclusive, Host)
{
    SliceT slice(flags_);
    scanner_.hostRun();
    eagle::util::slice::OMPscatter<SliceT, ScannerT>(
        slice.hostRef(), scanner_.hostRef());
    this->validate();
    this->validateSlice(slice.hostRef());
}

TEST_F(LargeExclusive, Host)
{
    SliceT slice(flags_);
    scanner_.hostRun();
    eagle::util::slice::OMPscatter<SliceT, ScannerT>(
        slice.hostRef(), scanner_.hostRef());
    this->validate();
    this->validateSlice(slice.hostRef());
}

/** @brief Boolean array type */
using BoolArrT = aether::Array<bool>;

/** @brief test the fully fledged slice with automatic scattering */
template<idx_t TSIZE_ = TEST_SIZE>
class ScatterSliceTest : public Test {
public:
    using Self                   = ScatterSliceTest;
    using SliceT                 = eagle::filtering::FilteringSlice<ArrT>;
    static constexpr idx_t TSIZE = TSIZE_;

    static idx_t effectiveSize()
    {
        return isMinimalMode()
            ? std::min<idx_t>(TSIZE_, static_cast<idx_t>(10'000))
            : TSIZE_;
    }

    /* Data members */
    ArrT data_;
    ArrT lflags_;
    ArrT rflags_;
    ArrT validation_;
    idx_t slicesize_ = 0;
    /** @brief Constructor */
    ScatterSliceTest()
        : data_{ effectiveSize() }
        , lflags_{ effectiveSize(), 0 }
        , rflags_{ effectiveSize(), 0 }
        , validation_{ effectiveSize() }
    {
        /* randomly fill left and right flags with 0 and 1 */
        ArrT::GRef vref                        = validation_.hostRef();
        ArrT::GRef lfref                       = lflags_.hostRef();
        ArrT::GRef rfref                       = rflags_.hostRef();
        for (idx_t i = 0; i < effectiveSize(); i++) {
            lfref(i) = randBool();
            rfref(i) = randBool();
            if (lfref(i) || rfref(i)) {
                vref(i) = 4;
                slicesize_++;
            }
        }
    }

    /** @brief Validate over the given slice */
    void validateSlice(const SliceT::GRef slice)
    {
        /* validate the edited slice */
        ASSERT_EQ(slice.size(), slicesize_) << "Wrong slice size" << std::endl;
        for (idx_t i = 0; i < slice.size(); i++) {
            eagle::SampleIndex idx = eagle::SampleIndex::make(i);
            ASSERT_EQ(slice[idx], 4) << "Wrong slice value at index i = " << i;
        }
    }
};


template<typename SliceT>
void OMPfeed(typename SliceT::GRef slice, ArrT::GRef left, ArrT::GRef right)
{
#pragma omp parallel for
    for (idx_t i = 0; i < idx_t(left.samples()); i++) {
        slice.scanner().inputs()(i) = left(i) || right(i);
    }
}

template<typename SliceT>
void OMPedit(typename SliceT::GRef slice)
{
#pragma omp parallel for
    for (idx_t i = 0; i < slice.size(); i++) {
        eagle::SampleIndex idx = eagle::SampleIndex::make(i);
        slice[idx]            = 4;
    }
}

/* Test runs */
using SmallScatterSlice = ScatterSliceTest<>;
using LargeScatterSlice = ScatterSliceTest<1000000>;

TEST_F(SmallScatterSlice, HostTest)
{
    SliceT slice(data_);
    SliceT::GRef sref = slice.hostRef();
    OMPfeed<SliceT>(slice.hostRef(), lflags_.hostRef(), rflags_.hostRef());
    slice.hostUpdate();
    OMPedit<SliceT>(slice.hostRef());
    validateSlice(sref);
}

TEST_F(LargeScatterSlice, HostTest)
{
    SliceT slice(data_);
    SliceT::GRef sref = slice.hostRef();
    OMPfeed<SliceT>(slice.hostRef(), lflags_.hostRef(), rflags_.hostRef());
    slice.hostUpdate();
    OMPedit<SliceT>(slice.hostRef());
    validateSlice(sref);
}

/** @brief Run ``Slice::hostUpdate()`` from inside an existing ``omp parallel``
 *  region.  Exercises the context-aware sliceFeed → scan → scatter chain:
 *  every thread enters ``hostUpdate`` together; the helpers must
 *  work-share via ``packetFlatFor``'s nested branch instead of spawning
 *  nested parallel regions.  Output (slice size + indexes) must match
 *  the non-nested call. */
template<typename SliceT>
void runHostUpdateInsideParallel(SliceT& slice, int numThreads)
{
#pragma omp parallel num_threads(numThreads)
    {
        slice.hostUpdate();
    }
}

/** @brief Reference run: ``hostUpdate`` outside any parallel region. */
template<typename SliceT>
void runHostUpdateOutsideParallel(SliceT& slice)
{
    slice.hostUpdate();
}

template<typename FixtureT>
void verifyNestedHostUpdateMatches(FixtureT& fixture, int numThreads)
{
    using SliceTT = typename FixtureT::SliceT;

    /* Reference: outside-parallel hostUpdate */
    SliceTT refSlice(fixture.data_);
    OMPfeed<SliceTT>(
        refSlice.hostRef(), fixture.lflags_.hostRef(), fixture.rflags_.hostRef());
    runHostUpdateOutsideParallel(refSlice);

    /* Subject: inside-parallel hostUpdate at the requested thread count */
    SliceTT subSlice(fixture.data_);
    OMPfeed<SliceTT>(
        subSlice.hostRef(), fixture.lflags_.hostRef(), fixture.rflags_.hostRef());
    runHostUpdateInsideParallel(subSlice, numThreads);

    /* Compare slice size */
    typename SliceTT::GRef rref = refSlice.hostRef();
    typename SliceTT::GRef sref = subSlice.hostRef();
    ASSERT_EQ(rref.size(), sref.size())
        << "slice size mismatch (numThreads=" << numThreads << ")";

    /* Compare indexes byte-by-byte */
    for (idx_t i = 0; i < rref.size(); i++) {
        eagle::SampleIndex idx = eagle::SampleIndex::make(i);
        ASSERT_EQ(rref.rawIndexes().eval(idx), sref.rawIndexes().eval(idx))
            << "index mismatch at i=" << i << " (numThreads=" << numThreads
            << ")";
    }
}

/** @brief A host slice update gives the same indexes for every team of the host pool (the scan inside is team-invariant). */
TEST_F(LargeScatterSlice, HostUpdateGivesTheSameIndexesForEveryTeam)
{
    std::vector<eagle::idx_t> one;
    for (int team : { 1, 2, 7 }) {
        eagle::exec::host_threads_setting().store(team);
        SliceT slice(data_);
        OMPfeed<SliceT>(slice.hostRef(), lflags_.hostRef(), rflags_.hostRef());
        slice.hostUpdate();
        const SliceT::GRef ref = slice.hostRef();
        std::vector<eagle::idx_t> got;
        for (idx_t i = 0; i < ref.size(); i++)
            got.push_back(ref.rawIndexes().eval(eagle::SampleIndex::make(i)));
        if (team == 1)
            one = got;
        EXPECT_EQ(got, one) << "team=" << team;
    }
    eagle::exec::host_threads_setting().store(0);
}

TEST_F(SmallScatterSlice, NestedHostUpdate_T1) { verifyNestedHostUpdateMatches(*this, 1); }
TEST_F(SmallScatterSlice, NestedHostUpdate_T2) { verifyNestedHostUpdateMatches(*this, 2); }
TEST_F(SmallScatterSlice, NestedHostUpdate_T4) { verifyNestedHostUpdateMatches(*this, 4); }
TEST_F(SmallScatterSlice, NestedHostUpdate_T8) { verifyNestedHostUpdateMatches(*this, 8); }

TEST_F(LargeScatterSlice, NestedHostUpdate_T1) { verifyNestedHostUpdateMatches(*this, 1); }
TEST_F(LargeScatterSlice, NestedHostUpdate_T2) { verifyNestedHostUpdateMatches(*this, 2); }
TEST_F(LargeScatterSlice, NestedHostUpdate_T4) { verifyNestedHostUpdateMatches(*this, 4); }
TEST_F(LargeScatterSlice, NestedHostUpdate_T8) { verifyNestedHostUpdateMatches(*this, 8); }

/** @brief The host scan runs its blocks on the host pool: for every team the results equal the one-thread run and the
 *  serial prefix, exactly (integer sums). */
template<bool Inclusive>
void expectScanTeamInvariant()
{
    for (idx_t n : { idx_t(1), idx_t(2), idx_t(255), idx_t(256), idx_t(257), idx_t(1000003) }) {
        const auto size = static_cast<std::size_t>(n);
        std::vector<idx_t> in(size), want(size);
        idx_t acc = 0;
        for (idx_t i = 0; i < n; ++i) {
            in[std::size_t(i)] = (i * 7919) % 13;
            want[std::size_t(i)] = Inclusive ? acc + in[std::size_t(i)] : acc;
            acc += in[std::size_t(i)];
        }
        const idx_t blocks = (n + 1023) / 1024;
        std::vector<idx_t> one;
        for (int team : { 1, 2, 7 }) {
            std::vector<idx_t> out(std::size_t(n), -1), sums(std::size_t(blocks), -1);
            eagle::cpu::Scan::scan<idx_t, aether::SumOp<idx_t>, Inclusive>(
                eagle::spanView<idx_t>(in.data(), n, eagle::hostDevice()).as_const(),
                eagle::spanView<idx_t>(out.data(), n, eagle::hostDevice()),
                eagle::spanView<idx_t>(sums.data(), blocks, eagle::hostDevice()), idx_t(0), team);
            EXPECT_EQ(out, want) << "n=" << n << " team=" << team;
            if (team == 1)
                one = out;
            EXPECT_EQ(out, one) << "n=" << n << " team=" << team;
        }
    }
}

TEST(HostScan, InclusiveGivesTheSameResultForEveryTeam) { expectScanTeamInvariant<true>(); }
TEST(HostScan, ExclusiveGivesTheSameResultForEveryTeam) { expectScanTeamInvariant<false>(); }

} // namespace ScanTest
} // namespace eagle_tests