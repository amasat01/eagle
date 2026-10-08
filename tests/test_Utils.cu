// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#include "eagle/util/ObservableArray.h"
#include "eagle/util/Slice.h"

#include "TestBase.h"

namespace eagle_tests {
namespace DStatesTest {
using namespace eagle;

/** @brief Test The observer */
using ArrT = eagle::util::observable::Array<double>;

/** @brief Test observer and reference holder */
class ObsTest : public Test {
public:
    ArrT arr;
    ArrT b;
    ArrT::ObserverT obs;

    /** @brief Construct by allocating a */
    ObsTest()
        : arr{ TEST_SIZE }
        , b{ 0, 0 }
    {
        ArrT::GRef aref = arr.hostRef();
        for (idx_t i = 0; i < TEST_SIZE; i++)
            aref(i) = randReal();
    }

    void move() { b = std::move(arr); }
    void test()
    {
        ArrT::ObserverT obs(arr);
        move();
        ArrT::GRef bref = b.hostRef();
        ArrT::GRef oref = obs->hostRef();
        for (idx_t i = 0; i < TEST_SIZE; i++)
            ASSERT_EQ(oref(i), bref(i)) << "i = " << i;
    }
};

TEST_F(ObsTest, ObservationTest)
{
    test();
}

using SliceT = util::Slice<ArrT>;

/** @brief Test the slice  */
class SliceTest : public Test {
public:
    aether::Array<bool> flags;
    ArrT arr;
    SliceT slice;

    SliceTest()
        : flags{ eagle::makeArray<bool>(TEST_SIZE) }
        , arr{ TEST_SIZE }
        , slice{ nullptr }
    {
        /* fill the array */
        ArrT::GRef aref                      = arr.hostRef();
        const eagle::GRefArrT<bool> fref     = flags.hostView();
        idx_t count                          = 0;
        for (idx_t i = 0; i < TEST_SIZE; i++) {
            aref(i) = randReal();
            fref(i) = randBool();
            if (fref(i))
                count++;
        }

        /* Finalize the slice creation */
        auto idxs = eagle::makeArray<idx_t>(
            idx_t(aref.samples()), idx_t(aref.samples()));
        const eagle::GRefArrT<idx_t> iref = idxs.hostView();
        idx_t j                               = 0;
        for (idx_t i = 0; i < TEST_SIZE; i++) {
            if (fref(i))
                iref(j++) = i;
        }

        slice = std::move(SliceT(arr, std::move(idxs), count));
    }

    /** @brief validate */
    void validate()
    {
        ArrT::GRef aref                      = arr.hostRef();
        const eagle::GRefArrT<bool> fref     = flags.hostView();
        for (idx_t i = 0; i < TEST_SIZE; i++) {
            if (fref(i)) {
                ASSERT_EQ(aref(i), 0.0) << "i = " << i;
            }
        }
    }
};

/** @brief kernel for device slice testing */
AETHER_KERNEL() void sliceTest(SliceT::GRef slice)
{
    const eagle::SampleIndex i
        = SampleIndex::make(threadIdx.x, blockIdx.x, blockDim.x);

    if (i.global() < slice.size())
        slice[i] = 0;
}

TEST_F(SliceTest, HostTest)
{
    SliceT::GRef sref = slice.hostRef();
    for (idx_t i = 0; i < sref.size(); i++) {
        eagle::SampleIndex idx = eagle::SampleIndex::make(i);
        sref[idx]             = 0.0;
    }

    validate();
}

TEST_F(SliceTest, DeviceTest)
{
    slice.upload();
    sliceTest<<<TEST_N_BLOCKS, EAGLE_BLOCKSIZE>>>(slice.deviceRef());
    slice.download();
    EAGLE_CHECK_ALWAYS(cudaDeviceSynchronize());
    validate();
}

} // namespace DStatesTest
} // namespace eagle_tests
