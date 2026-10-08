// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/reduce/detail.h"
#include "eagle/typedefs.h"
#include "eagle/cuda.h"
#include "eagle/util/DeviceError.h"
#include "eagle/util/log.h"
#include "eagle/util/throw.h"

#include <type_traits>

namespace eagle {
namespace cuda {

#ifndef EAGLE_CPU_ONLY

/**
 * @brief GPU parallel reduction over aether scalar arrays.
 *
 * All overloads of ``reduceBlocking`` synchronise the CUDA stream before
 * returning the result on the host.  For graph-based (non-blocking) variants,
 * use the overloads that accept a ``cuda::Graph&``.
 *
 * @note Available only when ``EAGLE_CPU_ONLY`` is not defined.
 */
struct Reduction {

    /**
     * @brief Generate a work buffer for use with `reduceBlocking`, of the
     * appropriate size to perform a reduction over `arr`.
     *
     * Note: the returned array is only allocated on the host. You will need to
     * call `oarr.upload` before invoking `reduceBlocking`.
     */
    template<typename T>
    static inline aether::Array<T> makeBuffer(const idx_t& size)
    {
        const idx_t nBlocks = reduce::detail::Core::countBlocks<>(size);
        const idx_t bufSize = (nBlocks > 1) ? nBlocks : size;
        return makeArray<T>(bufSize);
    }

    /**
     * @brief Generate a work buffer for use with `reduceBlocking`, of the
     * appropriate size to perform a reduction over `arr`.
     *
     * Note: the returned array is only allocated on the host. You will need to
     * call `oarr.upload` before invoking `reduceBlocking`.
     */
    template<typename T>
    static inline aether::Array<T> makeBuffer(const aether::Array<T>& arr)
    {
        return makeBuffer<T>(idx_t(arr.samples()));
    }

    /**
     * @brief Perform a reduction over a GPU-allocated array using a
     * pre-allocated work buffer.
     *
     * Note: causes a blocking synchronization of the given stream.
     *
     * @tparam T Data type of the elements
     * @tparam OP Reduction operator to use. Must be callable as
     * `OP{}(a, b)` (aether's functor shape). Example:
     * `aether::SumOp<T>`
     * @param arr Input scalar array. The reduction will be performed over the
     * device copy, but it will not be modified. Must be already allocated on
     * the device.
     * @param oarr Output array. Used for intermediate copies. Must be allocated
     * on host and device. Its values on both host and device will be
     * overwritten. It is recommended to generate it via `makeBuffer`
     * to ensure it has the appropriate size. Must be distinct from `arr`.
     * @param init Initial value for the accumulation.
     * @param stream CUDA stream to use.
     */
    template<typename T, typename OP>
    static T reduceBlocking(aether::Array<T>& arr, aether::Array<T>& oarr,
        const T& init = 0, const cudaStream_t& stream = 0)
    {
        const CRefArrT<T> ref = arr.deviceView().as_const();
        return reduceBlocking<T, OP>(ref, oarr, init, stream);
    }

    /**
     * @brief Perform a reduction over a GPU-allocated array, allocating a work
     * buffer ad-hoc.
     *
     * Note: causes a blocking synchronization of the given stream.
     *
     * Note that this will allocate a new, temporary work buffer on host and
     * device. If performing multiple reductions with same-sized arrays, it may
     * be more efficient to generate the buffer once with `makeBuffer`
     * and reuse it with the overloaded version of this function.
     *
     * @tparam T Data type of the elements
     * @tparam OP Reduction operator to use. Must be callable as
     * `OP{}(a, b)` (aether's functor shape). Example:
     * `aether::SumOp<T>`
     * @param arr Input scalar array. The reduction will be performed over the
     * device copy, but it will not be modified. Must be already allocated on
     * the device.
     * @param init Initial value for the accumulation.
     * @param stream CUDA stream to use.
     */
    template<typename T, typename OP>
    static inline T reduceBlocking(aether::Array<T>& arr, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        aether::Array<T> buffer = makeBuffer<T>(idx_t(arr.samples()));
        buffer.upload(stream);
        return reduceBlocking<T, OP>(arr, buffer, init, stream);
    }

    /**
     * @copydoc Reduction::reduceBlocking(aether::Array<T>&, aether::Array<T>&, const T&, const cudaStream_t&)
     */
    template<typename T, typename OP>
    static T reduceBlocking(const CRefArrT<T>& arr,
        aether::Array<T>& oarr, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        T result;
        cuda::Graph graph
            = reduceBlocking<T, OP>(&result, arr, oarr, init, stream);

        cuda::Launcher launcher = graph.launcher();
        launcher.launch();
        launcher.synchronize();

        return result;
    }

    /**
     * @brief Build a graph to perform a reduction over a GPU-allocated array
     * using a pre-allocated work buffer.
     *
     * Note: causes a blocking synchronization of the given stream.
     *
     * @tparam T Data type of the elements
     * @tparam OP Reduction operator to use. Must be callable as
     * `OP{}(a, b)` (aether's functor shape). Example:
     * `aether::SumOp<T>`
     * @param result Pointer to the reduction result
     * @param arr Input scalar array. The reduction will be performed over the
     * device copy, but it will not be modified. Must be already allocated on
     * the device.
     * @param oarr Output array. Used for intermediate copies. Must be allocated
     * on host and device. Its values on both host and device will be
     * overwritten. It is recommended to generate it via `makeBuffer`
     * to ensure it has the appropriate size. Must be distinct from `arr`.
     * @param init Initial value for the accumulation.
     * @param stream CUDA stream to use.
     */
    template<typename T, typename OP, bool DirectCopy = false>
    static void reduceBlocking(cuda::Graph& graph, T* result,
        const CRefArrT<T>& arr, aether::Array<T>& oarr, const idx_t N,
        const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        const idx_t blockSize = reduce::detail::Core::blockSize;
        idx_t prevNBlocks     = N;
        idx_t nBlocks         = reduce::detail::Core::countBlocks<>(N);

        EAGLE_TRACE("Starting reduction over %i elements", N);
        EAGLE_ASSERT(arr.data() != oarr.deviceView().data(),
            "Reduction array and work buffer must be distinct!");

        EAGLE_ASSERT(idx_t(oarr.samples()) >= nBlocks,
            "Output array `oarr` is too small!");

        if (nBlocks == 1) {
            EAGLE_TRACE("Doing entire reduction on CPU");
            EAGLE_ASSERT(idx_t(oarr.samples()) >= N,
                "Output array `oarr` is too small!");
        }

        graph.stream(stream);
        cuda::StreamCapturer capturer(stream);

        /* Dynamic pointer for data retrieval - in case no kernel is launched */
        bool first              = true;
        CRefArrT<T> from        = arr;
        GRefArrT<T> ohandle     = oarr.deviceView();
        while (nBlocks > 1) {
            EAGLE_TRACE("Reduction iteration with %i blocks of %i threads",
                nBlocks, blockSize);
            /* Capture the kernel for this step */
            capturer.begin();
            reduce::detail::reduceOnce<T, OP, blockSize>
                <<<nBlocks, blockSize, 0, stream>>>(
                    from, ohandle, prevNBlocks, init);
            graph.addNode(cuda::CapturedGraph{ capturer.end() });
            /* read only the next sub-array */
            if (first) {
                from  = ohandle.as_const();
                first = false;
            }
            /* Prepare the next step */
            prevNBlocks = nBlocks;
            nBlocks     = reduce::detail::Core::countBlocks<>(nBlocks);
        }

        /* Last run */
        capturer.begin();
        reduce::detail::reduceOnce<T, OP, blockSize>
            <<<1, blockSize, 0, stream>>>(from, ohandle, prevNBlocks, init);
        graph.addNode(cuda::CapturedGraph{ capturer.end() });

        /* Copy the data back to the host for the finalization */
        if constexpr (DirectCopy) {
            capturer.begin();
            oarr.download(stream);
            graph.addNode(cuda::CapturedGraph{ capturer.end() });

            /* Prepare the lambda that finishes the reduction */
            EAGLE_TRACE(
                "Finishing reduction of last %i items on CPU", prevNBlocks);
            T* ro       = oarr.hostView().data();
            auto launch = [=] { *result = *ro; };
            graph.addHostNode(launch);
        } else {
            /* This move CANNOT be
             * routed through aether. Its SOURCE is an aether array's device
             * view, but its DESTINATION is `result` — a caller-supplied raw
             * host pointer, and every in-tree caller passes the address of a
             * plain stack `T` (see the blocking `reduceBlocking` overload
             * above), i.e. PAGEABLE CPU memory. `aether::copyAsync` refuses
             * every CPU-involving pair by design at BOTH granularities
             * (`aether/chunk/Copy.h`: "no async path for plain (pageable) CPU
             * memory"; `aether/view/Copy.h` defers to it verbatim), and
             * `aether::residency::StreamTransport` delegates to the same
             * function for every non-STAGED route — so there is no aether
             * spelling of "async D2H into pageable host memory" to reach for.
             * The aether-native alternative is `Array::download(stream)` plus
             * a host node, which is EXACTLY the `DirectCopy == true` branch
             * above: a different graph shape (one extra host node) and a
             * caller-selected option, not a routing change. Kept raw, and
             * LISTED. */
            capturer.begin();
            EAGLE_CHECK_ALWAYS(cudaMemcpyAsync(result,
                oarr.deviceView().data(), sizeof(T), cudaMemcpyDeviceToHost,
                stream));
            graph.addNode(cuda::CapturedGraph{ capturer.end() });
        }

        /* Add the lambda that finishes the reduction on the host */
    }

    /**
     * @copydoc Reduction::reduceBlocking(cuda::Graph&, T*, const eagle::CRefArrT<T>&, aether::Array<T>&, const idx_t, const T&, const cudaStream_t&)
     */
    template<typename T, typename OP, bool DirectCopy = false>
    static void reduceBlocking(cuda::Graph& graph, T* result,
        const CRefArrT<T>& arr, aether::Array<T>& oarr, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        reduceBlocking<T, OP, DirectCopy>(
            graph, result, arr, oarr, idx_t(arr.samples()), init, stream);
    }

    /**
     * @copydoc Reduction::reduceBlocking(cuda::Graph&, T*, const eagle::CRefArrT<T>&, aether::Array<T>&, const idx_t, const T&, const cudaStream_t&)
     */
    template<typename T, typename OP>
    static cuda::Graph reduceBlocking(T* result,
        const CRefArrT<T>& arr, aether::Array<T>& oarr, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        cuda::Graph graph;
        reduceBlocking<T, OP>(graph, result, arr, oarr, init, stream);
        return graph;
    }

    /**
     * @copydoc Reduction::reduceBlocking(const eagle::CRefArrT<T>&, const T&, const cudaStream_t&)
     */
    template<typename T, typename OP>
    static inline T reduceBlocking(
        const CRefArrT<T> arr, const T& init = 0,
        const cudaStream_t& stream = 0)
    {
        aether::Array<T> buffer = makeBuffer<T>(idx_t(arr.samples()));
        buffer.upload(stream);
        return reduceBlocking<T, OP>(arr, buffer, init, stream);
    }
};

#endif

} // namespace cuda
} // namespace eagle
