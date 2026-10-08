// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "eagle/cuda/stream/Stream.h"

#include <functional>
#include <memory>

#ifndef EAGLE_CPU_ONLY

namespace eagle {
namespace cuda {

/** @brief Host-callback creator */
class HostCallback {
    /** @brief Handle struct to manage the lambda-based initialization */
    struct HostLambdaWrapper {
        std::function<void()> fn;
        static void CUDART_CB trampoline(void* data)
        {
            auto* wrapper = static_cast<HostLambdaWrapper*>(data);
            wrapper->fn();
        }
    };

public:
    /** @brief Construct with the given lambda function  */
    explicit HostCallback(std::function<void()> lambda)
        : wrapper_{ std::make_shared<HostLambdaWrapper>(
              HostLambdaWrapper{ std::move(lambda) }) }
    {
    }

    /** @brief Enqueue the callback in the given stream */
    void launch(const Stream& stream)
    {
        EAGLE_CHECK_ALWAYS(cudaLaunchHostFunc(
            stream.cuda(), HostLambdaWrapper::trampoline, wrapper_.get()));
    }
    void launch(const cudaStream_t& stream)
    {
        EAGLE_CHECK_ALWAYS(cudaLaunchHostFunc(
            stream, HostLambdaWrapper::trampoline, wrapper_.get()));
    }

private:
    std::shared_ptr<HostLambdaWrapper> wrapper_;
};

} // namespace cuda
} // namespace eagle

#endif
