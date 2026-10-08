// Copyright 2026 Alessandro Masat
// SPDX-License-Identifier: Apache-2.0

#pragma once

// Facade: the Stream / Event / HostCallback CUDA-stream RAII wrappers were split
// one-per-class into graph/stream/. This header preserves a single include path
// so existing consumers (StreamCapturer.h, Graph.h) are unchanged.
#include "eagle/cuda/stream/Stream.h"
#include "eagle/cuda/stream/Event.h"
#include "eagle/cuda/stream/HostCallback.h"
