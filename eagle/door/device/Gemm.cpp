// SPDX-License-Identifier: Apache-2.0

// The device group's matrix products (device/Gemm.h): the row-major products of a gemm record, or of eg_gemm_run, mapped
// onto column-major cuBLAS calls. The few cuBLAS entry points are declared here and resolved at run time, as the driver's are, so the
// library links no cuBLAS and a machine without it still runs every other unit.
#include "eagle/door/device/Gemm.h"
#include "eagle/door/eg_unit_meta.h"

#include <dlfcn.h>

namespace eagle {
namespace door {

namespace {

constexpr int kOpN = 0, kOpT = 1;   // cublasOperation_t

using CreateFn = int (*)(void**);
using DestroyFn = int (*)(void*);
using SetStreamFn = int (*)(void*, void*);
template <class T>
using GemmFn = int (*)(void*, int, int, int, int, int, const T*, const T*, int, const T*, int, const T*, T*, int);

/** @brief The cuBLAS entry points the gemm unit calls, and why they could not all be had when ``create`` is null. */
struct Cublas {
    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    SetStreamFn set_stream = nullptr;
    GemmFn<float> sgemm = nullptr;
    GemmFn<double> dgemm = nullptr;
    std::string why;
};

/** @brief Open the first of libcublas.so.13, libcublas.so.12 that loads and has every entry point; else say what each lacked. */
Cublas open_cublas() {
    Cublas c;
    for (const char* name : { "libcublas.so.13", "libcublas.so.12" }) {
        void* lib = ::dlopen(name, RTLD_NOW | RTLD_LOCAL);
        if (lib == nullptr) {
            const char* why = ::dlerror();
            c.why += std::string(c.why.empty() ? "" : "; ") + name + ": " + (why != nullptr ? why : "does not load");
            continue;
        }
        const char* lacks = nullptr;
        const auto entry = [&](auto& fn, const char* symbol) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(::dlsym(lib, symbol));
            if (fn == nullptr && lacks == nullptr) lacks = symbol;
        };
        entry(c.create, "cublasCreate_v2");
        entry(c.destroy, "cublasDestroy_v2");
        entry(c.set_stream, "cublasSetStream_v2");
        entry(c.sgemm, "cublasSgemm_v2");
        entry(c.dgemm, "cublasDgemm_v2");
        if (lacks == nullptr) return c;
        c = Cublas{ nullptr, nullptr, nullptr, nullptr, nullptr, c.why + (c.why.empty() ? "" : "; ") + name + " lacks " + lacks };
        ::dlclose(lib);
    }
    return c;
}

/** @brief The process's cuBLAS, opened at the first call; EG_FAULT naming the files tried when it cannot be. */
const Cublas& cublas() {
    static const Cublas c = open_cublas();
    if (c.create == nullptr)
        refuse(EG_FAULT, "a device matrix product needs cuBLAS and it cannot be opened (tried " + c.why + ")");
    return c;
}

/** @brief Refuse EG_FAULT naming @p call unless the cuBLAS status @p status is success. */
void check(int status, const char* call) {
    if (status != 0) refuse(EG_FAULT, std::string(call) + " returned cuBLAS status " + std::to_string(status));
}

/** @brief The column-major call of one row-major product: cuBLAS sees a row-major (r, c) plane with row stride ld as the
 *  (c, r) matrix of the transpose, so (X Y)^T = Y^T X^T swaps the operands and inverts their flags; a transposed C view
 *  is already the column-major result, so the operands keep their order and take the opposite flags. */
struct Mapped {
    int op_x, op_y, m, n, k, ldx, ldy;
    bool swapped;   ///< X is B and Y is A
};

Mapped map_call(const GemmCall& c, int ld_a, int ld_b) {
    const bool ta = c.flags & EG_GEMM_TRANS_A, tb = c.flags & EG_GEMM_TRANS_B;
    if (c.flags & EG_GEMM_TRANS_C)
        return { ta ? kOpN : kOpT, tb ? kOpN : kOpT, static_cast<int>(c.m), static_cast<int>(c.n), static_cast<int>(c.k), ld_a, ld_b, false };
    return { tb ? kOpT : kOpN, ta ? kOpT : kOpN, static_cast<int>(c.n), static_cast<int>(c.m), static_cast<int>(c.k), ld_b, ld_a, true };
}

/** @brief The first element of @p slot's bound plane and its row stride for @p rows rows. */
std::pair<char*, int> view(const eg_plan_s& plan, std::int32_t slot, std::int64_t rows) {
    const DLTensor& t = plan.bindings[static_cast<std::size_t>(slot)].plane->tensor->dl_tensor;
    return { static_cast<char*>(t.data) + t.byte_offset, static_cast<int>(elements(t) / rows) };
}

}  // namespace

void gemm_handle_make(eg_ctx& ctx, std::uint64_t stream) {
    const Cublas& b = cublas();
    if (ctx.dev.gemm_handles.count(stream) != 0) return;
    void* handle = nullptr;
    check(b.create(&handle), "cublasCreate");
    if (const int s = b.set_stream(handle, reinterpret_cast<void*>(stream)); s != 0) {
        b.destroy(handle);
        check(s, "cublasSetStream");
    }
    ctx.dev.gemm_handles[stream] = reinterpret_cast<std::uint64_t>(handle);
}

void gemm_handle_drop(eg_ctx& ctx, std::uint64_t stream) noexcept {
    const auto at = ctx.dev.gemm_handles.find(stream);
    if (at == ctx.dev.gemm_handles.end()) return;
    cublas().destroy(reinterpret_cast<void*>(at->second));
    ctx.dev.gemm_handles.erase(at);
}

void gemm_handles_free(eg_ctx& ctx) noexcept {
    for (const auto& h : ctx.dev.gemm_handles) cublas().destroy(reinterpret_cast<void*>(h.second));
    ctx.dev.gemm_handles.clear();
}

void gemm_product(const eg_ctx& ctx, std::uint64_t stream, const GemmCall& c, bool f32, const void* a, int ld_a, const void* b, int ld_b, void* out, int ld_c) {
    const Cublas& cb = cublas();
    const auto at = ctx.dev.gemm_handles.find(stream);
    if (at == ctx.dev.gemm_handles.end()) refuse(EG_FAULT, "no cuBLAS handle was made for the stream in scope");
    void* handle = reinterpret_cast<void*>(at->second);
    const Mapped m = map_call(c, ld_a, ld_b);
    const void* x = m.swapped ? b : a;
    const void* y = m.swapped ? a : b;
    const bool accumulate = c.flags & EG_GEMM_ACCUMULATE;
    if (f32) {
        const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
        check(cb.sgemm(handle, m.op_x, m.op_y, m.m, m.n, m.k, &alpha, static_cast<const float*>(x), m.ldx, static_cast<const float*>(y), m.ldy,
                       &beta, static_cast<float*>(out), ld_c), "cublasSgemm");
    } else {
        const double alpha = 1.0, beta = accumulate ? 1.0 : 0.0;
        check(cb.dgemm(handle, m.op_x, m.op_y, m.m, m.n, m.k, &alpha, static_cast<const double*>(x), m.ldx, static_cast<const double*>(y), m.ldy,
                       &beta, static_cast<double*>(out), ld_c), "cublasDgemm");
    }
}

void gemm_launch(const eg_plan_s& plan, std::uint64_t stream) {
    const bool f32 = plan.unit->scalar_type == EG_SCALAR_FLOAT32;
    for (const GemmCall& c : plan.gemm) {
        const auto [a, ld_a] = view(plan, c.a, c.a_rows);
        const auto [bp, ld_b] = view(plan, c.b, c.b_rows);
        const auto [cp, ld_c] = view(plan, c.c, c.c_rows);
        gemm_product(*plan.ctx, stream, c, f32, a, ld_a, bp, ld_b, cp, ld_c);
    }
}

}  // namespace door
}  // namespace eagle
