/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef MULTI_SEGMENT_VMM_HELPERS_HPP
#define MULTI_SEGMENT_VMM_HELPERS_HPP

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vector>

#include "rocmwrap.h"

// hipMemLocationTypeHost is undeclared without the CMake host probe; outside the gate the shim is harmless (HOST == 2).
#if !NCCL_CUMEM_HOST_GATE
#ifndef hipMemLocationTypeHost
#define hipMemLocationTypeHost (static_cast<hipMemLocationType>(2))
#endif
#endif

// Shared by hybrid, RMA and CE tests. Not gated on a transport backend.
namespace RCCLTestHelpers
{

// Unmap, release, and free a reserved VA. `segSizes` wins when non-empty;
// otherwise each mapped handle is `uniformSegSize` bytes.
inline void ReleaseMappedVmm(hipDeviceptr_t base, size_t totalSize,
                             const std::vector<hipMemGenericAllocationHandle_t>& handles,
                             const std::vector<size_t>& segSizes, size_t uniformSegSize)
{
    size_t off = 0;
    for (size_t i = 0; i < handles.size(); ++i)
    {
        const size_t len = segSizes.empty() ? uniformSegSize : segSizes[i];
        if (base != 0 && len != 0)
        {
            hipDeviceptr_t segVa = reinterpret_cast<hipDeviceptr_t>(
                reinterpret_cast<uintptr_t>(base) + off);
            (void)hipMemUnmap(segVa, len);
        }
        if (handles[i] != 0) (void)hipMemRelease(handles[i]);
        off += len;
    }
    if (base != 0 && totalSize != 0) (void)hipMemAddressFree(base, totalSize);
}

// Pinned, fd-exportable VMM allocation properties at one location.
inline hipMemAllocationProp MakeVmmProp(hipMemLocationType locType, int locId, bool gpuDirectRdma)
{
    hipMemAllocationProp prop            = {};
    prop.type                            = hipMemAllocationTypePinned;
    prop.location.type                   = locType;
    prop.location.id                     = locId;
    prop.requestedHandleType             = hipMemHandleTypePosixFileDescriptor;
    prop.allocFlags.gpuDirectRDMACapable = gpuDirectRdma ? 1 : 0;
    return prop;
}

// Contiguous VA range backed by N distinct VMM allocations mapped back-to-back.
struct MultiSegmentVmmBuffer
{
    void*                                        ptr        = nullptr;
    size_t                                       totalSize  = 0;
    size_t                                       segSize    = 0;
    int                                          nSegments  = 0;
    hipDeviceptr_t                               base       = 0;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    std::vector<size_t>                          segSizes;

    bool valid() const { return ptr != nullptr && nSegments > 0; }
};

// One VMM segment: its size, where its memory lives, and optionally an existing (e.g. imported) handle to map
// instead of a new allocation. AllocSegmentedVmm owns a given handle: it maps it or releases it on failure.
struct VmmSegSpec
{
    size_t                          bytes;
    hipMemLocationType              loc    = hipMemLocationTypeDevice;
    hipMemGenericAllocationHandle_t handle = 0;
};

// The AllocSegmentedVmm step that failed, for callers that report it.
enum class VmmSegStep { None, Setup, Create, Map, DeviceAccess, HostAccess };
struct VmmSegFailure
{
    VmmSegStep step    = VmmSegStep::None;
    int        segment = -1;
    hipError_t err     = hipSuccess;
};

// Map one granularity-rounded segment per spec contiguously on `dev`, in a VA aligned to `alignment`
// (0: the granularity). Device RW over every segment; host RW on host segments is best-effort unless
// `requireHostAccess`. `segSize` is the common size, or 0 when the sizes differ. Returns false (cleaned up)
// on any HIP failure and, when `failure` is set, says which step failed.
inline bool AllocSegmentedVmm(int dev, const std::vector<VmmSegSpec>& specs, MultiSegmentVmmBuffer* out,
                              size_t alignment = 0, bool requireHostAccess = false,
                              VmmSegFailure* failure = nullptr)
{
    const int nSegments = static_cast<int>(specs.size());
    hipDeviceptr_t base = 0;
    size_t totalSize = 0;
    std::vector<size_t> segSizes;
    std::vector<hipMemGenericAllocationHandle_t> handles;
    handles.reserve(nSegments);

    // Releases everything mapped so far and every given handle not yet mapped.
    auto fail = [&](VmmSegStep step, int segment, hipError_t err) {
        ReleaseMappedVmm(base, totalSize, handles, segSizes, /*uniformSegSize=*/0);
        for (size_t i = handles.size(); i < specs.size(); ++i)
            if (specs[i].handle != 0) (void)hipMemRelease(specs[i].handle);
        if (failure) *failure = {step, segment, err};
        return false;
    };
    if (out == nullptr || specs.empty()) return fail(VmmSegStep::Setup, -1, hipErrorInvalidValue);
    *out = MultiSegmentVmmBuffer{};

    auto propOf = [dev](hipMemLocationType loc) {
        const bool host = loc == hipMemLocationTypeHost;
        return MakeVmmProp(loc, host ? 0 : dev, /*gpuDirectRdma=*/!host);
    };
    size_t granularity = 0;
    bool anyHost = false;
    for (const VmmSegSpec& spec : specs)
    {
        const hipMemAllocationProp prop = propOf(spec.loc);
        size_t g = 0;
        const hipError_t err = hipMemGetAllocationGranularity(&g, &prop, hipMemAllocationGranularityMinimum);
        if (err != hipSuccess || g == 0) return fail(VmmSegStep::Setup, -1, err);
        granularity = std::max(granularity, g);
        anyHost = anyHost || spec.loc == hipMemLocationTypeHost;
    }
    if (alignment == 0) alignment = granularity;
    if (alignment % granularity != 0) return fail(VmmSegStep::Setup, -1, hipErrorInvalidValue);

    segSizes.reserve(nSegments);
    for (const VmmSegSpec& spec : specs)
    {
        segSizes.push_back(((spec.bytes + granularity - 1) / granularity) * granularity);
        totalSize += segSizes.back();
    }
    size_t segSize = segSizes[0];
    for (size_t len : segSizes)
        if (len != segSize) segSize = 0;

    const hipError_t reserveErr = hipMemAddressReserve(&base, totalSize, alignment, 0, 0);
    if (reserveErr != hipSuccess)
    {
        base = 0;
        return fail(VmmSegStep::Setup, -1, reserveErr);
    }

    auto segVa = [&](size_t off) {
        return reinterpret_cast<hipDeviceptr_t>(reinterpret_cast<uintptr_t>(base) + off);
    };

    hipMemAccessDesc accessDesc = {};
    accessDesc.location.type    = hipMemLocationTypeDevice;
    accessDesc.location.id      = dev;
    accessDesc.flags            = hipMemAccessFlagsProtReadWrite;

    size_t off = 0;
    for (int s = 0; s < nSegments; ++s)
    {
        const hipMemAllocationProp prop = propOf(specs[s].loc);
        hipMemGenericAllocationHandle_t h = specs[s].handle;
        hipError_t err = h != 0 ? hipSuccess : hipMemCreate(&h, segSizes[s], &prop, 0);
        if (err != hipSuccess) return fail(VmmSegStep::Create, s, err);
        err = hipMemMap(segVa(off), segSizes[s], 0, h, 0);
        if (err != hipSuccess)
        {
            if (specs[s].handle == 0) (void)hipMemRelease(h);
            return fail(VmmSegStep::Map, s, err);
        }
        handles.push_back(h);
        // A mixed-location range is granted per segment.
        if (anyHost && (err = hipMemSetAccess(segVa(off), segSizes[s], &accessDesc, 1)) != hipSuccess)
            return fail(VmmSegStep::DeviceAccess, s, err);
        off += segSizes[s];
    }

    // One access grant over a device-only range.
    if (!anyHost)
    {
        const hipError_t err = hipMemSetAccess(base, totalSize, &accessDesc, 1);
        if (err != hipSuccess) return fail(VmmSegStep::DeviceAccess, -1, err);
    }

    // Best-effort unless required: callers touch host segments only through hipMemcpy/hipMemset.
    hipMemAccessDesc hostAccess = {};
    hostAccess.location.type    = hipMemLocationTypeHost;
    hostAccess.location.id      = 0;
    hostAccess.flags            = hipMemAccessFlagsProtReadWrite;
    off = 0;
    for (int s = 0; s < nSegments; ++s)
    {
        if (specs[s].loc == hipMemLocationTypeHost)
        {
            const hipError_t err = hipMemSetAccess(segVa(off), segSizes[s], &hostAccess, 1);
            if (err != hipSuccess)
            {
                if (requireHostAccess) return fail(VmmSegStep::HostAccess, s, err);
                (void)hipGetLastError();
                fprintf(stderr, "[ WARNING  ] host access grant on VMM segment %d failed (%s); continuing\n", s,
                        hipGetErrorString(err));
            }
        }
        off += segSizes[s];
    }

    out->ptr       = reinterpret_cast<void*>(base);
    out->base      = base;
    out->totalSize = totalSize;
    out->segSize   = segSize;
    out->nSegments = nSegments;
    out->handles   = std::move(handles);
    out->segSizes  = std::move(segSizes);
    return true;
}

// Device segments of the given sizes.
inline bool AllocSegmentedVmm(int dev, const std::vector<size_t>& segBytes, MultiSegmentVmmBuffer* out)
{
    std::vector<VmmSegSpec> specs;
    specs.reserve(segBytes.size());
    for (size_t bytes : segBytes) specs.push_back({bytes});
    return AllocSegmentedVmm(dev, specs, out);
}

// DeepEP ElasticSymmetricMemory without hybrid import: [GPU][CPU] in one 2 MiB-aligned VA.
inline bool AllocDeepEpElasticVmm(int dev, size_t gpuBytes, size_t cpuBytes, MultiSegmentVmmBuffer* out)
{
    constexpr size_t kDeepEpAlignment = 2u * 1024 * 1024;
    if (gpuBytes == 0 || cpuBytes == 0 || gpuBytes % kDeepEpAlignment != 0 || cpuBytes % kDeepEpAlignment != 0)
        return false;
    return AllocSegmentedVmm(dev, {{gpuBytes}, {cpuBytes, hipMemLocationTypeHost}}, out, kDeepEpAlignment);
}

// Map `nSegments` granularity-rounded segments of `segBytes` each contiguously on `dev`.
inline bool AllocMultiSegmentVmm(int dev, int nSegments, size_t segBytes,
                                 MultiSegmentVmmBuffer* out)
{
    if (nSegments <= 0) return false;
    return AllocSegmentedVmm(dev, std::vector<size_t>(nSegments, segBytes), out);
}

// Release in HIP-required order: unmap each segment, release each handle, free VA.
inline void FreeMultiSegmentVmm(MultiSegmentVmmBuffer& b)
{
    if (b.ptr == nullptr && b.base == 0) return;
    ReleaseMappedVmm(b.base, b.totalSize, b.handles, b.segSizes, b.segSize);
    b = MultiSegmentVmmBuffer{};
}

} // namespace RCCLTestHelpers

#endif // MULTI_SEGMENT_VMM_HELPERS_HPP
