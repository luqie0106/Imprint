#include "imprint_renderer.h"
#include "acr_tone_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <stdexcept>

namespace {
bool span(const void* p, std::size_t bytes, std::size_t alignment) noexcept {
    if (!p) return false;
    const auto address=reinterpret_cast<std::uintptr_t>(p);
    return address%alignment==0 && bytes<=std::numeric_limits<std::uintptr_t>::max()-address;
}

bool overlap(const void* a, std::size_t a_bytes, const void* b, std::size_t b_bytes) noexcept {
    const auto x=reinterpret_cast<std::uintptr_t>(a);
    const auto y=reinterpret_cast<std::uintptr_t>(b);
    return x<=y ? y-x<a_bytes : x-y<b_bytes;
}

void write_stats(const imprint::acr::LogSceneStats& s, double* out) noexcept {
    out[0]=s.center; out[1]=s.low; out[2]=s.high; out[3]=s.median;
    out[4]=static_cast<double>(s.effective_count); out[5]=s.fallback ? 1.0 : 0.0;
}
} // namespace

im_status im_native_acr_guide_mask_prepared_f32(
    uint32_t width, uint32_t height, const float* log_guide, size_t source_count,
    double source_epsilon, uint32_t sigma_mode, uint32_t boundary_mode,
    float* delta, size_t destination_count, double* diagnostics, size_t diagnostic_count) {
    const std::uint64_t pixels=std::uint64_t{width}*height;
    if (!width || !height || width>65535u || height>65535u || pixels<2u || pixels>(1u<<26) ||
        source_count!=pixels || destination_count!=pixels || diagnostic_count!=20u ||
        sigma_mode>2u || (boundary_mode!=2u && boundary_mode!=4u) ||
        !std::isfinite(source_epsilon) || source_epsilon<=0.0 || source_epsilon>0x1p-12 ||
        static_cast<float>(source_epsilon)==0.0f)
        return IM_STATUS_INVALID_ARGUMENT;
    const auto bytes=static_cast<std::size_t>(pixels)*sizeof(float);
    constexpr std::size_t diagnostic_bytes=20*sizeof(double);
    if (!span(log_guide,bytes,alignof(float)) || !span(delta,bytes,alignof(float)) ||
        !span(diagnostics,diagnostic_bytes,alignof(double)) ||
        overlap(log_guide,bytes,delta,bytes) ||
        overlap(log_guide,bytes,diagnostics,diagnostic_bytes) ||
        overlap(delta,bytes,diagnostics,diagnostic_bytes))
        return IM_STATUS_INVALID_ARGUMENT;
    try {
        const auto candidate=imprint::acr::build_local_tone_mask(
            log_guide,width,height,source_epsilon,static_cast<int>(sigma_mode),boundary_mode);
        double metadata[20]{};
        write_stats(candidate.source_stats,metadata);
        write_stats(candidate.mask_stats,metadata+6);
        write_stats(candidate.aligned_mask_stats,metadata+12);
        metadata[18]=candidate.remap_sigma;
        metadata[19]=candidate.statistics_level;
        if (candidate.delta.pixels.size()!=pixels ||
            !std::all_of(std::begin(metadata),std::end(metadata),[](double v){return std::isfinite(v);}))
            return IM_STATUS_RUNTIME_ERROR;
        // No fallible operations remain: commit both caller-owned outputs.
        std::copy(candidate.delta.pixels.begin(),candidate.delta.pixels.end(),delta);
        std::copy(std::begin(metadata),std::end(metadata),diagnostics);
        return IM_STATUS_OK;
    } catch (const std::invalid_argument&) {
        return IM_STATUS_INVALID_ARGUMENT;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}
