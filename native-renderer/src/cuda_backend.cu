#include "backend.hpp"

#include <cuda_runtime.h>

#include <array>
#include <sstream>

namespace imprint {
namespace {

__device__ float clamp01(float x) { return fminf(1.0f, fmaxf(0.0f, x)); }
__device__ float smooth(float a, float b, float x) {
    float t = clamp01((x - a) / (b - a));
    return t * t * (3.0f - 2.0f * t);
}
__device__ float luminance(const float3 &c) { return c.x * 0.2126f + c.y * 0.7152f + c.z * 0.0722f; }
__device__ float3 sat_adjust(float3 c, float amount) {
    float y = luminance(c);
    return make_float3(clamp01(y + (c.x - y) * amount),
                       clamp01(y + (c.y - y) * amount),
                       clamp01(y + (c.z - y) * amount));
}

__device__ float3 smooth_chroma_gamut(float3 color) {
    const float y = luminance(color);
    const float3 chroma = make_float3(color.x - y, color.y - y, color.z - y);
    float limit = CUDART_INF_F;
    const float values[] = {chroma.x, chroma.y, chroma.z};
    for (int channel = 0; channel < 3; ++channel) {
        if (values[channel] > 1e-7f) limit = fminf(limit, (1.0f - y) / values[channel]);
        else if (values[channel] < -1e-7f) limit = fminf(limit, y / -values[channel]);
    }
    if (!isfinite(limit)) limit = 1.0f;
    const float delta = 1.0f - limit;
    const float scale = clamp01(0.5f * (1.0f + limit - sqrtf(delta * delta + 1e-10f)));
    return make_float3(y + chroma.x * scale, y + chroma.y * scale, y + chroma.z * scale);
}

__device__ float3 smooth_chroma_caps(float3 color, float3 caps) {
    const float y = luminance(color);
    const float3 chroma = make_float3(color.x - y, color.y - y, color.z - y);
    const float values[] = {chroma.x, chroma.y, chroma.z};
    const float bounds[] = {caps.x, caps.y, caps.z};
    float limit = CUDART_INF_F;
    for (int channel = 0; channel < 3; ++channel) {
        if (values[channel] > 1e-7f) limit = fminf(limit, (bounds[channel] - y) / values[channel]);
        else if (values[channel] < -1e-7f) limit = fminf(limit, y / -values[channel]);
    }
    if (!isfinite(limit)) limit = 1.0f;
    const float delta = 1.0f - limit;
    const float scale = clamp01(0.5f * (1.0f + limit - sqrtf(delta * delta + 1e-10f)));
    return make_float3(y + chroma.x * scale, y + chroma.y * scale, y + chroma.z * scale);
}

__device__ float3 dehaze(float3 original, const im_dehaze_params &p, const ImageStats &stats) {
    if (p.strength <= 1e-6f) return original;
    const float y = luminance(original);
    float3 air = make_float3(stats.air_r, stats.air_g, stats.air_b);
    const float air_mean = (air.x + air.y + air.z) / 3.0f;
    const float air_mix = 0.45f * p.color_protection * (0.65f + 0.35f * p.naturalness);
    air = make_float3(fminf(1.0f, fmaxf(0.35f, air.x * (1.0f - air_mix) + air_mean * air_mix)),
                      fminf(1.0f, fmaxf(0.35f, air.y * (1.0f - air_mix) + air_mean * air_mix)),
                      fminf(1.0f, fmaxf(0.35f, air.z * (1.0f - air_mix) + air_mean * air_mix)));
    const float omega = p.strength * (0.66f - 0.14f * p.fog_retention) *
                        (0.90f + 0.10f * (1.0f - p.naturalness)) * (0.92f + 0.08f * stats.haze_level);
    const float floor = 0.27f + 0.21f * p.fog_retention + 0.11f * p.naturalness;
    const float transmission = fminf(1.0f, fmaxf(floor, 1.0f - omega));
    float3 recovered = make_float3(clamp01((original.x - air.x) / transmission + air.x),
                                   clamp01((original.y - air.y) / transmission + air.y),
                                   clamp01((original.z - air.z) / transmission + air.z));
    const float amount = fminf(0.82f, fmaxf(0.0f, p.strength * (0.78f - 0.28f * p.fog_retention) *
                                               (0.90f + 0.10f * (1.0f - p.naturalness))));
    float3 natural = make_float3(original.x * (1.0f - amount) + recovered.x * amount,
                                 original.y * (1.0f - amount) + recovered.y * amount,
                                 original.z * (1.0f - amount) + recovered.z * amount);
    const float result_y = luminance(natural);
    const float source_y = fmaxf(y, 1e-4f);
    float3 luma_only = make_float3(clamp01(original.x * (result_y / source_y)),
                                   clamp01(original.y * (result_y / source_y)),
                                   clamp01(original.z * (result_y / source_y)));
    const float chroma_mix = p.color_protection * (0.72f + 0.28f * p.naturalness);
    natural = make_float3(natural.x * (1.0f - chroma_mix) + luma_only.x * chroma_mix,
                          natural.y * (1.0f - chroma_mix) + luma_only.y * chroma_mix,
                          natural.z * (1.0f - chroma_mix) + luma_only.z * chroma_mix);
    if (p.color_recovery > 1e-6f) {
        const float3 source_chroma = make_float3(original.x - y, original.y - y, original.z - y);
        const float source_chroma_norm = sqrtf(source_chroma.x * source_chroma.x +
                                               source_chroma.y * source_chroma.y +
                                               source_chroma.z * source_chroma.z);
        float confidence = clamp01((source_chroma_norm - 0.006f) / 0.084f);
        confidence = confidence * confidence * (3.0f - 2.0f * confidence);
        const float direction_norm = fmaxf(source_chroma_norm, 1e-6f);
        const float3 source_direction = make_float3(source_chroma.x / direction_norm,
                                                     source_chroma.y / direction_norm,
                                                     source_chroma.z / direction_norm);
        const float natural_y = luminance(natural);
        const float3 natural_chroma = make_float3(natural.x - natural_y, natural.y - natural_y,
                                                   natural.z - natural_y);
        const float natural_chroma_norm = sqrtf(natural_chroma.x * natural_chroma.x +
                                                natural_chroma.y * natural_chroma.y +
                                                natural_chroma.z * natural_chroma.z);
        const float aligned = fmaxf(0.0f, natural_chroma.x * source_direction.x +
                                           natural_chroma.y * source_direction.y +
                                           natural_chroma.z * source_direction.z);
        const float recovery_amount = p.color_recovery * p.strength;
        const float source_target = source_chroma_norm * (1.0f + 1.80f * p.color_recovery * confidence);
        const float natural_target = natural_chroma_norm * (1.0f + 0.30f * recovery_amount * confidence) * confidence;
        const float target_norm = fmaxf(fmaxf(aligned * confidence, natural_target), source_target);
        const float3 target = make_float3(source_direction.x * target_norm,
                                          source_direction.y * target_norm,
                                          source_direction.z * target_norm);
        const float high_pos = smooth(0.58f, 0.98f, y);
        const float shadow_pos = smooth(0.0f, 0.26f, 0.26f - y);
        const float protection = (1.0f - high_pos * p.highlight_protection) *
                                 (1.0f - shadow_pos * p.shadow_protection);
        const float requested = fminf(0.95f, fmaxf(0.0f, recovery_amount * protection *
                                                          (0.95f + 0.35f * (1.0f - confidence))));
        const float neutral_guard = p.color_protection * (1.0f - confidence) *
                                    (1.08f + 0.12f * p.naturalness) * protection;
        const float correction = fminf(0.95f, fmaxf(requested, neutral_guard));
        natural = make_float3(natural_y + natural_chroma.x * (1.0f - correction) + target.x * correction,
                              natural_y + natural_chroma.y * (1.0f - correction) + target.y * correction,
                              natural_y + natural_chroma.z * (1.0f - correction) + target.z * correction);
        natural = smooth_chroma_gamut(natural);
    }

    if (p.local_contrast > 1e-6f) {
        const float current_y = luminance(natural);
        const float amount_contrast = 0.55f * p.local_contrast;
        const float curve_y = current_y + amount_contrast * (2.0f * current_y - 1.0f) * current_y * (1.0f - current_y);
        const float scale = curve_y / fmaxf(current_y, 1e-4f);
        natural = make_float3(natural.x * scale, natural.y * scale, natural.z * scale);
    }

    const float highlight_position = smooth(0.58f, 0.98f, y);
    const float highlight_blend = clamp01(highlight_position * p.highlight_protection);
    natural = make_float3(natural.x * (1.0f - highlight_blend) + original.x * highlight_blend,
                          natural.y * (1.0f - highlight_blend) + original.y * highlight_blend,
                          natural.z * (1.0f - highlight_blend) + original.z * highlight_blend);
    const float shadow_position = smooth(0.0f, 0.26f, 0.26f - y);
    const float shadow_blend = clamp01(shadow_position * p.shadow_protection);
    natural = make_float3(natural.x * (1.0f - shadow_blend) + original.x * shadow_blend,
                          natural.y * (1.0f - shadow_blend) + original.y * shadow_blend,
                          natural.z * (1.0f - shadow_blend) + original.z * shadow_blend);

    if (p.brightness_protection > 1e-6f && stats.brightness_gain > 1.0f) {
        const float pre_y = luminance(natural);
        const float curve_y = pre_y * stats.brightness_gain /
                              (1.0f + (stats.brightness_gain - 1.0f) * pre_y);
        const float scale = pre_y > 1e-6f ? curve_y / pre_y : 1.0f;
        natural = make_float3(clamp01(natural.x * scale), clamp01(natural.y * scale), clamp01(natural.z * scale));
        natural = smooth_chroma_gamut(natural);
    }

    const float saturation_limit = fmaxf(0.0f, 0.97f - 1.5f / 65535.0f);
    const float final_y = luminance(natural);
    const float luma_cap = y < 0.97f ? saturation_limit : 1.0f;
    const float luma_scale = fminf(1.0f, luma_cap / fmaxf(final_y, 1e-4f));
    natural = make_float3(natural.x * luma_scale, natural.y * luma_scale, natural.z * luma_scale);
    const float3 caps = make_float3(original.x < 0.97f ? saturation_limit : 1.0f,
                                    original.y < 0.97f ? saturation_limit : 1.0f,
                                    original.z < 0.97f ? saturation_limit : 1.0f);
    if (p.color_recovery > 1e-6f) natural = smooth_chroma_caps(natural, caps);
    else natural = make_float3(fminf(natural.x, caps.x), fminf(natural.y, caps.y), fminf(natural.z, caps.z));
    return make_float3(clamp01(natural.x), clamp01(natural.y), clamp01(natural.z));
}

__device__ float3 basic_adjust(float3 c, const im_basic_params &p) {
    const float exposure = exp2f(p.exposure);
    c.x *= exposure; c.y *= exposure; c.z *= exposure;
    const float contrast = 1.0f + p.contrast / 100.0f;
    c = make_float3((c.x - 0.5f) * contrast + 0.5f,
                    (c.y - 0.5f) * contrast + 0.5f,
                    (c.z - 0.5f) * contrast + 0.5f);
    const float y = luminance(c);
    const float shadow_mask = powf(clamp01((0.62f - y) / 0.62f), 1.5f);
    const float highlight_mask = powf(clamp01((y - 0.38f) / 0.62f), 1.5f);
    const float tone = (p.shadows * 0.0022f + p.blacks * 0.0008f) * shadow_mask
                     + (p.highlights * 0.0018f + p.whites * 0.0008f) * highlight_mask;
    c.x += tone; c.y += tone; c.z += tone;
    const float adjusted_y = luminance(c);
    const float3 chroma = make_float3(c.x - adjusted_y, c.y - adjusted_y, c.z - adjusted_y);
    const float chroma_level = fmaxf(fabsf(chroma.x), fmaxf(fabsf(chroma.y), fabsf(chroma.z)));
    const float vibrance = 1.0f + p.vibrance / 100.0f * clamp01(1.0f - chroma_level);
    const float saturation = (1.0f + p.saturation / 100.0f) * vibrance;
    return make_float3(clamp01(adjusted_y + chroma.x * saturation),
                       clamp01(adjusted_y + chroma.y * saturation),
                       clamp01(adjusted_y + chroma.z * saturation));
}

__device__ float3 filter_controls(float3 c, const im_filter_params &p) {
    const float exposure = exp2f(p.exposure_ev);
    c.x *= exposure; c.y *= exposure; c.z *= exposure;
    c = make_float3((c.x - 0.18f) * p.contrast + 0.18f,
                    (c.y - 0.18f) * p.contrast + 0.18f,
                    (c.z - 0.18f) * p.contrast + 0.18f);
    c.x *= 1.0f + p.warmth * 0.12f;
    c.z *= 1.0f - p.warmth * 0.12f;
    c.y *= 1.0f + p.tint * 0.08f;
    c = sat_adjust(make_float3(clamp01(c.x), clamp01(c.y), clamp01(c.z)), p.saturation);
    const float fade = p.fade * 0.08f * (0.5f - luminance(c));
    return make_float3(clamp01(c.x + fade), clamp01(c.y + fade), clamp01(c.z + fade));
}

__device__ float curve_value(const uint16_t *curve, int channel, float value) {
    const float position = clamp01(value) * 255.0f;
    const int lo = static_cast<int>(floorf(position));
    const int hi = min(lo + 1, 255);
    const float fraction = position - lo;
    const float a = curve[channel * 256 + lo] / 65535.0f;
    const float b = curve[channel * 256 + hi] / 65535.0f;
    return a * (1.0f - fraction) + b * fraction;
}

__device__ float lut_value(const uint16_t *lut, int edge, int r, int g, int b, int channel) {
    const size_t at = ((static_cast<size_t>(r) * edge + g) * edge + b) * 3 + channel;
    return lut[at] / 65535.0f;
}

__device__ float3 lut_sample(const uint16_t *lut, int edge, float3 c) {
    const float xr = clamp01(c.x) * (edge - 1), xg = clamp01(c.y) * (edge - 1), xb = clamp01(c.z) * (edge - 1);
    const int r0 = static_cast<int>(floorf(xr)), g0 = static_cast<int>(floorf(xg)), b0 = static_cast<int>(floorf(xb));
    const int r1 = min(r0 + 1, edge - 1), g1 = min(g0 + 1, edge - 1), b1 = min(b0 + 1, edge - 1);
    const float fr = xr - r0, fg = xg - g0, fb = xb - b0;
    float result[3];
    for (int channel = 0; channel < 3; ++channel) {
        const float c000 = lut_value(lut, edge, r0, g0, b0, channel), c001 = lut_value(lut, edge, r0, g0, b1, channel);
        const float c010 = lut_value(lut, edge, r0, g1, b0, channel), c011 = lut_value(lut, edge, r0, g1, b1, channel);
        const float c100 = lut_value(lut, edge, r1, g0, b0, channel), c101 = lut_value(lut, edge, r1, g0, b1, channel);
        const float c110 = lut_value(lut, edge, r1, g1, b0, channel), c111 = lut_value(lut, edge, r1, g1, b1, channel);
        const float c00 = c000 * (1.0f - fb) + c001 * fb, c01 = c010 * (1.0f - fb) + c011 * fb;
        const float c10 = c100 * (1.0f - fb) + c101 * fb, c11 = c110 * (1.0f - fb) + c111 * fb;
        const float c0 = c00 * (1.0f - fg) + c01 * fg, c1 = c10 * (1.0f - fg) + c11 * fg;
        result[channel] = c0 * (1.0f - fr) + c1 * fr;
    }
    return make_float3(result[0], result[1], result[2]);
}

__global__ void render_kernel(const uint16_t *source, uint16_t *destination, size_t pixels,
                              im_dehaze_params dehaze_params, im_basic_params basic_params,
                              im_filter_params filter_params, ImageStats stats,
                              const uint16_t *curve, const uint16_t *lut, int lut_edge) {
    const size_t pixel = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    float3 color = make_float3(source[pixel * 3] / 65535.0f,
                               source[pixel * 3 + 1] / 65535.0f,
                               source[pixel * 3 + 2] / 65535.0f);
    color = dehaze(color, dehaze_params, stats);
    color = basic_adjust(color, basic_params);
    color = filter_controls(color, filter_params);
    const float3 curved = make_float3(curve_value(curve, 0, color.x),
                                      curve_value(curve, 1, color.y),
                                      curve_value(curve, 2, color.z));
    color = make_float3(color.x * (1.0f - filter_params.curve_mix) + curved.x * filter_params.curve_mix,
                        color.y * (1.0f - filter_params.curve_mix) + curved.y * filter_params.curve_mix,
                        color.z * (1.0f - filter_params.curve_mix) + curved.z * filter_params.curve_mix);
    if (lut_edge >= 2) {
        const float3 mapped = lut_sample(lut, lut_edge, color);
        color = make_float3(color.x * (1.0f - filter_params.lut_mix) + mapped.x * filter_params.lut_mix,
                            color.y * (1.0f - filter_params.lut_mix) + mapped.y * filter_params.lut_mix,
                            color.z * (1.0f - filter_params.lut_mix) + mapped.z * filter_params.lut_mix);
    }
    destination[pixel * 3] = static_cast<uint16_t>(__float2uint_rn(clamp01(color.x) * 65535.0f));
    destination[pixel * 3 + 1] = static_cast<uint16_t>(__float2uint_rn(clamp01(color.y) * 65535.0f));
    destination[pixel * 3 + 2] = static_cast<uint16_t>(__float2uint_rn(clamp01(color.z) * 65535.0f));
}

std::string cuda_error(const char *operation, cudaError_t result) {
    std::ostringstream stream;
    stream << operation << ": " << cudaGetErrorString(result);
    return stream.str();
}

} // namespace

class CudaBackend final : public Backend {
public:
    ~CudaBackend() override {
        for (void *&pointer : source_) free_device(pointer);
        free_device(curve_); free_device(lut_); free_device(output_);
    }

    const char *name() const override { return "CUDA"; }

    bool set_images(const std::array<ImageLevel, 3> &levels, std::string &error) override {
        std::array<void *, 3> next{};
        for (size_t i = 0; i < levels.size(); ++i) {
            if (!copy_to_device(&next[i], levels[i].pixels.data(), levels[i].pixels.size() * sizeof(uint16_t), error,
                                "Could not upload preview image to CUDA")) {
                for (void *pointer : next) free_device(pointer);
                return false;
            }
        }
        for (void *&pointer : source_) free_device(pointer);
        source_ = next;
        levels_ = levels;
        return true;
    }

    bool set_filter(const im_filter_params &params, const std::vector<uint16_t> &curve,
                    const std::vector<uint16_t> &lut, uint32_t lut_edge, std::string &error) override {
        const uint16_t dummy[] = {0, 0, 0};
        void *next_curve = nullptr, *next_lut = nullptr;
        if (!copy_to_device(&next_curve, curve.data(), curve.size() * sizeof(uint16_t), error,
                            "Could not upload filter curve to CUDA") ||
            !copy_to_device(&next_lut, lut.empty() ? dummy : lut.data(),
                            (lut.empty() ? 3 : lut.size()) * sizeof(uint16_t), error,
                            "Could not upload filter LUT to CUDA")) {
            free_device(next_curve); free_device(next_lut);
            return false;
        }
        free_device(curve_); free_device(lut_);
        curve_ = next_curve; lut_ = next_lut;
        filter_ = params; lut_edge_ = lut_edge;
        return true;
    }

    bool render_cached(unsigned level, const im_dehaze_params &dehaze, const im_basic_params &basic,
                       const ImageStats &stats,
                       uint16_t *destination, size_t destination_values, std::string &error) override {
        if (level >= levels_.size() || !source_[level]) {
            error = "Requested preview level has not been uploaded";
            return false;
        }
        return dispatch(source_[level], levels_[level], stats, dehaze, basic, destination, destination_values, error);
    }

    bool render_full(const ImageLevel &source, const ImageStats &stats,
                     const im_dehaze_params &dehaze, const im_basic_params &basic,
                     uint16_t *destination, size_t destination_values, std::string &error) override {
        void *device_source = nullptr;
        if (!copy_to_device(&device_source, source.pixels.data(), source.pixels.size() * sizeof(uint16_t), error,
                            "Could not upload full-resolution input to CUDA")) return false;
        const bool ok = dispatch(device_source, source, stats, dehaze, basic, destination, destination_values, error);
        free_device(device_source);
        return ok;
    }

private:
    static void free_device(void *&pointer) {
        if (pointer) cudaFree(pointer);
        pointer = nullptr;
    }

    static bool copy_to_device(void **destination, const void *source, size_t bytes,
                               std::string &error, const char *operation) {
        *destination = nullptr;
        cudaError_t result = cudaMalloc(destination, bytes);
        if (result != cudaSuccess) { error = cuda_error(operation, result); return false; }
        result = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
        if (result != cudaSuccess) { error = cuda_error(operation, result); free_device(*destination); return false; }
        return true;
    }

    bool dispatch(const void *source, const ImageLevel &level, const ImageStats &stats,
                  const im_dehaze_params &dehaze, const im_basic_params &basic,
                  uint16_t *destination, size_t destination_values, std::string &error) {
        const size_t pixels = static_cast<size_t>(level.width) * level.height;
        if (!destination || destination_values != pixels * 3 || !curve_ || !lut_) {
            error = "CUDA render buffers or output dimensions are invalid";
            return false;
        }
        const size_t bytes = destination_values * sizeof(uint16_t);
        if (output_bytes_ < bytes) {
            void *next = nullptr;
            const cudaError_t allocated = cudaMalloc(&next, bytes);
            if (allocated != cudaSuccess) { error = cuda_error("Could not allocate CUDA output", allocated); return false; }
            free_device(output_); output_ = next; output_bytes_ = bytes;
        }
        constexpr int threads = 256;
        render_kernel<<<static_cast<unsigned>((pixels + threads - 1) / threads), threads>>>(
            static_cast<const uint16_t *>(source), static_cast<uint16_t *>(output_), pixels,
            dehaze, basic, filter_, stats, static_cast<const uint16_t *>(curve_),
            static_cast<const uint16_t *>(lut_), static_cast<int>(lut_edge_));
        cudaError_t result = cudaGetLastError();
        if (result != cudaSuccess) { error = cuda_error("Could not launch CUDA renderer", result); return false; }
        result = cudaMemcpy(destination, output_, bytes, cudaMemcpyDeviceToHost);
        if (result != cudaSuccess) { error = cuda_error("Could not read CUDA render output", result); return false; }
        return true;
    }

    std::array<void *, 3> source_{};
    std::array<ImageLevel, 3> levels_;
    void *curve_ = nullptr;
    void *lut_ = nullptr;
    void *output_ = nullptr;
    size_t output_bytes_ = 0;
    im_filter_params filter_{};
    uint32_t lut_edge_ = 0;
};

std::unique_ptr<Backend> create_cuda_backend(std::string &error) {
    int device_count = 0;
    const cudaError_t result = cudaGetDeviceCount(&device_count);
    if (result != cudaSuccess || device_count == 0) {
        error = result == cudaSuccess ? "No CUDA device is available" : cuda_error("CUDA initialization failed", result);
        return nullptr;
    }
    return std::make_unique<CudaBackend>();
}

} // namespace imprint
