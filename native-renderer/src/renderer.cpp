#include "backend.hpp"
#include "dehaze_reference.hpp"
#include "warp_kernels.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace {

thread_local std::string g_create_error;
constexpr uint64_t kMaxSharedBufferBytes = 1ull << 30;

bool valid_dehaze(const im_dehaze_params &p) {
    const float values[] = {p.strength, p.naturalness, p.fog_retention, p.local_contrast,
                            p.color_recovery, p.color_protection, p.highlight_protection,
                            p.shadow_protection, p.brightness_protection};
    for (float value : values) if (!std::isfinite(value) || value < 0.0f || value > 1.0f) return false;
    return true;
}

bool valid_basic(const im_basic_params &p) {
    const float values[] = {p.exposure, p.contrast, p.highlights, p.shadows, p.whites,
                            p.blacks, p.vibrance, p.saturation};
    const float minimum[] = {-5.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f, -100.0f};
    const float maximum[] = {5.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    for (size_t i = 0; i < 8; ++i) {
        if (!std::isfinite(values[i]) || values[i] < minimum[i] || values[i] > maximum[i]) return false;
    }
    return true;
}

bool valid_filter(const im_filter_params &p) {
    const float values[] = {p.curve_mix, p.lut_mix, p.exposure_ev, p.contrast,
                            p.saturation, p.warmth, p.tint, p.fade};
    const float minimum[] = {0.0f, 0.0f, -8.0f, 0.0f, 0.0f, -1.0f, -1.0f, -1.0f};
    const float maximum[] = {1.0f, 1.0f, 8.0f, 4.0f, 3.0f, 1.0f, 1.0f, 1.0f};
    for (size_t i = 0; i < 8; ++i) {
        if (!std::isfinite(values[i]) || values[i] < minimum[i] || values[i] > maximum[i]) return false;
    }
    return true;
}

template <typename T>
bool camera_profile_byte_length(size_t count, size_t *bytes) {
    if (!bytes || count > std::numeric_limits<size_t>::max() / sizeof(T)) return false;
    *bytes = count * sizeof(T);
    return true;
}

bool camera_profile_ranges_overlap(const void *left, size_t left_bytes,
                                   const void *right, size_t right_bytes) {
    const uintptr_t left_start = reinterpret_cast<uintptr_t>(left);
    const uintptr_t right_start = reinterpret_cast<uintptr_t>(right);
    const uintptr_t address_max = std::numeric_limits<uintptr_t>::max();
    if (left_bytes > address_max - left_start || right_bytes > address_max - right_start) return true;
    const uintptr_t left_end = left_start + left_bytes;
    const uintptr_t right_end = right_start + right_bytes;
    return left_start < right_end && right_start < left_end;
}

bool valid_camera_profile_config(const float *constants22, size_t constants_count,
                                 const float *look_table, size_t look_values,
                                 uint32_t hue_count, uint32_t saturation_count,
                                 uint32_t value_count, uint32_t look_encoding,
                                 const float *tone_curve, size_t tone_values) {
    if (!constants22 || !look_table || !tone_curve || constants_count != 22 ||
        hue_count < 1 || saturation_count < 2 || value_count < 2 ||
        hue_count > 1000000 || saturation_count > 1000000 || value_count > 1000000 ||
        look_encoding > 1 || tone_values > 4u * 1024u * 1024u) return false;
    const uint64_t entries = static_cast<uint64_t>(hue_count) * saturation_count * value_count;
    if (entries > 1000000 || entries * 3u != look_values) return false;
    for (size_t index = 0; index < 22; ++index) {
        if (!std::isfinite(constants22[index])) return false;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        if (constants22[9 + channel] <= 0.0f || constants22[9 + channel] > 1.0f) return false;
    }
    if (constants22[12] < 0x1p-40f || constants22[12] > 0x1p40f) return false;
    for (size_t index = 0; index < 9; ++index) {
        if (std::abs(constants22[index]) > 1.0e6f ||
            std::abs(constants22[13 + index]) > 1.0e6f) return false;
    }
    for (size_t entry = 0; entry < static_cast<size_t>(entries); ++entry) {
        const float hue_delta = look_table[entry * 3];
        const float saturation = look_table[entry * 3 + 1];
        const float value = look_table[entry * 3 + 2];
        if (!std::isfinite(hue_delta) || !std::isfinite(saturation) || !std::isfinite(value) ||
            hue_delta < -360.0f || hue_delta > 360.0f ||
            saturation < 0.0f || saturation > 16.0f || value < 0.0f || value > 16.0f) return false;
    }
    if (tone_values < 4 || (tone_values & 1u) != 0u) return false;
    const size_t points = tone_values / 2;
    if (std::abs(tone_curve[0]) > 1.0e-6f ||
        std::abs(tone_curve[(points - 1) * 2] - 1.0f) > 1.0e-6f ||
        std::abs(tone_curve[1]) > 1.0e-6f ||
        std::abs(tone_curve[(points - 1) * 2 + 1] - 1.0f) > 1.0e-6f) return false;
    for (size_t point = 0; point < points; ++point) {
        const float x = tone_curve[point * 2];
        const float y = tone_curve[point * 2 + 1];
        if (!std::isfinite(x) || !std::isfinite(y) || x < 0.0f || x > 1.0f ||
            y < 0.0f || y > 1.0f) return false;
        if (point > 0) {
            const float previous_x = tone_curve[(point - 1) * 2];
            const float previous_y = tone_curve[(point - 1) * 2 + 1];
            if (x <= previous_x || y < previous_y - 1.0e-7f) return false;
        }
    }
    return true;
}

imprint::ImageLevel resize_box(const imprint::ImageLevel &source, uint32_t divisor) {
    imprint::ImageLevel output;
    output.width = std::max(1u, (source.width + divisor - 1) / divisor);
    output.height = std::max(1u, (source.height + divisor - 1) / divisor);
    output.pixels.resize(static_cast<size_t>(output.width) * output.height * 3);
    for (uint32_t y = 0; y < output.height; ++y) {
        const uint32_t y0 = static_cast<uint32_t>((static_cast<uint64_t>(y) * source.height) / output.height);
        const uint32_t y1 = std::max(y0 + 1, static_cast<uint32_t>((static_cast<uint64_t>(y + 1) * source.height) / output.height));
        for (uint32_t x = 0; x < output.width; ++x) {
            const uint32_t x0 = static_cast<uint32_t>((static_cast<uint64_t>(x) * source.width) / output.width);
            const uint32_t x1 = std::max(x0 + 1, static_cast<uint32_t>((static_cast<uint64_t>(x + 1) * source.width) / output.width));
            uint64_t sums[3] = {0, 0, 0};
            uint64_t count = 0;
            for (uint32_t sy = y0; sy < std::min(y1, source.height); ++sy) {
                for (uint32_t sx = x0; sx < std::min(x1, source.width); ++sx) {
                    const size_t at = (static_cast<size_t>(sy) * source.width + sx) * 3;
                    for (size_t channel = 0; channel < 3; ++channel) sums[channel] += source.pixels[at + channel];
                    ++count;
                }
            }
            const size_t dest = (static_cast<size_t>(y) * output.width + x) * 3;
            for (size_t channel = 0; channel < 3; ++channel) {
                output.pixels[dest + channel] = static_cast<uint16_t>((sums[channel] + count / 2) / count);
            }
        }
    }
    return output;
}

std::vector<uint16_t> identity_curve() {
    std::vector<uint16_t> curve(3 * 256);
    for (size_t i = 0; i < 256; ++i) {
        const uint16_t value = static_cast<uint16_t>((i * 65535u + 127u) / 255u);
        curve[i] = curve[256 + i] = curve[512 + i] = value;
    }
    return curve;
}

} // namespace

struct im_renderer {
    std::unique_ptr<imprint::Backend> backend;
    std::array<imprint::ImageLevel, 3> levels;
    std::array<imprint::ImageStats, 3> stats{};
    std::vector<uint16_t> curve;
    std::vector<uint16_t> lut;
    uint32_t lut_edge = 0;
    im_filter_params filter{};
    std::vector<uint16_t> output;
    uint32_t output_width = 0;
    uint32_t output_height = 0;
    bool has_camera_profile_source = false;
    uint32_t camera_profile_source_width = 0;
    uint32_t camera_profile_source_height = 0;
    std::shared_ptr<imprint::SharedBuffer> camera_profile_source_shared;
    bool has_camera_profile_transfer_source = false;
    uint32_t camera_profile_transfer_width = 0;
    uint32_t camera_profile_transfer_height = 0;
    mutable std::mutex mutex;
    mutable std::string error;
};

struct im_shared_buffer {
    std::shared_ptr<imprint::SharedBuffer> owner;
};

extern "C" {

im_status im_native_dehaze_spatial_run(const uint16_t *rgb16, uint32_t width, uint32_t height,
                                       const float *transmission, size_t transmission_count,
                                       const float *airlight_rgb, const im_dehaze_params *params,
                                       uint16_t *destination, size_t destination_samples) {
    if (!rgb16 || !transmission || !airlight_rgb || !params || !destination ||
        width == 0 || height == 0 || width > 65535 || height > 65535 ||
        !valid_dehaze(*params)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    const uint64_t pixels64 = static_cast<uint64_t>(width) * height;
    if (pixels64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        pixels64 > (1ull << 29) / 3) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const size_t pixels = static_cast<size_t>(pixels64);
    const size_t required_samples = pixels * 3;
    if (transmission_count != pixels || destination_samples < required_samples) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(airlight_rgb[channel])) return IM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        if (!std::isfinite(transmission[pixel])) return IM_STATUS_INVALID_ARGUMENT;
    }

    if (params->strength <= 1e-6f) {
        std::memmove(destination, rgb16, required_samples * sizeof(uint16_t));
        return IM_STATUS_OK;
    }

    try {
        imprint::apply_dehaze_spatial_reference(rgb16, width, height, transmission,
                                                airlight_rgb, *params, destination);
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &) {
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_spatial_full(im_renderer *renderer, uint32_t width, uint32_t height,
                                          const uint16_t *rgb16, size_t value_count,
                                          const float *transmission, size_t transmission_count,
                                          const float *airlight_rgb, const im_dehaze_params *params,
                                          uint16_t *destination, size_t destination_samples) {
    if (!renderer || !rgb16 || !transmission || !airlight_rgb || !params || !destination ||
        !width || !height || width > 65535 || height > 65535 || !valid_dehaze(*params))
        return IM_STATUS_INVALID_ARGUMENT;
    const uint64_t pixels64 = static_cast<uint64_t>(width) * height;
    if (pixels64 > (1ull << 29) / 3 || transmission_count != pixels64 ||
        value_count != pixels64 * 3 || destination_samples < pixels64 * 3)
        return IM_STATUS_INVALID_ARGUMENT;
    for (size_t channel = 0; channel < 3; ++channel)
        if (!std::isfinite(airlight_rgb[channel])) return IM_STATUS_INVALID_ARGUMENT;
    for (size_t pixel = 0; pixel < transmission_count; ++pixel)
        if (!std::isfinite(transmission[pixel])) return IM_STATUS_INVALID_ARGUMENT;
    if (params->strength <= 1e-6f) {
        std::memmove(destination, rgb16, value_count * sizeof(uint16_t));
        return IM_STATUS_OK;
    }
    try {
        imprint::ImageLevel source;
        source.width = width;
        source.height = height;
        source.pixels.assign(rgb16, rgb16 + value_count);
        imprint::ImageStats stats{0.0f, 0.0f, airlight_rgb[0], airlight_rgb[1],
                                  airlight_rgb[2], 0.0f, 1.0f};
        stats.brightness_gain = imprint::dehaze_spatial_brightness_gain(
            source.pixels.data(), width, height, transmission, airlight_rgb, *params);
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->render_spatial_full(source, transmission, stats, *params,
                                                    destination, destination_samples, error)) {
            renderer->error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while rendering spatial dehaze";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    }
}

static im_status render_physical_float_impl(im_renderer *renderer,
                                            uint32_t width, uint32_t height,
                                            const float *source, size_t source_values,
                                            const float *transmission, size_t transmission_count,
                                            const float *airlight_rgb,
                                            const im_dehaze_params *params,
                                            float *destination, size_t destination_values,
                                            float dark_floor, bool guarded) {
    if (!renderer) return IM_STATUS_INVALID_ARGUMENT;
    const auto fail = [renderer](im_status status, const char *message) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = message;
        return status;
    };
    if (!source || !transmission || !airlight_rgb || !params || !destination ||
        !width || !height || width > 65535 || height > 65535 || !valid_dehaze(*params) ||
        !std::isfinite(dark_floor) || dark_floor < 0.0f || dark_floor > 2.0f) {
        return fail(IM_STATUS_INVALID_ARGUMENT, "Physical float dehaze arguments or parameters are invalid");
    }

    const uint64_t pixels64 = static_cast<uint64_t>(width) * height;
    if (pixels64 > (1ull << 29) / 3 ||
        pixels64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max() / 3) ||
        source_values != pixels64 * 3 || transmission_count != pixels64 ||
        destination_values < pixels64 * 3) {
        return fail(IM_STATUS_INVALID_ARGUMENT, "Physical float dehaze sample counts are invalid");
    }
    const size_t pixels = static_cast<size_t>(pixels64);
    const size_t values = pixels * 3;
    for (size_t channel = 0; channel < 3; ++channel) {
        if (!std::isfinite(airlight_rgb[channel]) || airlight_rgb[channel] < 0.0f ||
            airlight_rgb[channel] > 1.0f) {
            return fail(IM_STATUS_INVALID_ARGUMENT, "Physical float airlight must be finite and within [0, 1]");
        }
    }
    for (size_t value = 0; value < values; ++value) {
        if (!std::isfinite(source[value]) || source[value] < 0.0f || source[value] > 1.0f) {
            return fail(IM_STATUS_INVALID_ARGUMENT, "Physical float source must be finite and within [0, 1]");
        }
    }
    for (size_t pixel = 0; pixel < pixels; ++pixel) {
        if (!std::isfinite(transmission[pixel]) || transmission[pixel] < 0.0f ||
            transmission[pixel] > 1.0f) {
            return fail(IM_STATUS_INVALID_ARGUMENT,
                        "Physical float transmission must be finite and within [0, 1]");
        }
    }

    if (params->strength <= 1e-6f) {
        std::memmove(destination, source, values * sizeof(float));
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error.clear();
        return IM_STATUS_OK;
    }

    try {
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        const bool completed = guarded
            ? renderer->backend->render_physical_guarded_float(width, height, source, transmission,
                airlight_rgb, *params, dark_floor, destination, destination_values, error)
            : renderer->backend->render_physical_float(width, height, source, transmission,
                airlight_rgb, *params, destination, destination_values, error);
        if (!completed) {
            renderer->error = error.empty() ? "Physical float dehaze is unavailable on this GPU backend"
                                            : std::move(error);
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(IM_STATUS_RUNTIME_ERROR, "Insufficient memory while rendering physical float dehaze");
    } catch (const std::exception &exception) {
        return fail(IM_STATUS_RUNTIME_ERROR, exception.what());
    } catch (...) {
        return fail(IM_STATUS_RUNTIME_ERROR, "Unknown error while rendering physical float dehaze");
    }
}

im_status im_renderer_create(im_backend_kind backend, im_renderer **out_renderer) {
    if (!out_renderer) return IM_STATUS_INVALID_ARGUMENT;
    *out_renderer = nullptr;
    try {
        std::string error;
        auto selected = imprint::create_backend(backend, error);
        if (!selected) {
            g_create_error = std::move(error);
            return g_create_error.rfind("Could not", 0) == 0 || g_create_error.rfind("CUDA initialization", 0) == 0
                       ? IM_STATUS_RUNTIME_ERROR
                       : IM_STATUS_BACKEND_UNAVAILABLE;
        }
        auto instance = std::make_unique<im_renderer>();
        instance->backend = std::move(selected);
        instance->curve = identity_curve();
        instance->filter = {0, 0, 0, 1, 1, 0, 0, 0};
        if (!instance->backend->set_filter(instance->filter, instance->curve, instance->lut, 0, error)) {
            g_create_error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        *out_renderer = instance.release();
        g_create_error.clear();
        return IM_STATUS_OK;
    } catch (const std::exception &exception) {
        g_create_error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        g_create_error = "Unknown error creating native renderer";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

void im_renderer_destroy(im_renderer *renderer) { delete renderer; }

im_status im_renderer_create_shared_buffer(im_renderer *renderer, size_t byte_count,
                                           im_shared_buffer **out_buffer) {
    if (!out_buffer) return IM_STATUS_INVALID_ARGUMENT;
    *out_buffer = nullptr;
    if (!renderer || byte_count == 0 || static_cast<uint64_t>(byte_count) > kMaxSharedBufferBytes) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_shared_buffers()) {
            renderer->error = "Metal shared buffers are unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::shared_ptr<imprint::SharedBuffer> owner;
        std::string error;
        if (!renderer->backend->create_shared_buffer(byte_count, owner, error) || !owner ||
            owner->size() != byte_count || !owner->data()) {
            renderer->error = error.empty() ? "Could not allocate a Metal shared buffer" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        auto handle = std::make_unique<im_shared_buffer>();
        handle->owner = std::move(owner);
        *out_buffer = handle.release();
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while creating a shared buffer";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while creating a shared buffer";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

void *im_shared_buffer_data(const im_shared_buffer *buffer) {
    return buffer && buffer->owner ? buffer->owner->data() : nullptr;
}

size_t im_shared_buffer_size(const im_shared_buffer *buffer) {
    return buffer && buffer->owner ? buffer->owner->size() : 0;
}

void im_shared_buffer_destroy(im_shared_buffer *buffer) { delete buffer; }

const char *im_renderer_last_error(const im_renderer *renderer) {
    if (!renderer) return g_create_error.empty() ? "Renderer handle is null" : g_create_error.c_str();
    return renderer->error.c_str();
}

const char *im_renderer_backend_name(const im_renderer *renderer) {
    return renderer && renderer->backend ? renderer->backend->name() : "unavailable";
}

im_status im_renderer_render_physical_float(im_renderer *renderer,
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *transmission, size_t transmission_count, const float *airlight_rgb,
    const im_dehaze_params *params, float *destination, size_t destination_values) {
    return render_physical_float_impl(renderer, width, height, source, source_values,
        transmission, transmission_count, airlight_rgb, params, destination,
        destination_values, 0.0f, false);
}

im_status im_renderer_render_physical_guarded_float(im_renderer *renderer,
    uint32_t width, uint32_t height, const float *source, size_t source_values,
    const float *transmission, size_t transmission_count, const float *airlight_rgb,
    const im_dehaze_params *params, float dark_floor, float *destination,
    size_t destination_values) {
    return render_physical_float_impl(renderer, width, height, source, source_values,
        transmission, transmission_count, airlight_rgb, params, destination,
        destination_values, dark_floor, true);
}

int im_renderer_supports_physical_guarded_float(const im_renderer *renderer) {
    return renderer && renderer->backend && renderer->backend->supports_physical_guarded_float() ? 1 : 0;
}

int im_renderer_supports_physical_float(const im_renderer *renderer) {
    return renderer && renderer->backend && renderer->backend->supports_physical_float() ? 1 : 0;
}

im_status im_renderer_warp_rectilinear_rgb16(im_renderer *renderer,
    uint32_t width, uint32_t height, const uint16_t *source, size_t source_values,
    const float *constants, size_t constant_values,
    uint16_t *destination, size_t destination_values) {
    if (!renderer) return IM_STATUS_INVALID_ARGUMENT;
    const auto fail = [renderer](im_status status, const char *message) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = message;
        return status;
    };
    size_t pixels = 0;
    float radius = 0.0f;
    if (!imprint::validate_warp_rectilinear_rgb16(width, height, source, source_values,
                                                  constants, constant_values, destination,
                                                  destination_values, &pixels, &radius)) {
        return fail(IM_STATUS_INVALID_ARGUMENT,
                    "WarpRectilinear buffers, dimensions, or constants are invalid");
    }
    (void)pixels;
    (void)radius;

    try {
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_warp_rectilinear()) {
            renderer->error = "WarpRectilinear is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        if (!renderer->backend->warp_rectilinear_rgb16(width, height, source, source_values,
                                                       constants, constant_values,
                                                       destination, destination_values, error)) {
            renderer->error = error.empty() ? "GPU WarpRectilinear failed" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        return fail(IM_STATUS_RUNTIME_ERROR, "Insufficient memory while running GPU WarpRectilinear");
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        return fail(IM_STATUS_RUNTIME_ERROR, "Unknown error while running GPU WarpRectilinear");
    }
}

int im_renderer_supports_warp_rectilinear(const im_renderer *renderer) {
    return renderer && renderer->backend && renderer->backend->supports_warp_rectilinear() ? 1 : 0;
}

int im_renderer_supports_camera_profile_render(const im_renderer *renderer) {
    return renderer && renderer->backend && renderer->backend->supports_camera_profile_render() ? 1 : 0;
}

im_status im_renderer_set_camera_profile_source(im_renderer *renderer,
                                                uint32_t width, uint32_t height,
                                                const uint16_t *camera_rgb,
                                                size_t source_values) {
    if (!renderer || !camera_rgb || !width || !height || width > 65535 || height > 65535) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels > (1ull << 29) / 3 || source_values != pixels * 3) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Camera profile source sample count does not match its dimensions";
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_camera_profile_render()) {
            renderer->error = "Camera profile rendering is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::string error;
        if (!renderer->backend->set_camera_profile_source(width, height, camera_rgb,
                                                          source_values, error)) {
            renderer->has_camera_profile_source = false;
            renderer->camera_profile_source_width = renderer->camera_profile_source_height = 0;
            renderer->camera_profile_source_shared.reset();
            renderer->error = error.empty() ? "Could not upload camera profile source" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->has_camera_profile_source = true;
        renderer->camera_profile_source_width = width;
        renderer->camera_profile_source_height = height;
        renderer->camera_profile_source_shared.reset();
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while uploading camera profile source";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while uploading camera profile source";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_camera_profile(im_renderer *renderer,
                                            const float *constants22, size_t constants_count,
                                            const float *look_table, size_t look_values,
                                            uint32_t hue_count, uint32_t saturation_count,
                                            uint32_t value_count, uint32_t look_encoding,
                                            const float *tone_curve, size_t tone_values,
                                            uint8_t *destination, size_t destination_values) {
    if (!renderer || !destination ||
        !valid_camera_profile_config(constants22, constants_count, look_table, look_values,
                                     hue_count, saturation_count, value_count, look_encoding,
                                     tone_curve, tone_values)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    size_t constants_bytes = 0, look_bytes = 0, tone_bytes = 0;
    if (!camera_profile_byte_length<float>(constants_count, &constants_bytes) ||
        !camera_profile_byte_length<float>(look_values, &look_bytes) ||
        !camera_profile_byte_length<float>(tone_values, &tone_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    size_t destination_bytes = destination_values;
    if (!camera_profile_byte_length<float>(constants_count, &constants_bytes) ||
        !camera_profile_byte_length<float>(look_values, &look_bytes) ||
        !camera_profile_byte_length<float>(tone_values, &tone_bytes) ||
        camera_profile_ranges_overlap(destination, destination_bytes, constants22, constants_bytes) ||
        camera_profile_ranges_overlap(destination, destination_bytes, look_table, look_bytes) ||
        camera_profile_ranges_overlap(destination, destination_bytes, tone_curve, tone_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->has_camera_profile_source) {
            renderer->error = "No camera profile source has been uploaded";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const uint64_t expected64 = static_cast<uint64_t>(renderer->camera_profile_source_width) *
                                    renderer->camera_profile_source_height * 3;
        if (expected64 > (1ull << 29) || destination_values != expected64) {
            renderer->error = "Camera profile destination sample count does not match the source";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (!renderer->backend->supports_camera_profile_render()) {
            renderer->error = "Camera profile rendering is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::string error;
        if (!renderer->backend->render_camera_profile(constants22, look_table, look_values,
                                                       hue_count, saturation_count, value_count,
                                                       look_encoding, tone_curve, tone_values,
                                                       destination, destination_values, error)) {
            renderer->error = error.empty() ? "Camera profile GPU render failed" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while rendering camera profile";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while rendering camera profile";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_clear_camera_profile_source(im_renderer *renderer) {
    if (!renderer) return IM_STATUS_INVALID_ARGUMENT;
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_camera_profile_render()) {
            renderer->error = "Camera profile rendering is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::string error;
        if (!renderer->backend->clear_camera_profile_source(error)) {
            renderer->error = error.empty() ? "Could not clear camera profile GPU resources" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->has_camera_profile_source = false;
        renderer->camera_profile_source_width = renderer->camera_profile_source_height = 0;
        renderer->camera_profile_source_shared.reset();
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while clearing camera profile GPU resources";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_set_camera_profile_source_shared(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const im_shared_buffer *source) {
    if (!renderer || !source || !source->owner || !width || !height ||
        width > 65535 || height > 65535) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    size_t source_bytes = 0;
    if (pixels > (1ull << 29) / 3 ||
        !camera_profile_byte_length<uint16_t>(static_cast<size_t>(pixels * 3), &source_bytes) ||
        source->owner->size() != source_bytes) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_shared_buffers() ||
            !renderer->backend->supports_camera_profile_render()) {
            renderer->error = "Shared camera profile sources are unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        if (!renderer->backend->is_shared_buffer_compatible(source->owner.get())) {
            renderer->error = "Shared camera profile source belongs to another GPU device";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        std::string error;
        if (!renderer->backend->set_camera_profile_source_shared(width, height,
                                                                 source->owner, error)) {
            renderer->has_camera_profile_source = false;
            renderer->camera_profile_source_width = renderer->camera_profile_source_height = 0;
            renderer->camera_profile_source_shared.reset();
            renderer->error = error.empty() ? "Could not retain shared camera profile source" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->has_camera_profile_source = true;
        renderer->camera_profile_source_width = width;
        renderer->camera_profile_source_height = height;
        renderer->camera_profile_source_shared = source->owner;
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while retaining shared camera profile source";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while retaining shared camera profile source";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_camera_profile_shared(
    im_renderer *renderer, const float *constants22, size_t constants_count,
    const float *look_table, size_t look_values,
    uint32_t hue_count, uint32_t saturation_count, uint32_t value_count,
    uint32_t look_encoding, const float *tone_curve, size_t tone_values,
    im_shared_buffer *destination) {
    if (!renderer || !destination || !destination->owner ||
        !valid_camera_profile_config(constants22, constants_count, look_table, look_values,
                                     hue_count, saturation_count, value_count, look_encoding,
                                     tone_curve, tone_values)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    size_t constants_bytes = 0, look_bytes = 0, tone_bytes = 0;
    if (!camera_profile_byte_length<float>(constants_count, &constants_bytes) ||
        !camera_profile_byte_length<float>(look_values, &look_bytes) ||
        !camera_profile_byte_length<float>(tone_values, &tone_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->has_camera_profile_source) {
            renderer->error = "No camera profile source has been uploaded";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const uint64_t expected64 = static_cast<uint64_t>(renderer->camera_profile_source_width) *
                                    renderer->camera_profile_source_height * 3;
        if (expected64 > (1ull << 29) || destination->owner->size() != expected64) {
            renderer->error = "Shared camera profile destination size does not match the source";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (!renderer->backend->supports_shared_buffers() ||
            !renderer->backend->supports_camera_profile_render()) {
            renderer->error = "Shared camera profile rendering is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        if (!renderer->backend->is_shared_buffer_compatible(destination->owner.get())) {
            renderer->error = "Shared camera profile destination belongs to another GPU device";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const void *destination_data = destination->owner->data();
        if (camera_profile_ranges_overlap(destination_data, static_cast<size_t>(expected64),
                                          constants22, constants_bytes) ||
            camera_profile_ranges_overlap(destination_data, static_cast<size_t>(expected64),
                                          look_table, look_bytes) ||
            camera_profile_ranges_overlap(destination_data, static_cast<size_t>(expected64),
                                          tone_curve, tone_bytes) ||
            (renderer->camera_profile_source_shared &&
             camera_profile_ranges_overlap(destination_data, static_cast<size_t>(expected64),
                                           renderer->camera_profile_source_shared->data(),
                                           renderer->camera_profile_source_shared->size()))) {
            renderer->error = "Shared camera profile destination overlaps an input";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        std::string error;
        if (!renderer->backend->render_camera_profile_shared(
                constants22, look_table, look_values, hue_count, saturation_count, value_count,
                look_encoding, tone_curve, tone_values, destination->owner.get(), error)) {
            renderer->error = error.empty() ? "Shared camera profile GPU render failed" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while rendering shared camera profile";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while rendering shared camera profile";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

int im_renderer_supports_camera_profile_transfer(const im_renderer *renderer) {
    return renderer && renderer->backend &&
                   renderer->backend->supports_camera_profile_transfer() ? 1 : 0;
}

im_status im_renderer_set_camera_profile_transfer_source(
    im_renderer *renderer, uint32_t width, uint32_t height,
    const uint16_t *camera_rgb, const uint16_t *reference_rgb, size_t source_values) {
    if (!renderer || !camera_rgb || !reference_rgb || !width || !height ||
        width > 65535 || height > 65535) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t pixels = static_cast<uint64_t>(width) * height;
    if (pixels > (1ull << 29) / 3 || source_values != pixels * 3) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Camera profile transfer source count does not match its dimensions";
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->supports_camera_profile_transfer()) {
            renderer->error = "Camera profile enhancement transfer is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::string error;
        if (!renderer->backend->set_camera_profile_transfer_source(
                width, height, camera_rgb, reference_rgb, source_values, error)) {
            renderer->has_camera_profile_transfer_source = false;
            renderer->camera_profile_transfer_width = renderer->camera_profile_transfer_height = 0;
            renderer->error = error.empty() ? "Could not upload camera profile transfer sources"
                                            : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->has_camera_profile_transfer_source = true;
        renderer->camera_profile_transfer_width = width;
        renderer->camera_profile_transfer_height = height;
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->has_camera_profile_transfer_source = false;
        renderer->camera_profile_transfer_width = renderer->camera_profile_transfer_height = 0;
        renderer->error = "Insufficient memory while uploading camera profile transfer sources";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->has_camera_profile_transfer_source = false;
        renderer->camera_profile_transfer_width = renderer->camera_profile_transfer_height = 0;
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->has_camera_profile_transfer_source = false;
        renderer->camera_profile_transfer_width = renderer->camera_profile_transfer_height = 0;
        renderer->error = "Unknown error while uploading camera profile transfer sources";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_camera_profile_transfer(
    im_renderer *renderer, const uint16_t *processed_rgb, size_t processed_values,
    const float *inverse_matrix9, size_t inverse_count,
    uint16_t *destination, size_t destination_values) {
    if (!renderer || !processed_rgb || !inverse_matrix9 || !destination || inverse_count != 9) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < inverse_count; ++index) {
        if (!std::isfinite(inverse_matrix9[index]) || std::abs(inverse_matrix9[index]) > 1.0e6f) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
    }
    size_t matrix_bytes = 0;
    if (!camera_profile_byte_length<float>(inverse_count, &matrix_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->has_camera_profile_transfer_source) {
            renderer->error = "No camera profile transfer sources have been uploaded";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const uint64_t expected64 = static_cast<uint64_t>(renderer->camera_profile_transfer_width) *
                                    renderer->camera_profile_transfer_height * 3;
        if (expected64 > (1ull << 29) || processed_values != expected64 ||
            destination_values != expected64) {
            renderer->error = "Camera profile transfer sample count does not match its sources";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        size_t image_bytes = 0;
        if (!camera_profile_byte_length<uint16_t>(processed_values, &image_bytes) ||
            camera_profile_ranges_overlap(destination, image_bytes, processed_rgb, image_bytes) ||
            camera_profile_ranges_overlap(destination, image_bytes, inverse_matrix9, matrix_bytes)) {
            renderer->error = "Camera profile transfer destination overlaps an input";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (!renderer->backend->supports_camera_profile_transfer()) {
            renderer->error = "Camera profile enhancement transfer is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        std::string error;
        if (!renderer->backend->render_camera_profile_transfer(
                processed_rgb, processed_values, inverse_matrix9, inverse_count,
                destination, destination_values, error)) {
            renderer->error = error.empty() ? "Camera profile GPU transfer failed" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while transferring camera profile enhancement";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while transferring camera profile enhancement";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_camera_profile_transfer_shared(
    im_renderer *renderer, const im_shared_buffer *processed,
    const float *inverse_matrix9, size_t inverse_count,
    im_shared_buffer *destination) {
    if (!renderer || !processed || !processed->owner || !destination || !destination->owner ||
        !inverse_matrix9 || inverse_count != 9) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < inverse_count; ++index) {
        if (!std::isfinite(inverse_matrix9[index]) || std::abs(inverse_matrix9[index]) > 1.0e6f) {
            return IM_STATUS_INVALID_ARGUMENT;
        }
    }
    size_t matrix_bytes = 0;
    if (!camera_profile_byte_length<float>(inverse_count, &matrix_bytes)) {
        return IM_STATUS_INVALID_ARGUMENT;
    }

    try {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->has_camera_profile_transfer_source) {
            renderer->error = "No camera profile transfer sources have been uploaded";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        const uint64_t expected64 = static_cast<uint64_t>(renderer->camera_profile_transfer_width) *
                                    renderer->camera_profile_transfer_height * 3;
        size_t expected_bytes = 0;
        if (expected64 > (1ull << 29) ||
            !camera_profile_byte_length<uint16_t>(static_cast<size_t>(expected64), &expected_bytes) ||
            processed->owner->size() != expected_bytes ||
            destination->owner->size() != expected_bytes) {
            renderer->error = "Shared camera profile transfer buffer size does not match its sources";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (!renderer->backend->supports_camera_profile_transfer()) {
            renderer->error = "Camera profile enhancement transfer is unavailable on this GPU backend";
            return IM_STATUS_BACKEND_UNAVAILABLE;
        }
        if (!renderer->backend->is_shared_buffer_compatible(processed->owner.get()) ||
            !renderer->backend->is_shared_buffer_compatible(destination->owner.get())) {
            renderer->error = "Shared camera profile transfer buffers belong to another GPU device";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        if (camera_profile_ranges_overlap(destination->owner->data(), expected_bytes,
                                          processed->owner->data(), expected_bytes) ||
            camera_profile_ranges_overlap(destination->owner->data(), expected_bytes,
                                          inverse_matrix9, matrix_bytes)) {
            renderer->error = "Shared camera profile transfer destination overlaps an input";
            return IM_STATUS_INVALID_ARGUMENT;
        }
        std::string error;
        if (!renderer->backend->render_camera_profile_transfer_shared(
                processed->owner.get(), inverse_matrix9, inverse_count,
                destination->owner.get(), error)) {
            renderer->error = error.empty() ? "Shared camera profile GPU transfer failed" : std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while transferring shared camera profile enhancement";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    } catch (...) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Unknown error while transferring shared camera profile enhancement";
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_upload_preview_image(im_renderer *renderer, uint32_t width, uint32_t height,
                                           const uint16_t *rgb16, size_t value_count) {
    if (!renderer || !rgb16 || !width || !height || width > 65535 || height > 65535) return IM_STATUS_INVALID_ARGUMENT;
    const uint64_t expected64 = static_cast<uint64_t>(width) * height * 3;
    if (expected64 > (1ull << 29) || value_count != expected64) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "RGB16 value count does not match a supported image size";
        return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::array<imprint::ImageLevel, 3> next;
        next[0].width = width;
        next[0].height = height;
        next[0].pixels.assign(rgb16, rgb16 + value_count);
        next[1] = resize_box(next[0], 2);
        next[2] = resize_box(next[0], 4);
        std::array<imprint::ImageStats, 3> next_stats;
        for (size_t level = 0; level < next.size(); ++level) {
            const size_t pixels = static_cast<size_t>(next[level].width) * next[level].height;
            next_stats[level] = imprint::dehaze_image_stats(next[level].pixels.data(), pixels);
        }
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->set_images(next, error)) {
            renderer->error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->levels = std::move(next);
        renderer->stats = next_stats;
        renderer->output.clear();
        renderer->output_width = renderer->output_height = 0;
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while caching image pyramid";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_upload_filter(im_renderer *renderer, const im_filter_params *params,
                                    const uint16_t *curve_rgb16, size_t curve_value_count,
                                    const uint16_t *lut_rgb16, size_t lut_value_count, uint32_t lut_edge) {
    if (!renderer || !params || !valid_filter(*params)) return IM_STATUS_INVALID_ARGUMENT;
    if ((curve_value_count != 0 && (!curve_rgb16 || curve_value_count != 3 * 256)) ||
        (curve_value_count == 0 && curve_rgb16 != nullptr)) return IM_STATUS_INVALID_ARGUMENT;
    if (lut_value_count == 0) {
        if (lut_rgb16 != nullptr || lut_edge != 0) return IM_STATUS_INVALID_ARGUMENT;
    } else {
        const uint64_t expected = static_cast<uint64_t>(lut_edge) * lut_edge * lut_edge * 3;
        if (!lut_rgb16 || lut_edge < 2 || lut_edge > 65 || expected != lut_value_count) return IM_STATUS_INVALID_ARGUMENT;
    }
    try {
        std::vector<uint16_t> curve;
        if (curve_value_count) curve.assign(curve_rgb16, curve_rgb16 + curve_value_count);
        else curve = identity_curve();
        std::vector<uint16_t> lut;
        if (lut_value_count) lut.assign(lut_rgb16, lut_rgb16 + lut_value_count);
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->set_filter(*params, curve, lut, lut_edge, error)) {
            renderer->error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->filter = *params;
        renderer->curve = std::move(curve);
        renderer->lut = std::move(lut);
        renderer->lut_edge = lut_edge;
        renderer->output.clear();
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while copying filter resources";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render(im_renderer *renderer, im_render_level level,
                             const im_dehaze_params *dehaze, const im_basic_params *basic) {
    if (!renderer || !dehaze || !basic || !valid_dehaze(*dehaze) || !valid_basic(*basic) ||
        level < IM_RENDER_L0 || level > IM_RENDER_L2) return IM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(renderer->mutex);
    if (!renderer->levels[0].width) {
        renderer->error = "No preview image has been uploaded";
        return IM_STATUS_NOT_READY;
    }
    const auto &input = renderer->levels[static_cast<unsigned>(level)];
    const size_t value_count = static_cast<size_t>(input.width) * input.height * 3;
    try {
        std::vector<uint16_t> output(value_count);
        std::string error;
        imprint::ImageStats stats = renderer->stats[static_cast<unsigned>(level)];
        stats.brightness_gain = imprint::dehaze_brightness_gain(input.pixels.data(),
                                                                 static_cast<size_t>(input.width) * input.height,
                                                                 *dehaze, stats);
        if (!renderer->backend->render_cached(static_cast<unsigned>(level), *dehaze, *basic,
                                              stats,
                                              output.data(), output.size(), error)) {
            renderer->error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->output = std::move(output);
        renderer->output_width = input.width;
        renderer->output_height = input.height;
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        renderer->error = "Insufficient memory while allocating render output";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_full(im_renderer *renderer, uint32_t width, uint32_t height,
                                  const uint16_t *rgb16, size_t value_count,
                                  const im_dehaze_params *dehaze, const im_basic_params *basic) {
    if (!renderer || !rgb16 || !width || !height || width > 65535 || height > 65535 ||
        !dehaze || !basic || !valid_dehaze(*dehaze) || !valid_basic(*basic)) return IM_STATUS_INVALID_ARGUMENT;
    const uint64_t expected64 = static_cast<uint64_t>(width) * height * 3;
    if (expected64 > (1ull << 29) || value_count != expected64) return IM_STATUS_INVALID_ARGUMENT;
    try {
        imprint::ImageLevel source;
        source.width = width;
        source.height = height;
        source.pixels.assign(rgb16, rgb16 + value_count);
        imprint::ImageStats stats = imprint::dehaze_image_stats(source.pixels.data(),
                                                                 static_cast<size_t>(width) * height);
        stats.brightness_gain = imprint::dehaze_brightness_gain(source.pixels.data(),
                                                                static_cast<size_t>(width) * height,
                                                                *dehaze, stats);
        std::vector<uint16_t> output(value_count);
        std::string error;
        std::lock_guard<std::mutex> lock(renderer->mutex);
        if (!renderer->backend->render_full(source, stats, *dehaze, *basic,
                                            output.data(), output.size(), error)) {
            renderer->error = std::move(error);
            return IM_STATUS_RUNTIME_ERROR;
        }
        renderer->output = std::move(output);
        renderer->output_width = width;
        renderer->output_height = height;
        renderer->error.clear();
        return IM_STATUS_OK;
    } catch (const std::bad_alloc &) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = "Insufficient memory while rendering full-resolution input";
        return IM_STATUS_RUNTIME_ERROR;
    } catch (const std::exception &exception) {
        std::lock_guard<std::mutex> lock(renderer->mutex);
        renderer->error = exception.what();
        return IM_STATUS_RUNTIME_ERROR;
    }
}

im_status im_renderer_render_ricoh_full(im_renderer *renderer, uint32_t width, uint32_t height,
                                        const uint16_t *rgb16, size_t value_count,
                                        const uint16_t *lut_rgb16, size_t lut_value_count,
                                        uint32_t lut_edge) {
    if (!renderer || !rgb16 || !width || !height || width > 65535 || height > 65535 ||
        !lut_rgb16 || lut_edge < 2 || lut_edge > 65) return IM_STATUS_INVALID_ARGUMENT;
    const uint64_t expected_image_values = static_cast<uint64_t>(width) * height * 3;
    if (expected_image_values > (1ull << 29) || value_count != expected_image_values) {
        return IM_STATUS_INVALID_ARGUMENT;
    }
    const uint64_t expected_lut_values = static_cast<uint64_t>(lut_edge) * lut_edge * lut_edge * 3;
    if (expected_lut_values != lut_value_count) return IM_STATUS_INVALID_ARGUMENT;

    const im_filter_params neutral_lut_filter{0.0f, 1.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.0f};
    const im_status filter_status = im_renderer_upload_filter(
        renderer, &neutral_lut_filter, nullptr, 0, lut_rgb16, lut_value_count, lut_edge);
    if (filter_status != IM_STATUS_OK) return filter_status;

    const im_dehaze_params no_dehaze{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const im_basic_params no_basic{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    return im_renderer_render_full(renderer, width, height, rgb16, value_count, &no_dehaze, &no_basic);
}

im_status im_renderer_get_output_size(const im_renderer *renderer, uint32_t *width,
                                      uint32_t *height, size_t *value_count) {
    if (!renderer || !width || !height || !value_count) return IM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(renderer->mutex);
    if (renderer->output.empty()) return IM_STATUS_NOT_READY;
    *width = renderer->output_width;
    *height = renderer->output_height;
    *value_count = renderer->output.size();
    return IM_STATUS_OK;
}

im_status im_renderer_copy_output(const im_renderer *renderer, uint16_t *destination,
                                  size_t destination_value_count) {
    if (!renderer || !destination) return IM_STATUS_INVALID_ARGUMENT;
    std::lock_guard<std::mutex> lock(renderer->mutex);
    if (renderer->output.empty()) return IM_STATUS_NOT_READY;
    if (destination_value_count < renderer->output.size()) return IM_STATUS_BUFFER_TOO_SMALL;
    std::copy(renderer->output.begin(), renderer->output.end(), destination);
    return IM_STATUS_OK;
}

} // extern "C"
