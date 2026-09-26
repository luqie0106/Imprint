#include "backend.hpp"
#include "dehaze_reference.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <mutex>
#include <new>
#include <utility>

namespace {

thread_local std::string g_create_error;

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
    mutable std::mutex mutex;
    mutable std::string error;
};

extern "C" {

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

const char *im_renderer_last_error(const im_renderer *renderer) {
    if (!renderer) return g_create_error.empty() ? "Renderer handle is null" : g_create_error.c_str();
    return renderer->error.c_str();
}

const char *im_renderer_backend_name(const im_renderer *renderer) {
    return renderer && renderer->backend ? renderer->backend->name() : "unavailable";
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
