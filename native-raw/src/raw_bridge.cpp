#include "imprint_raw.h"

#include "ticoraw_decode.hpp"

#include <libraw/libraw.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <mutex>
#include <string>

namespace {

constexpr size_t kBlackLevelEntries = 4102;
constexpr int32_t kOk = 0;
constexpr int32_t kError = 1;

struct BlackLevels {
    unsigned cblack[kBlackLevelEntries]{};
    unsigned black = 0;
};

struct NativeRawHandle {
    std::unique_ptr<LibRaw> decoder;
    libraw_processed_image_t *processed = nullptr;
    libraw_processed_image_t *thumbnail = nullptr;
    imprint_raw_info info{};
    std::string backend;
    std::mutex mutex;

    ~NativeRawHandle() {
        if (processed) LibRaw::dcraw_clear_mem(processed);
        if (thumbnail) LibRaw::dcraw_clear_mem(thumbnail);
    }
};

void set_error(char *buffer, size_t capacity, const std::string &message) {
    if (!buffer || capacity == 0) return;
    std::snprintf(buffer, capacity, "%s", message.c_str());
    buffer[capacity - 1] = '\0';
}

template <size_t N>
void copy_text(char (&destination)[N], const char *source) {
    if (!source) {
        destination[0] = '\0';
        return;
    }
    std::snprintf(destination, N, "%s", source);
    destination[N - 1] = '\0';
}

BlackLevels rawpy_black_levels(LibRaw &raw) {
    /* Keep rawpy's adjust_bl_ semantics: combine the shared black offset and
       cblack[], normalize common channel offsets, then expose four levels. */
    BlackLevels result;
    auto &color = raw.imgdata.rawdata.color;
    std::memcpy(result.cblack, color.cblack, sizeof(result.cblack));
    result.black = color.black;
    const unsigned filters = raw.imgdata.idata.filters;

    if (filters > 1000 && result.cblack[4] && result.cblack[5]
        && (result.cblack[4] + 1) / 2 == 1
        && (result.cblack[5] + 1) / 2 == 1) {
        int colors[4];
        int last_green = -1;
        int green_count = 0;
        for (int index = 0; index < 4; ++index) {
            colors[index] = raw.COLOR(index / 2, index % 2);
            if (colors[index] == 1) {
                ++green_count;
                last_green = index;
            }
        }
        if (green_count > 1 && last_green >= 0) colors[last_green] = 3;
        for (int index = 0; index < 4; ++index) {
            const unsigned row = static_cast<unsigned>(index / 2) % result.cblack[4];
            const unsigned column = static_cast<unsigned>(index % 2) % result.cblack[5];
            result.cblack[colors[index]] += result.cblack[6 + row * result.cblack[5] + column];
        }
        result.cblack[4] = result.cblack[5] = 0;
    } else if (filters <= 1000 && result.cblack[4] == 1 && result.cblack[5] == 1) {
        for (int index = 0; index < 4; ++index) result.cblack[index] += result.cblack[6];
        result.cblack[4] = result.cblack[5] = 0;
    }

    unsigned common = result.cblack[3];
    for (int index = 0; index < 3; ++index) common = std::min(common, result.cblack[index]);
    for (int index = 0; index < 4; ++index) result.cblack[index] -= common;
    result.black += common;

    if (result.cblack[4] && result.cblack[5]) {
        common = result.cblack[6];
        for (unsigned index = 1; index < result.cblack[4] * result.cblack[5]; ++index)
            common = std::min(common, result.cblack[6 + index]);
        unsigned nonzero = 0;
        for (unsigned index = 0; index < result.cblack[4] * result.cblack[5]; ++index) {
            result.cblack[6 + index] -= common;
            nonzero += result.cblack[6 + index] != 0;
        }
        result.black += common;
        if (!nonzero) result.cblack[4] = result.cblack[5] = 0;
    }
    for (int index = 0; index < 4; ++index) result.cblack[index] += result.black;
    return result;
}

bool fill_info(LibRaw &raw, uint32_t compression, imprint_raw_info &info, std::string &error) {
    const auto &source_sizes = raw.imgdata.sizes;
    const auto &crop = source_sizes.raw_inset_crops[0];
    const auto raw_pitch = source_sizes.raw_pitch;
    info = {};
    info.sizes.raw_height = source_sizes.raw_height;
    info.sizes.raw_width = source_sizes.raw_width;
    info.sizes.height = source_sizes.height;
    info.sizes.width = source_sizes.width;
    info.sizes.top_margin = source_sizes.top_margin;
    info.sizes.left_margin = source_sizes.left_margin;
    info.sizes.iheight = source_sizes.iheight;
    info.sizes.iwidth = source_sizes.iwidth;
    info.sizes.pixel_aspect = source_sizes.pixel_aspect;
    info.sizes.flip = source_sizes.flip;
    info.sizes.crop_left_margin = crop.cleft == std::numeric_limits<unsigned short>::max() ? 0 : crop.cleft;
    info.sizes.crop_top_margin = crop.ctop == std::numeric_limits<unsigned short>::max() ? 0 : crop.ctop;
    info.sizes.crop_width = crop.cwidth;
    info.sizes.crop_height = crop.cheight;
    info.sizes.raw_pitch_bytes = raw_pitch ? raw_pitch : source_sizes.raw_width * sizeof(uint16_t);
    info.compression = compression;
    info.filters = raw.imgdata.idata.filters;
    info.num_colors = raw.imgdata.idata.colors;
    std::memcpy(info.color_desc, raw.imgdata.idata.cdesc, sizeof(info.color_desc));

    uint32_t pattern_size = 2;
    if (info.filters == 0) pattern_size = 1;
    else if (info.filters == LIBRAW_XTRANS) pattern_size = 6;
    else if (info.filters == 1) pattern_size = 4;
    else if (info.filters < 1000) {
        error = "unsupported non-Bayer LibRaw filter layout";
        return false;
    } else {
        pattern_size = 4;
    }
    if (pattern_size == 4 && info.filters > 1000) {
        bool repeats = true;
        for (uint32_t row = 0; row < 2; ++row) {
            for (uint32_t column = 0; column < 2; ++column) {
                const int value = raw.COLOR(static_cast<int>(row), static_cast<int>(column));
                repeats &= value == raw.COLOR(static_cast<int>(row), static_cast<int>(column + 2));
                repeats &= value == raw.COLOR(static_cast<int>(row + 2), static_cast<int>(column));
                repeats &= value == raw.COLOR(static_cast<int>(row + 2), static_cast<int>(column + 2));
            }
        }
        if (repeats) pattern_size = 2;
    }
    info.raw_pattern_size = pattern_size;
    if (pattern_size > 6) {
        error = "unsupported RAW color pattern";
        return false;
    }
    const int top = static_cast<int>(source_sizes.top_margin);
    const int left = static_cast<int>(source_sizes.left_margin);
    for (uint32_t row = 0; row < pattern_size; ++row) {
        for (uint32_t column = 0; column < pattern_size; ++column) {
            const int color = raw.COLOR(static_cast<int>(row) - top, static_cast<int>(column) - left);
            if (color < 0 || color > 255) {
                error = "LibRaw returned an invalid CFA color index";
                return false;
            }
            info.raw_pattern[row * pattern_size + column] = static_cast<uint8_t>(color);
        }
    }

    auto &raw_color = raw.imgdata.rawdata.color;
    for (size_t index = 0; index < 4; ++index) {
        info.camera_whitebalance[index] = raw_color.cam_mul[index];
        info.daylight_whitebalance[index] = raw_color.pre_mul[index];
        for (size_t axis = 0; axis < 3; ++axis)
            info.cam_xyz[index * 3 + axis] = raw_color.cam_xyz[index][axis];
    }
    const BlackLevels black = rawpy_black_levels(raw);
    for (size_t index = 0; index < 4; ++index) {
        info.black_level_per_channel[index] = black.cblack[index];
        info.camera_white_level_per_channel[index] = raw_color.linear_max[index];
    }
    info.white_level = raw_color.maximum;
    info.has_camera_white_level_per_channel = 1;
    for (uint32_t value : info.camera_white_level_per_channel)
        info.has_camera_white_level_per_channel &= value > 0;

    copy_text(info.make, raw.imgdata.idata.make);
    copy_text(info.model, raw.imgdata.idata.model);
    copy_text(info.lens_make, raw.imgdata.lens.LensMake);
    copy_text(info.lens_model, raw.imgdata.lens.Lens);
    copy_text(info.artist, raw.imgdata.other.artist);
    info.iso_speed = raw.imgdata.other.iso_speed;
    info.shutter_speed = raw.imgdata.other.shutter;
    info.aperture = raw.imgdata.other.aperture;
    info.focal_length = raw.imgdata.other.focal_len;
    info.timestamp = static_cast<int64_t>(raw.imgdata.other.timestamp);
    info.shot_order = raw.imgdata.other.shot_order;
    info.lens_min_focal = raw.imgdata.lens.MinFocal;
    info.lens_max_focal = raw.imgdata.lens.MaxFocal;

    if (!raw.imgdata.rawdata.raw_image || !info.sizes.width || !info.sizes.height
        || !info.sizes.raw_width || !info.sizes.raw_height) {
        error = "LibRaw did not expose a usable Bayer RAW buffer";
        return false;
    }
    if (info.sizes.left_margin + info.sizes.width > info.sizes.raw_width
        || info.sizes.top_margin + info.sizes.height > info.sizes.raw_height) {
        error = "LibRaw visible image bounds exceed the RAW buffer";
        return false;
    }
    if (info.sizes.raw_pitch_bytes < info.sizes.raw_width * sizeof(uint16_t)
        || info.sizes.raw_pitch_bytes % sizeof(uint16_t) != 0) {
        error = "LibRaw returned an invalid RAW row pitch";
        return false;
    }
    return true;
}

NativeRawHandle *as_handle(void *handle, char *error, size_t error_capacity) {
    auto *native = static_cast<NativeRawHandle *>(handle);
    if (!native) set_error(error, error_capacity, "native RAW handle is null");
    return native;
}

} // namespace

extern "C" int32_t imprint_raw_open(const char *path, void **out_handle,
                                    imprint_raw_info *out_info, char *error,
                                    size_t error_capacity) {
    if (out_handle) *out_handle = nullptr;
    if (!path || !*path || !out_handle || !out_info) {
        set_error(error, error_capacity, "path, handle output, and metadata output are required");
        return kError;
    }
    try {
        auto native = std::make_unique<NativeRawHandle>();
        native->decoder = std::make_unique<LibRaw>();
        int result = native->decoder->open_file(path);
        if (result != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw open_file failed: ")
                      + libraw_strerror(result));
            return kError;
        }
        const uint32_t compression = native->decoder->imgdata.makernotes.nikon.NEFCompression;
        const char *decoder_name = native->decoder->unpack_function_name();
        if (compression != 13 && compression != 14) {
            set_error(error, error_capacity, "NEF Nikon NEFCompression is not 13 or 14");
            return kError;
        }
        if (!decoder_name || std::strcmp(decoder_name, "nikon_he_load_raw()") != 0) {
            set_error(error, error_capacity, std::string("LibRaw selected an unexpected decoder: ")
                      + (decoder_name ? decoder_name : "(null)"));
            return kError;
        }
        result = native->decoder->unpack();
        if (result != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw unpack failed: ")
                      + libraw_strerror(result));
            return kError;
        }
        const char *backend = imprint_tico_last_backend();
        if (!backend || (std::strcmp(backend, "cpu") != 0 && std::strcmp(backend, "metal") != 0)) {
            set_error(error, error_capacity, "TicoRAW decoder did not report a CPU or Metal backend");
            return kError;
        }
        native->backend = backend;
        std::string metadata_error;
        if (!fill_info(*native->decoder, compression, native->info, metadata_error)) {
            set_error(error, error_capacity, metadata_error);
            return kError;
        }
        *out_info = native->info;
        *out_handle = native.release();
        if (error && error_capacity) error[0] = '\0';
        return kOk;
    } catch (const std::exception &exception) {
        set_error(error, error_capacity, std::string("native RAW open failed: ") + exception.what());
    } catch (...) {
        set_error(error, error_capacity, "native RAW open failed with an unknown C++ exception");
    }
    return kError;
}

extern "C" int32_t imprint_raw_probe_nikon_he(const char *path,
                                              int32_t *out_is_nikon_he,
                                              char *error,
                                              size_t error_capacity) {
    if (out_is_nikon_he) *out_is_nikon_he = 0;
    if (!path || !*path || !out_is_nikon_he) {
        set_error(error, error_capacity, "path and Nikon HE probe output are required");
        return kError;
    }
    try {
        LibRaw probe;
        const int result = probe.open_file(path);
        if (result != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw probe open_file failed: ")
                      + libraw_strerror(result));
            return kError;
        }
        const uint32_t compression = probe.imgdata.makernotes.nikon.NEFCompression;
        const char *decoder_name = probe.unpack_function_name();
        *out_is_nikon_he = (compression == 13 || compression == 14) && decoder_name
            && std::strcmp(decoder_name, "nikon_he_load_raw()") == 0;
        if (error && error_capacity) error[0] = '\0';
        return kOk;
    } catch (const std::exception &exception) {
        set_error(error, error_capacity, std::string("Nikon HE probe failed: ") + exception.what());
    } catch (...) {
        set_error(error, error_capacity, "Nikon HE probe failed with an unknown C++ exception");
    }
    return kError;
}

extern "C" void imprint_raw_close(void *handle) {
    delete static_cast<NativeRawHandle *>(handle);
}

extern "C" int32_t imprint_raw_copy_visible(void *handle, uint16_t *destination,
                                            size_t destination_elements, char *error,
                                            size_t error_capacity) {
    auto *native = as_handle(handle, error, error_capacity);
    if (!native || !destination) {
        if (native) set_error(error, error_capacity, "visible CFA destination is null");
        return kError;
    }
    try {
        std::lock_guard<std::mutex> guard(native->mutex);
        const imprint_raw_sizes &sizes = native->info.sizes;
        const size_t needed = static_cast<size_t>(sizes.width) * sizes.height;
        if (destination_elements < needed) {
            set_error(error, error_capacity, "visible CFA destination buffer is too small");
            return kError;
        }
        const auto *raw = native->decoder->imgdata.rawdata.raw_image;
        const size_t pitch_samples = sizes.raw_pitch_bytes / sizeof(uint16_t);
        for (uint32_t row = 0; row < sizes.height; ++row) {
            const uint16_t *source = raw
                + static_cast<size_t>(sizes.top_margin + row) * pitch_samples
                + sizes.left_margin;
            std::memcpy(destination + static_cast<size_t>(row) * sizes.width,
                        source, static_cast<size_t>(sizes.width) * sizeof(uint16_t));
        }
        if (error && error_capacity) error[0] = '\0';
        return kOk;
    } catch (const std::exception &exception) {
        set_error(error, error_capacity, std::string("visible CFA copy failed: ") + exception.what());
    } catch (...) {
        set_error(error, error_capacity, "visible CFA copy failed with an unknown C++ exception");
    }
    return kError;
}

extern "C" int32_t imprint_raw_postprocess(
    void *handle, const imprint_raw_postprocess_params *params,
    const uint8_t **out_data, size_t *out_size, uint32_t *out_width,
    uint32_t *out_height, uint32_t *out_channels, uint32_t *out_bits,
    char *error, size_t error_capacity) {
    auto *native = as_handle(handle, error, error_capacity);
    if (!native || !params || !out_data || !out_size || !out_width || !out_height
        || !out_channels || !out_bits) {
        if (native) set_error(error, error_capacity, "postprocess parameters and output pointers are required");
        return kError;
    }
    try {
        std::lock_guard<std::mutex> guard(native->mutex);
        if (params->output_bps != 8 && params->output_bps != 16) {
            set_error(error, error_capacity, "output_bps must be 8 or 16");
            return kError;
        }
        if ((params->output_color != 0 && params->output_color != 1)
            || !std::isfinite(params->gamma[0]) || !std::isfinite(params->gamma[1])
            || params->gamma[0] <= 0.0 || params->gamma[1] <= 0.0) {
            set_error(error, error_capacity, "invalid output color space or gamma");
            return kError;
        }
        if (params->has_user_flip && (params->user_flip < 0 || params->user_flip > 7)) {
            set_error(error, error_capacity, "user_flip must be between 0 and 7");
            return kError;
        }
        if (params->has_user_wb) {
            for (float value : params->user_wb) {
                if (!std::isfinite(value) || value <= 0.0f) {
                    set_error(error, error_capacity, "user_wb values must be finite and positive");
                    return kError;
                }
            }
        }
        if (native->processed) {
            LibRaw::dcraw_clear_mem(native->processed);
            native->processed = nullptr;
        }
        auto &libraw_params = native->decoder->imgdata.params;
        libraw_params.half_size = params->half_size != 0;
        libraw_params.use_camera_wb = params->use_camera_wb != 0;
        libraw_params.no_auto_bright = params->no_auto_bright != 0;
        libraw_params.output_bps = params->output_bps;
        libraw_params.output_color = params->output_color;
        libraw_params.gamm[0] = params->gamma[0];
        libraw_params.gamm[1] = params->gamma[1];
        for (size_t index = 0; index < 4; ++index) {
            libraw_params.user_mul[index] = params->has_user_wb ? params->user_wb[index] : 0.0f;
        }
        libraw_params.user_flip = params->has_user_flip ? params->user_flip : -1;

        int result = native->decoder->dcraw_process();
        if (result != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw dcraw_process failed: ")
                      + libraw_strerror(result));
            return kError;
        }
        int image_error = LIBRAW_SUCCESS;
        native->processed = native->decoder->dcraw_make_mem_image(&image_error);
        if (!native->processed || image_error != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw memory image failed: ")
                      + libraw_strerror(image_error));
            return kError;
        }
        if (native->processed->type != LIBRAW_IMAGE_BITMAP
            || native->processed->colors != 3
            || native->processed->bits != params->output_bps) {
            set_error(error, error_capacity, "LibRaw returned an unexpected RGB image format");
            return kError;
        }
        *out_data = native->processed->data;
        *out_size = native->processed->data_size;
        *out_width = native->processed->width;
        *out_height = native->processed->height;
        *out_channels = native->processed->colors;
        *out_bits = native->processed->bits;
        if (error && error_capacity) error[0] = '\0';
        return kOk;
    } catch (const std::exception &exception) {
        set_error(error, error_capacity, std::string("RAW postprocess failed: ") + exception.what());
    } catch (...) {
        set_error(error, error_capacity, "RAW postprocess failed with an unknown C++ exception");
    }
    return kError;
}

extern "C" int32_t imprint_raw_extract_thumb(
    void *handle, const uint8_t **out_data, size_t *out_size, int32_t *out_format,
    uint32_t *out_width, uint32_t *out_height, uint32_t *out_channels,
    uint32_t *out_bits, char *error, size_t error_capacity) {
    auto *native = as_handle(handle, error, error_capacity);
    if (!native || !out_data || !out_size || !out_format || !out_width || !out_height
        || !out_channels || !out_bits) {
        if (native) set_error(error, error_capacity, "thumbnail output pointers are required");
        return kError;
    }
    try {
        std::lock_guard<std::mutex> guard(native->mutex);
        if (native->thumbnail) {
            LibRaw::dcraw_clear_mem(native->thumbnail);
            native->thumbnail = nullptr;
        }
        int result = native->decoder->unpack_thumb();
        if (result != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw unpack_thumb failed: ")
                      + libraw_strerror(result));
            return kError;
        }
        int image_error = LIBRAW_SUCCESS;
        native->thumbnail = native->decoder->dcraw_make_mem_thumb(&image_error);
        if (!native->thumbnail || image_error != LIBRAW_SUCCESS) {
            set_error(error, error_capacity, std::string("LibRaw memory thumbnail failed: ")
                      + libraw_strerror(image_error));
            return kError;
        }
        if (native->thumbnail->type == LIBRAW_IMAGE_JPEG) *out_format = 1;
        else if (native->thumbnail->type == LIBRAW_IMAGE_BITMAP) *out_format = 2;
        else {
            set_error(error, error_capacity, "LibRaw returned an unsupported thumbnail format");
            return kError;
        }
        *out_data = native->thumbnail->data;
        *out_size = native->thumbnail->data_size;
        *out_width = native->thumbnail->width;
        *out_height = native->thumbnail->height;
        *out_channels = native->thumbnail->colors;
        *out_bits = native->thumbnail->bits;
        if (error && error_capacity) error[0] = '\0';
        return kOk;
    } catch (const std::exception &exception) {
        set_error(error, error_capacity, std::string("RAW thumbnail extraction failed: ") + exception.what());
    } catch (...) {
        set_error(error, error_capacity, "RAW thumbnail extraction failed with an unknown C++ exception");
    }
    return kError;
}

extern "C" const char *imprint_raw_decoder_backend(const void *handle) {
    const auto *native = static_cast<const NativeRawHandle *>(handle);
    if (!native) return "none";
    if (native->backend == "metal") return "c++/metal";
    if (native->backend == "cpu") return "c++/cpu";
    return "none";
}
