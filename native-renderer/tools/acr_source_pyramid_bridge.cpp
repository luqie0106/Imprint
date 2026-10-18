#include "../src/acr_source_pyramid.hpp"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <vector>

namespace {

bool checked_multiply(std::size_t left,
                      std::size_t right,
                      std::size_t* product) noexcept {
    if (product == nullptr || (right != 0 && left >
                                                   std::numeric_limits<std::size_t>::max() /
                                                       right)) {
        return false;
    }
    *product = left * right;
    return true;
}

bool same_levels(const std::vector<acr_source_pyramid::Level>& left,
                 const std::vector<acr_source_pyramid::Level>& right) noexcept {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (left[index].width != right[index].width ||
            left[index].height != right[index].height ||
            left[index].pixels.size() != right[index].pixels.size()) {
            return false;
        }
        const std::size_t bytes = left[index].pixels.size() * sizeof(float);
        if (bytes != 0 &&
            std::memcmp(left[index].pixels.data(), right[index].pixels.data(), bytes) != 0) {
            return false;
        }
    }
    return true;
}

}  // namespace

extern "C" int acr_test_build_source_pyramid_float32(
    const float* packed_rgb,
    std::size_t image_width,
    std::size_t image_height,
    std::size_t crop_x,
    std::size_t crop_y,
    std::size_t crop_width,
    std::size_t crop_height,
    int nonlinear_enabled,
    int alias_input_from_output_storage,
    std::size_t* output_level_count,
    std::size_t output_dimensions[10],
    float** output_packed_levels,
    std::size_t* output_sample_count,
    int* output_unchanged_on_failure) {
    using acr_source_pyramid::Level;
    using acr_source_pyramid::Status;

    if (output_level_count == nullptr || output_dimensions == nullptr ||
        output_packed_levels == nullptr || output_sample_count == nullptr ||
        output_unchanged_on_failure == nullptr ||
        (nonlinear_enabled != 0 && nonlinear_enabled != 1) ||
        (alias_input_from_output_storage != 0 && alias_input_from_output_storage != 1)) {
        return static_cast<int>(Status::invalid_argument);
    }

    try {
        std::vector<Level> levels;
        const float* input = packed_rgb;
        if (alias_input_from_output_storage != 0) {
            std::size_t image_pixels = 0;
            std::size_t input_samples = 0;
            if (packed_rgb == nullptr || image_width == 0 || image_height == 0 ||
                !checked_multiply(image_width, image_height, &image_pixels) ||
                image_pixels > 100'000'000 ||
                !checked_multiply(image_pixels, 3, &input_samples)) {
                return static_cast<int>(Status::invalid_argument);
            }
            levels.emplace_back();
            levels.back().width = image_width;
            levels.back().height = image_height;
            levels.back().pixels.assign(packed_rgb, packed_rgb + input_samples);
            input = levels.back().pixels.data();
        } else {
            levels.emplace_back();
            levels.back().width = 701;
            levels.back().height = 702;
            levels.back().pixels = {703.0f, 704.0f};
        }

        const std::vector<Level> before = levels;
        const Status status = acr_source_pyramid::build_float_rgb(
            input,
            image_width,
            image_height,
            acr_flare::Crop{crop_x, crop_y, crop_width, crop_height},
            nonlinear_enabled != 0,
            &levels);
        if (status != Status::ok) {
            *output_unchanged_on_failure = same_levels(levels, before) ? 1 : 0;
            return static_cast<int>(status);
        }

        std::size_t total_samples = 0;
        std::size_t dimensions[10] = {};
        for (std::size_t index = 0; index < levels.size(); ++index) {
            std::size_t level_pixels = 0;
            std::size_t expected_samples = 0;
            std::size_t next_total = 0;
            if (index >= 5 ||
                !checked_multiply(levels[index].width, levels[index].height,
                                  &level_pixels) ||
                !checked_multiply(level_pixels, 3, &expected_samples) ||
                expected_samples != levels[index].pixels.size() ||
                !checked_multiply(levels[index].pixels.size(), 1, &next_total) ||
                next_total > std::numeric_limits<std::size_t>::max() - total_samples) {
                return static_cast<int>(Status::size_overflow);
            }
            dimensions[index * 2] = levels[index].width;
            dimensions[index * 2 + 1] = levels[index].height;
            total_samples += next_total;
        }

        std::size_t output_bytes = 0;
        if (!checked_multiply(total_samples, sizeof(float), &output_bytes)) {
            return static_cast<int>(Status::size_overflow);
        }
        float* packed_output = nullptr;
        if (output_bytes != 0) {
            packed_output = static_cast<float*>(std::malloc(output_bytes));
            if (packed_output == nullptr) return static_cast<int>(Status::allocation_failure);
            std::size_t sample_offset = 0;
            for (const Level& level : levels) {
                const std::size_t bytes = level.pixels.size() * sizeof(float);
                std::memcpy(packed_output + sample_offset, level.pixels.data(), bytes);
                sample_offset += level.pixels.size();
            }
        }

        *output_level_count = levels.size();
        for (std::size_t index = 0; index < 10; ++index) {
            output_dimensions[index] = dimensions[index];
        }
        *output_packed_levels = packed_output;
        *output_sample_count = total_samples;
        *output_unchanged_on_failure = 0;
        return static_cast<int>(Status::ok);
    } catch (const std::bad_alloc&) {
        return static_cast<int>(Status::allocation_failure);
    } catch (...) {
        return static_cast<int>(Status::allocation_failure);
    }
}

extern "C" void acr_test_free_source_pyramid_float32(float* pixels) {
    std::free(pixels);
}
