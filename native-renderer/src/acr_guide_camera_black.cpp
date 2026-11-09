#include "acr_guide_camera_black.hpp"

#include "acr_buffer_format.hpp"
#include "acr_camera_combined.hpp"
#include "acr_camera_prepare.hpp"
#include "acr_pipe_format.hpp"
#include "acr_postcamera_black.hpp"
#include "acr_tone_kernels.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace imprint::acr {
namespace {

constexpr std::size_t kActiveSampleLimit = 128u * 1024u * 1024u;
constexpr std::size_t kCameraControlCount = 13u;

struct AddressSpan {
    std::uintptr_t begin = 0;
    std::uintptr_t end = 0;
};

bool checked_multiply(std::size_t left, std::size_t right,
                      std::size_t* result) noexcept {
    if (left != 0u && right > std::numeric_limits<std::size_t>::max() / left)
        return false;
    *result = left * right;
    return true;
}

bool checked_add(std::size_t left, std::size_t right,
                 std::size_t* result) noexcept {
    if (right > std::numeric_limits<std::size_t>::max() - left) return false;
    *result = left + right;
    return true;
}

bool make_span(const void* pointer, std::size_t byte_count,
               AddressSpan* span) noexcept {
    const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
    if (byte_count > static_cast<std::size_t>(
                         std::numeric_limits<std::ptrdiff_t>::max()) ||
        byte_count > std::numeric_limits<std::uintptr_t>::max() - begin) {
        return false;
    }
    span->begin = begin;
    span->end = begin + byte_count;
    return true;
}

bool make_sample_span(const void* pointer, std::size_t count,
                      std::size_t sample_size, AddressSpan* span) noexcept {
    std::size_t byte_count = 0u;
    if (!checked_multiply(count, sample_size, &byte_count)) return false;
    return make_span(pointer, byte_count, span);
}

bool overlaps(const AddressSpan& left, const AddressSpan& right) noexcept {
    return left.begin < right.end && right.begin < left.end;
}

GuideCameraBlackStatus map_camera_status(CombinedCameraStatus status) noexcept {
    using Source = CombinedCameraStatus;
    using Target = GuideCameraBlackStatus;
    switch (status) {
        case Source::ok: return Target::ok;
        case Source::invalid_argument: return Target::invalid_argument;
        case Source::invalid_dimensions: return Target::invalid_dimensions;
        case Source::invalid_stride: return Target::invalid_stride;
        case Source::buffer_too_small: return Target::buffer_too_small;
        case Source::unsupported_alignment: return Target::unsupported_alignment;
        case Source::overlap: return Target::overlap;
        case Source::arithmetic_overflow: return Target::arithmetic_overflow;
        case Source::resource_limit: return Target::resource_limit;
        case Source::invalid_controls: return Target::invalid_controls;
        case Source::allocation_failure: return Target::allocation_failure;
        case Source::length_error: return Target::length_error;
        case Source::non_finite_input:
        case Source::non_finite_intermediate:
            return Target::camera_failure;
    }
    return Target::camera_failure;
}

GuideCameraBlackStatus map_buffer_status(BufferFormatStatus status) noexcept {
    using Source = BufferFormatStatus;
    using Target = GuideCameraBlackStatus;
    switch (status) {
        case Source::ok: return Target::ok;
        case Source::invalid_argument: return Target::invalid_argument;
        case Source::invalid_dimensions: return Target::invalid_dimensions;
        case Source::invalid_stride: return Target::invalid_stride;
        case Source::buffer_too_small: return Target::buffer_too_small;
        case Source::unsupported_alignment: return Target::unsupported_alignment;
        case Source::overlap: return Target::overlap;
        case Source::arithmetic_overflow: return Target::arithmetic_overflow;
        case Source::resource_limit: return Target::resource_limit;
        case Source::allocation_failure: return Target::allocation_failure;
        case Source::length_error: return Target::length_error;
    }
    return Target::conversion_failure;
}

GuideCameraBlackStatus map_black_status(PostcameraBlackStatus status) noexcept {
    using Source = PostcameraBlackStatus;
    using Target = GuideCameraBlackStatus;
    switch (status) {
        case Source::ok: return Target::ok;
        case Source::invalid_argument: return Target::invalid_argument;
        case Source::invalid_dimensions: return Target::invalid_dimensions;
        case Source::invalid_stride: return Target::invalid_stride;
        case Source::buffer_too_small: return Target::buffer_too_small;
        case Source::unsupported_alignment: return Target::unsupported_alignment;
        case Source::overlap: return Target::overlap;
        case Source::arithmetic_overflow: return Target::arithmetic_overflow;
        case Source::resource_limit: return Target::resource_limit;
        case Source::allocation_failure: return Target::allocation_failure;
        case Source::length_error: return Target::length_error;
        case Source::invalid_black: return Target::invalid_controls;
        case Source::non_finite_input:
        case Source::out_of_domain:
        case Source::non_finite_intermediate:
            return Target::black_failure;
    }
    return Target::black_failure;
}

bool valid_camera_controls(const GuideCameraBlackControls& controls) noexcept {
    if (controls.input_selector != 1u && controls.input_selector != 2u)
        return false;
    if (controls.camera_flags > 1u) return false;

    const double retained_black = controls.camera[3];
    if (!std::isfinite(retained_black) || retained_black < 0.0 ||
        !(retained_black < 1.0)) {
        return false;
    }

    for (double value : controls.camera) {
        if (!std::isfinite(value) ||
            value > static_cast<double>(std::numeric_limits<float>::max()) ||
            value < -static_cast<double>(std::numeric_limits<float>::max())) {
            return false;
        }
        if (!std::isfinite(static_cast<float>(value))) return false;
    }

    const float a = static_cast<float>(controls.camera[0]);
    const float b_gain = static_cast<float>(controls.camera[1]);
    const float upper = static_cast<float>(controls.camera[2]);
    const float camera_black = static_cast<float>(controls.camera[3]);
    const float minimum_upper = static_cast<float>(32768.0 / 65535.0);
    if (!(a >= b_gain) || !(b_gain >= 1.0f) ||
        !(upper >= minimum_upper) || !(upper <= 1.0f) ||
        !(camera_black >= 0.0f) || !(camera_black < 1.0f)) {
        return false;
    }

    for (std::size_t i = 0; i < 3u; ++i) {
        if (controls.order[i] > 2u) return false;
        for (std::size_t j = 0; j < i; ++j) {
            if (controls.order[i] == controls.order[j]) return false;
        }
    }
    return true;
}

GuideCameraBlackStatus make_pipe_plan(
    const GuideCameraBlackControls& controls, bool has_black,
    GuideCameraBlackPlan* plan) noexcept {
    CpuPipeFormatState state{};
    const bool has_pin = (controls.camera_flags & 1u) != 0u;
    state.stage_count = has_black ? 2u : 1u;
    state.stages[0] = CpuPipeStageFlags{false, true, true, false, true, 3u};
    if (has_black) {
        state.stages[1] = CpuPipeStageFlags{true, true, true, true, true, 3u};
    }
    const std::size_t pin_index = state.stage_count;
    if (has_pin) {
        // 00a9cae0 + 00bd5d70: A/B/generic=1, C=0, D=1, planes=3.
        state.stages[state.stage_count++] =
            CpuPipeStageFlags{true, true, true, false, true, 3u};
    }

    if (select_cpu_buffer_types_00bbf644(
            state, controls.input_selector, 0u, FirstStagePreference::unknown) !=
        CpuPipeFormatStatus::ok) {
        return GuideCameraBlackStatus::camera_failure;
    }

    const std::size_t gray_index = state.stage_count;
    if (gray_index >= kMaxCpuPipeStages)
        return GuideCameraBlackStatus::camera_failure;
    state.stages[gray_index] = CpuPipeStageFlags{false, true, true, false, true, 1u};
    state.selected[gray_index] = 0u;
    ++state.stage_count;

    if (select_cpu_buffer_types_00bbf644(
            state, 2u, 0u, FirstStagePreference::unknown) !=
            CpuPipeFormatStatus::ok ||
        propagate_cpu_encoding_00bbfb30(state) != CpuPipeFormatStatus::ok) {
        return GuideCameraBlackStatus::camera_failure;
    }

    if (state.selected[0] != 2u || state.selected[gray_index] != 2u)
        return GuideCameraBlackStatus::camera_failure;

    GuideCameraBlackPlan candidate{};
    candidate.camera_type = state.selected[0];
    if (has_pin) {
        candidate.pin_type = state.selected[pin_index];
        if ((candidate.pin_type != 1u && candidate.pin_type != 2u) ||
            state.encoding[pin_index] != 0u)
            return GuideCameraBlackStatus::camera_failure;
    }
    if (has_black) {
        candidate.black_type = state.selected[1];
        candidate.black_encoding = state.encoding[1];
        if (candidate.black_type == 1u) {
            if (candidate.black_encoding != 0u)
                return GuideCameraBlackStatus::camera_failure;
            candidate.quantized_before_black = true;
        } else if (candidate.black_type != 2u) {
            return GuideCameraBlackStatus::camera_failure;
        }
    }
    *plan = candidate;
    return GuideCameraBlackStatus::ok;
}

}  // namespace

GuideCameraBlackStatus bind_guide_camera_black_controls_prepared(
    const CameraStagePrepared& camera, const GuideCameraBlackSource& source,
    GuideCameraBlackControls& output) noexcept {
    using Status = GuideCameraBlackStatus;
    if (source.response_limit_u16 < 32768u || source.response_limit_u16 > 65535u ||
        source.retained_black_u16 >= 65535u || source.source_float_flag > 1u ||
        (source.guide_image_type != 3u && source.guide_image_type != 11u)) {
        return Status::invalid_controls;
    }
    GuideCameraBlackControls candidate{};
    for (std::size_t i = 0; i < 3u; ++i) {
        const auto index = camera.sorted_plane_indices[i];
        if (index > 2u || camera.gain_q12[i] < 4096u ||
            camera.gain_float[i] != static_cast<float>(camera.gain_q12[i]) / 4096.0f) {
            return Status::invalid_controls;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (camera.sorted_plane_indices[j] == index)
                return Status::invalid_controls;
        }
        candidate.order[i] = index;
    }
    if (camera.gain_q12[candidate.order[0]] < camera.gain_q12[candidate.order[1]] ||
        camera.gain_q12[candidate.order[1]] < camera.gain_q12[candidate.order[2]]) {
        return Status::invalid_controls;
    }
    candidate.camera[0] = camera.gain_float[candidate.order[0]];
    candidate.camera[1] = camera.gain_float[candidate.order[1]];
    // 00878b20 is FDIV S; do not substitute the CopyArea reciprocal multiply.
    const float upper = static_cast<float>(source.response_limit_u16) / 65535.0f;
    candidate.camera[2] = static_cast<double>(upper);
    // 00dbe738..750 uses UCVTF D and FMUL with bits 0x3ef0001000100010.
    candidate.camera[3] = static_cast<double>(source.retained_black_u16) *
                          0x1.0001000100010p-16;
    for (std::size_t row = 0; row < 3u; ++row) {
        for (std::size_t col = 0; col < 3u; ++col) {
            const std::size_t i = row * 3u + col;
            if (camera.sorted_matrix[i] != camera.matrix[row * 3u + candidate.order[col]])
                return Status::invalid_controls;
            candidate.camera[4u + i] = camera.sorted_matrix[i];
        }
    }
    candidate.camera_flags = source.source_float_flag;
    candidate.input_selector = source.guide_image_type == 11u ? 2u : 1u;
    if (!valid_camera_controls(candidate)) return Status::invalid_controls;
    output = candidate;
    return Status::ok;
}

GuideCameraBlackStatus apply_guide_camera_black_prepared(
    const float* source, std::size_t source_count,
    std::size_t source_row_stride, std::size_t height, std::size_t width,
    const GuideCameraBlackControls& controls,
    GuideCameraBlackResult& output) noexcept {
    using Status = GuideCameraBlackStatus;

    static_assert(sizeof(float) == 4u &&
                      std::numeric_limits<float>::is_iec559 &&
                      std::numeric_limits<float>::digits == 24,
                  "prepared guide requires IEEE-754 binary32 float");
    static_assert(std::is_nothrow_move_assignable<GuideCameraBlackResult>::value,
                  "result commit must not throw");

    if (source == nullptr) return Status::invalid_argument;
    if (height == 0u || width == 0u) return Status::invalid_dimensions;

    std::size_t row_samples = 0u;
    if (!checked_multiply(width, 3u, &row_samples))
        return Status::arithmetic_overflow;
    if (source_row_stride < row_samples) return Status::invalid_stride;

    std::size_t active_samples = 0u;
    std::size_t last_row_offset = 0u;
    std::size_t required_source_samples = 0u;
    if (!checked_multiply(height, row_samples, &active_samples) ||
        !checked_multiply(height - 1u, source_row_stride, &last_row_offset) ||
        !checked_add(last_row_offset, row_samples, &required_source_samples)) {
        return Status::arithmetic_overflow;
    }

    AddressSpan source_span{};
    AddressSpan controls_span{};
    AddressSpan output_span{};
    if (!make_sample_span(source, source_count, sizeof(float), &source_span) ||
        !make_span(&controls, sizeof(controls), &controls_span) ||
        !make_span(&output, sizeof(output), &output_span)) {
        return Status::arithmetic_overflow;
    }

    if (active_samples > kActiveSampleLimit) return Status::resource_limit;
    if (source_count < required_source_samples)
        return Status::buffer_too_small;

    const std::size_t ptrdiff_sample_limit =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
        sizeof(float);
    if (required_source_samples > ptrdiff_sample_limit ||
        active_samples > ptrdiff_sample_limit) {
        return Status::arithmetic_overflow;
    }

    if (reinterpret_cast<std::uintptr_t>(source) % alignof(float) != 0u)
        return Status::unsupported_alignment;

    if (overlaps(source_span, output_span) ||
        overlaps(controls_span, output_span) ||
        overlaps(source_span, controls_span)) {
        return Status::overlap;
    }

    if (!valid_camera_controls(controls)) return Status::invalid_controls;

    const bool has_black = controls.camera[3] > 0.0;
    GuideCameraBlackPlan plan{};
    const Status pipe_status = make_pipe_plan(controls, has_black, &plan);
    if (pipe_status != Status::ok) return pipe_status;

    std::vector<float> camera_rgb;
    std::vector<float> black_rgb;
    std::vector<float> result_rgb;
    std::vector<std::uint16_t> integer_rgb;
    if (active_samples > camera_rgb.max_size() ||
        active_samples > black_rgb.max_size() ||
        active_samples > result_rgb.max_size() ||
        active_samples > integer_rgb.max_size()) {
        return Status::length_error;
    }

    try {
        camera_rgb.resize(active_samples);
        const CombinedCameraStatus camera_status =
            apply_camera_combined_prepared_f32_to_buffer(
                source, source_count, source_row_stride, height, width,
                controls.camera.data(), kCameraControlCount, controls.order.data(),
                controls.camera_flags, camera_rgb.data(), active_samples,
                row_samples);
        const Status mapped_camera = map_camera_status(camera_status);
        if (mapped_camera != Status::ok) return mapped_camera;

        // Camera range=4 marks overrange. 00bc4b44 pins a float buffer
        // before a range=0 consumer (black LUT, or pin when b==0). The LUT
        // itself deliberately retains its narrower standalone input contract.
        if ((controls.camera_flags & 1u) != 0u &&
            (plan.black_type == 2u || (!has_black && plan.pin_type == 2u))) {
            for (float& value : camera_rgb) {
                if (!(value > 0.0f)) value = 0.0f;
                else if (value > 1.0f) value = 1.0f;
            }
        }

        if (plan.black_type == 1u || (!has_black && plan.pin_type == 1u)) {
            integer_rgb.resize(active_samples);
            const BufferFormatStatus to_integer =
                copy_buffer_f32_to_u16_012eaeb8(
                    camera_rgb.data(), active_samples, row_samples, height,
                    row_samples, integer_rgb.data(), active_samples, row_samples);
            const Status mapped_conversion = map_buffer_status(to_integer);
            if (mapped_conversion != Status::ok) return mapped_conversion;

            if (has_black) {
                const PostcameraBlackStatus black_status =
                    apply_postcamera_black_prepared_u16(
                        integer_rgb.data(), active_samples, row_samples, height,
                        row_samples, controls.camera[3], integer_rgb.data(),
                        active_samples, row_samples);
                const Status mapped_black = map_black_status(black_status);
                if (mapped_black != Status::ok) return mapped_black;
            }

            result_rgb.resize(active_samples);
            const BufferFormatStatus to_float =
                copy_buffer_u16_to_f32_012eac64(
                    integer_rgb.data(), active_samples, row_samples, height,
                    row_samples, result_rgb.data(), active_samples, row_samples);
            const Status mapped_result_conversion = map_buffer_status(to_float);
            if (mapped_result_conversion != Status::ok)
                return mapped_result_conversion;
        } else if (plan.black_type == 2u) {
            black_rgb.resize(active_samples);
            const PostcameraBlackStatus black_status =
                apply_postcamera_black_prepared_f32(
                    camera_rgb.data(), active_samples, row_samples, height,
                    row_samples, controls.camera[3], black_rgb.data(),
                    active_samples, row_samples);
            const Status mapped_black = map_black_status(black_status);
            if (mapped_black != Status::ok) return mapped_black;
            result_rgb = std::move(black_rgb);
        } else if (!has_black) {
            result_rgb = std::move(camera_rgb);
        } else {
            return Status::camera_failure;
        }

        GuideCameraBlackResult candidate{};
        candidate.height = height;
        candidate.width = width;
        candidate.plan = plan;
        candidate.rgb = std::move(result_rgb);
        output = std::move(candidate);
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (const std::length_error&) {
        return Status::length_error;
    }
}

GuideCameraBlackStatus apply_guide_gray_log_prepared(
    const float* source, std::size_t source_count,
    std::size_t source_row_stride, std::size_t height, std::size_t width,
    const GuideGrayLogControls& controls, GuideGrayLogResult& output) noexcept {
    using Status = GuideCameraBlackStatus;
    static_assert(std::is_nothrow_move_assignable<GuideGrayLogResult>::value,
                  "guide result commit must not throw");
    if (source == nullptr) return Status::invalid_argument;
    if (height == 0u || width == 0u || height > 65535u || width > 65535u)
        return Status::invalid_dimensions;
    AddressSpan source_span{}, controls_span{}, output_span{};
    if (!make_sample_span(source, source_count, sizeof(float), &source_span) ||
        !make_span(&controls, sizeof(controls), &controls_span) ||
        !make_span(&output, sizeof(output), &output_span))
        return Status::arithmetic_overflow;
    if (overlaps(source_span, output_span) || overlaps(controls_span, output_span) ||
        overlaps(source_span, controls_span))
        return Status::overlap;
    if (!std::isfinite(controls.ev90) || std::abs(controls.ev90) > 120.0 ||
        !std::isfinite(controls.source_epsilon) || controls.source_epsilon <= 0.0 ||
        controls.source_epsilon > 0x1p-12 ||
        static_cast<float>(controls.source_epsilon) == 0.0f)
        return Status::invalid_controls;
    GuideCameraBlackResult rgb{};
    const auto status = apply_guide_camera_black_prepared(
        source, source_count, source_row_stride, height, width, controls.camera, rgb);
    if (status != Status::ok) return status;
    try {
        auto gray = make_gray_log(rgb.rgb.data(), static_cast<std::uint32_t>(width),
            static_cast<std::uint32_t>(height), controls.ev90, controls.source_epsilon);
        GuideGrayLogResult candidate{};
        candidate.height = height;
        candidate.width = width;
        candidate.camera_plan = rgb.plan;
        candidate.gain_stage_present = controls.ev90 != 0.0;
        candidate.log_gray = std::move(gray.pixels);
        output = std::move(candidate);
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (const std::length_error&) {
        return Status::length_error;
    } catch (const std::invalid_argument&) {
        return Status::gray_log_failure;
    }
}

GuideCameraBlackStatus prepare_guide_source_level(
    const GuideSourceLevelInput& input, GuideSourceLevelPlan& output) noexcept {
    if (input.profile_pair_present)
        return GuideCameraBlackStatus::profile_preparation_required;
    GuideSourceLevelPlan candidate{};
    candidate.initial_level = input.field_ad9 != 0 && input.field_adb != 0 ? 3u : 2u;
    candidate.selected_level = candidate.initial_level;
    while (candidate.selected_level != 0u) {
        candidate.tested_level_mask |= 1u << candidate.selected_level;
        if (input.level_images[candidate.selected_level] != nullptr) break;
        --candidate.selected_level;
    }
    candidate.selected_image = input.level_images[candidate.selected_level];
    candidate.available = candidate.selected_image != nullptr;
    output = candidate;
    return GuideCameraBlackStatus::ok;
}

std::uint32_t select_guide_base_cache_slot(std::uint32_t strength) noexcept {
    return strength < 4u ? strength : 4u;
}

GuideMaskTransfer classify_guide_mask_transfer(std::uint32_t cached_source_level,
                                              std::uint32_t request_level) noexcept {
    if (cached_source_level == request_level) return GuideMaskTransfer::reuse;
    return request_level < cached_source_level ? GuideMaskTransfer::upsample
                                               : GuideMaskTransfer::downsample;
}

GuideCameraBlackStatus apply_guide_local_mask_prepared(
    const float* source, std::size_t source_count, std::size_t source_row_stride,
    const GuideLocalMaskRequest& request, GuideLocalMaskResult& output) noexcept {
    using Status = GuideCameraBlackStatus;
    static_assert(std::is_nothrow_move_assignable<GuideLocalMaskResult>::value,
                  "guide/mask result commit must not throw");
    if (source == nullptr) return Status::invalid_argument;
    AddressSpan source_span{}, request_span{}, output_span{};
    if (!make_sample_span(source,source_count,sizeof(float),&source_span) ||
        !make_span(&request,sizeof(request),&request_span) ||
        !make_span(&output,sizeof(output),&output_span))
        return Status::arithmetic_overflow;
    if (overlaps(source_span,output_span) || overlaps(request_span,output_span) ||
        overlaps(source_span,request_span))
        return Status::overlap;
    if (request.bounds[0] != 0 || request.bounds[1] != 0)
        return Status::unsupported_geometry;
    const auto height = request.bounds[2], width = request.bounds[3];
    if (height <= 0 || width <= 0 || height > 65535 || width > 65535 ||
        (height == 1 && width == 1))
        return Status::invalid_dimensions;
    if (request.boundary_mode != 2u && request.boundary_mode != 4u)
        return Status::unsupported_boundary;
    if (request.sigma_mode < 0 || request.sigma_mode > 2) return Status::invalid_controls;
    // The stricter prefix's active-RGB limit is checked before allocation.
    if (std::uint64_t{static_cast<std::uint32_t>(height)} *
        static_cast<std::uint32_t>(width) > kActiveSampleLimit/3u)
        return Status::resource_limit;
    try {
        GuideLocalMaskResult candidate{};
        candidate.request = request;
        candidate.geometry = prepare_local_tone_geometry(width,height);
        const auto status = apply_guide_gray_log_prepared(source,source_count,source_row_stride,
            height,width,candidate.request.guide,candidate.guide);
        if (status != Status::ok) return status;
        candidate.mask = build_local_tone_mask(candidate.guide.log_gray.data(),width,height,
            candidate.request.guide.source_epsilon,candidate.request.sigma_mode,candidate.request.boundary_mode);
        if (candidate.mask.delta.width != static_cast<std::uint32_t>(width) ||
            candidate.mask.delta.height != static_cast<std::uint32_t>(height) ||
            candidate.mask.statistics_level != candidate.geometry.statistics_level)
            return Status::mask_failure;
        output = std::move(candidate);
        return Status::ok;
    } catch (const std::bad_alloc&) {
        return Status::allocation_failure;
    } catch (const std::length_error&) {
        return Status::length_error;
    } catch (const std::invalid_argument&) {
        return Status::mask_failure;
    }
}

}  // namespace imprint::acr
