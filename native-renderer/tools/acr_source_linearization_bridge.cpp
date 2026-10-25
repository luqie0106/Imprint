// C ABI thin bridge for the recovered source-linearization stage. Used by the
// independent Python oracle; it performs no pixel math of its own.
#include "acr_source_linearization.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>

using imprint::acr::DestinationFloor;
using imprint::acr::LinearizationInfo;
using imprint::acr::LinearizationInputType;
using imprint::acr::LinearizationOutputType;
using imprint::acr::LinearizationStatus;
using imprint::acr::SourceLinearization;

extern "C" {

struct imprint_acr_linearization_prepared {
    std::int32_t rect_top;
    std::int32_t rect_left;
    std::int32_t rect_bottom;
    std::int32_t rect_right;
    std::uint32_t black_repeat_rows;
    std::uint32_t black_repeat_cols;
    std::uint32_t plane_count;
    double black_level[8][8][4];
    double white_level[4];
    std::int32_t has_delta_h;
    std::int32_t has_delta_v;
    double* delta_h;
    std::uint32_t delta_h_count;
    double* delta_v;
    std::uint32_t delta_v_count;
    std::int32_t has_linearization_table;
    const std::uint16_t* linearization_table;
    std::uint32_t linearization_table_count;
};

namespace {

bool copy_in(const imprint_acr_linearization_prepared* prepared, LinearizationInfo* info) {
    if (prepared == nullptr || info == nullptr) return false;
    info->rect_top = prepared->rect_top;
    info->rect_left = prepared->rect_left;
    info->rect_bottom = prepared->rect_bottom;
    info->rect_right = prepared->rect_right;
    info->black_repeat_rows = prepared->black_repeat_rows;
    info->black_repeat_cols = prepared->black_repeat_cols;
    info->plane_count = prepared->plane_count;
    std::memcpy(info->black_level, prepared->black_level, sizeof(info->black_level));
    std::memcpy(info->white_level, prepared->white_level, sizeof(info->white_level));
    info->has_black_level_delta_h = prepared->has_delta_h != 0;
    info->has_black_level_delta_v = prepared->has_delta_v != 0;
    info->has_linearization_table = prepared->has_linearization_table != 0;
    if (prepared->delta_h != nullptr && prepared->delta_h_count != 0) {
        info->black_level_delta_h.assign(prepared->delta_h,
                                         prepared->delta_h + prepared->delta_h_count);
    }
    if (prepared->delta_v != nullptr && prepared->delta_v_count != 0) {
        info->black_level_delta_v.assign(prepared->delta_v,
                                         prepared->delta_v + prepared->delta_v_count);
    }
    if (prepared->linearization_table != nullptr &&
        prepared->linearization_table_count != 0) {
        info->linearization_table.assign(
            prepared->linearization_table,
            prepared->linearization_table + prepared->linearization_table_count);
    }
    return true;
}

SourceLinearization* as_stage(void* handle) {
    return static_cast<SourceLinearization*>(handle);
}

}  // namespace

std::int32_t imprint_acr_linearization_quantize(imprint_acr_linearization_prepared* prepared) {
    LinearizationInfo info;
    try {
        if (!copy_in(prepared, &info)) {
            return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
        }
    } catch (...) {
        return static_cast<std::int32_t>(LinearizationStatus::allocation_failure);
    }
    LinearizationStatus status = imprint::acr::quantize_linearization_info(&info);
    if (status != LinearizationStatus::ok) return static_cast<std::int32_t>(status);
    std::memcpy(prepared->black_level, info.black_level, sizeof(info.black_level));
    for (std::uint32_t i = 0; i < prepared->delta_h_count && prepared->delta_h != nullptr; ++i) {
        prepared->delta_h[i] = info.black_level_delta_h[i];
    }
    for (std::uint32_t i = 0; i < prepared->delta_v_count && prepared->delta_v != nullptr; ++i) {
        prepared->delta_v[i] = info.black_level_delta_v[i];
    }
    return 0;
}

std::int32_t imprint_acr_linearization_floor(
    const imprint_acr_linearization_prepared* prepared,
    std::uint32_t plane_count,
    std::int32_t output_is_u16,
    std::int32_t source_gate,
    std::int32_t skip_recompute,
    std::uint16_t cached_max16,
    std::int32_t source_has_mosaic,
    std::uint16_t* out_max16,
    std::int32_t* out_clamp_flag) {
    if (out_max16 == nullptr || out_clamp_flag == nullptr) return -1;
    LinearizationInfo info;
    try {
        if (!copy_in(prepared, &info)) return -1;
    } catch (...) {
        return static_cast<std::int32_t>(LinearizationStatus::allocation_failure);
    }
    DestinationFloor floor = imprint::acr::compute_destination_floor(
        info, plane_count,
        output_is_u16 ? LinearizationOutputType::u16 : LinearizationOutputType::f32,
        source_gate != 0, skip_recompute != 0, cached_max16, source_has_mosaic != 0);
    *out_max16 = floor.max16;
    *out_clamp_flag = floor.clamp_flag ? 1 : 0;
    return static_cast<std::int32_t>(floor.status);
}

void* imprint_acr_linearization_create(void) {
    try {
        return new (std::nothrow) SourceLinearization();
    } catch (...) {
        return nullptr;
    }
}

void imprint_acr_linearization_destroy(void* handle) { delete as_stage(handle); }

std::int32_t imprint_acr_linearization_build(
    void* handle,
    const imprint_acr_linearization_prepared* prepared,
    std::uint32_t plane,
    std::uint32_t input_type,
    std::uint32_t output_type,
    std::uint16_t max16,
    std::int32_t clamp_flag) {
    if (handle == nullptr) return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
    LinearizationInfo info;
    try {
        if (!copy_in(prepared, &info)) {
            return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
        }
    } catch (...) {
        return static_cast<std::int32_t>(LinearizationStatus::allocation_failure);
    }
    return static_cast<std::int32_t>(as_stage(handle)->build(
        info, plane, static_cast<LinearizationInputType>(input_type),
        static_cast<LinearizationOutputType>(output_type), max16, clamp_flag != 0));
}

std::int32_t imprint_acr_linearization_dims(
    void* handle,
    std::uint32_t plane,
    std::uint32_t* out_branch_b,
    std::uint32_t* out_main_u16_count,
    std::uint32_t* out_main_i24_count,
    std::uint32_t* out_black_2d_rows,
    std::uint32_t* out_black_2d_cols,
    std::uint32_t* out_row_lut_count) {
    if (handle == nullptr) return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
    SourceLinearization* stage = as_stage(handle);
    if (!stage->built() || stage->plane() != plane) {
        return static_cast<std::int32_t>(LinearizationStatus::not_built);
    }
    if (out_branch_b) *out_branch_b = stage->branch_b() ? 1u : 0u;
    if (out_main_u16_count) *out_main_u16_count = static_cast<std::uint32_t>(stage->main_u16().size());
    if (out_main_i24_count) *out_main_i24_count = static_cast<std::uint32_t>(stage->main_i24().size());
    if (out_black_2d_rows) *out_black_2d_rows = stage->black_2d_rows();
    if (out_black_2d_cols) *out_black_2d_cols = stage->black_2d_cols();
    if (out_row_lut_count) *out_row_lut_count = stage->row_lut_count();
    return 0;
}

std::int32_t imprint_acr_linearization_copy_tables(
    void* handle,
    std::uint32_t plane,
    std::uint16_t* main_u16,
    std::int32_t* main_i24,
    std::int32_t* black_2d,
    std::int32_t* row_lut) {
    if (handle == nullptr) return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
    SourceLinearization* stage = as_stage(handle);
    if (!stage->built() || stage->plane() != plane) {
        return static_cast<std::int32_t>(LinearizationStatus::not_built);
    }
    if (main_u16 != nullptr) std::memcpy(main_u16, stage->main_u16().data(),
                                         stage->main_u16().size() * sizeof(std::uint16_t));
    if (main_i24 != nullptr) std::memcpy(main_i24, stage->main_i24().data(),
                                         stage->main_i24().size() * sizeof(std::int32_t));
    if (black_2d != nullptr) std::memcpy(black_2d, stage->black_2d().data(),
                                         stage->black_2d().size() * sizeof(std::int32_t));
    if (row_lut != nullptr) std::memcpy(row_lut, stage->row_lut().data(),
                                        stage->row_lut().size() * sizeof(std::int32_t));
    return 0;
}

std::int32_t imprint_acr_linearization_process(
    void* handle,
    std::uint32_t plane,
    const void* input,
    std::size_t input_row_stride_samples,
    void* output,
    std::size_t output_row_stride_samples,
    std::int32_t region_top,
    std::int32_t region_left,
    std::int32_t region_bottom,
    std::int32_t region_right) {
    if (handle == nullptr) return static_cast<std::int32_t>(LinearizationStatus::invalid_argument);
    SourceLinearization* stage = as_stage(handle);
    if (!stage->built() || stage->plane() != plane) {
        return static_cast<std::int32_t>(LinearizationStatus::not_built);
    }
    return static_cast<std::int32_t>(stage->process(
        input, input_row_stride_samples, output, output_row_stride_samples,
        region_top, region_left, region_bottom, region_right));
}

std::int32_t imprint_acr_linearization_status_ok(void) { return 0; }

}  // extern "C"
