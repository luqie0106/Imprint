// Research-only ROI composition. The affine values are explicit harness
// inputs; this does not choose an Adobe render level or source matrix.
#include "acr_buffer_format.hpp"
#include "acr_resample_axis.hpp"
#include "acr_resample_coefficients.hpp"
#include "acr_resample_separable.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace {
using namespace imprint::acr;

PreparedResampleRows rows(const ResampleCoefficientTable& table) {
    return {table.coefficients.data(), table.coefficients.size(),
            table.tap_count, table.stride};
}

bool prepare(double translation, double scale, std::int32_t end,
             ResampleCoefficientTable& coefficients, ResampleAxisPlan& plan) {
    ResampleAxisPlan preliminary{};
    if (prepare_ordinary_resample_axis(translation, scale, 0.0, 8, end, 1,
                                      preliminary) != ResampleAxisStatus::ok ||
        preliminary.kind != ResampleAxisKind::resample) return false;
    // Constructor SCVTF, multiplication by 2^-32, then binary64 reciprocal.
    const double q = 1.0 / (static_cast<double>(preliminary.step) * 0x1p-32);
    if (build_resample_coefficients(q, &coefficients) !=
        ResampleCoefficientStatus::ok) return false;
    if (coefficients.tap_count >
        static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
        return false;
    return prepare_ordinary_resample_axis(
        translation, scale, 0.0, 8, end,
        static_cast<std::int32_t>(coefficients.tap_count / 2), plan) ==
        ResampleAxisStatus::ok;
}

void encode(const ResampleAxisPlan& plan, std::uint64_t* output) {
    output[0] = static_cast<std::uint64_t>(plan.kind);
    std::memcpy(output + 1, &plan.origin, 8);
    std::memcpy(output + 2, &plan.step, 8);
    std::memcpy(output + 3, &plan.lower, 8);
    std::memcpy(output + 4, &plan.upper, 8);
    output[5] = static_cast<std::uint32_t>(plan.integer_offset);
}
}  // namespace

// manual8 is null for native-produced geometry, otherwise V then H fields
// {origin,step,lower,upper} supplied by the independent instruction oracle.
extern "C" int acr_test_resample_axis_roi_u16(
    const std::uint16_t* source, std::size_t source_count,
    std::size_t height, std::size_t width, const std::int64_t* manual8,
    float* output, std::size_t output_count, std::uint64_t* plans12) noexcept {
    constexpr std::size_t kHeight = 20, kWidth = 24, kChannels = 3;
    if (!source || !output || !plans12 || height != 96 || width != 96 ||
        source_count != height * width * kChannels ||
        output_count < kHeight * kWidth * kChannels) return 1;
    try {
        std::vector<float> normalized(source_count);
        if (copy_buffer_u16_to_f32_012eac64(
                source, source_count, width * kChannels, height,
                width * kChannels, normalized.data(), normalized.size(),
                width * kChannels) != BufferFormatStatus::ok) return 2;
        ResampleCoefficientTable vertical_coefficients, horizontal_coefficients;
        ResampleAxisPlan vertical{}, horizontal{};
        if (!prepare(16.25 - 0x1p-8, 2.0, 88, vertical_coefficients, vertical) ||
            !prepare(8.125 - 0x1p-8, 1.25, 88, horizontal_coefficients, horizontal))
            return 3;
        std::array<std::uint64_t, 12> prepared{};
        encode(vertical, prepared.data());
        encode(horizontal, prepared.data() + 6);
        PreparedSeparableAxis vertical_axis{
            vertical.origin, vertical.step, vertical.lower, vertical.upper,
            rows(vertical_coefficients)};
        PreparedSeparableAxis horizontal_axis{
            horizontal.origin, horizontal.step, horizontal.lower, horizontal.upper,
            rows(horizontal_coefficients)};
        if (manual8) {
            vertical_axis.origin = manual8[0];
            vertical_axis.step = manual8[1];
            vertical_axis.lower = manual8[2];
            vertical_axis.upper = manual8[3];
            horizontal_axis.origin = manual8[4];
            horizontal_axis.step = manual8[5];
            horizontal_axis.lower = manual8[6];
            horizontal_axis.upper = manual8[7];
        }
        const SourceView view{normalized.data(), normalized.size(), 0, 0,
                              height, width, kChannels};
        const DestinationRect destination{0, 0, kHeight, kWidth};
        std::vector<float> result;
        if (process_resample_separable(view, destination, vertical_axis,
                                      horizontal_axis, result) !=
            ResampleSeparableStatus::ok) return 4;
        std::memcpy(output, result.data(), result.size() * sizeof(float));
        std::memcpy(plans12, prepared.data(), sizeof(prepared));
        return 0;
    } catch (...) {
        return 5;
    }
}
