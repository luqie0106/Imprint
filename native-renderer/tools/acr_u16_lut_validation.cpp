#include "acr_float_lut.hpp"
#include "acr_u16_lut.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>

extern "C" {
int acr_test_u16_lut_build(const float* table, std::size_t table_count,
                           std::uint16_t* out, std::size_t capacity);
int acr_test_u16_lut_apply(const std::uint16_t* table,
                           std::size_t table_count,
                           const std::uint16_t* source,
                           std::size_t source_count,
                           std::uint16_t* destination,
                           std::size_t destination_count,
                           std::size_t width);
}

namespace {

using imprint::acr::FloatLutPrepared;
using imprint::acr::FloatLutStatus;
using imprint::acr::U16LutPrepared;

int failures = 0;

void check(bool condition, const char* label) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", label);
    ++failures;
}

bool identity(double x, const void*, double* out) noexcept {
    *out = x;
    return true;
}

bool polynomial(double x, const void*, double* out) noexcept {
    *out = std::fma(0.37, x * x, 0.2);
    return true;
}

void fill_constant(FloatLutPrepared* lut, float value) {
    for (float& entry : lut->values) entry = value;
}

bool all_u16(const std::uint16_t* values, std::size_t count,
             std::uint16_t expected) {
    for (std::size_t i = 0; i < count; ++i) {
        if (values[i] != expected) return false;
    }
    return true;
}

}  // namespace

int main() {
    alignas(FloatLutPrepared) FloatLutPrepared identity_float{};
    auto status = imprint::acr::build_float_lut_01300490(
        identity, nullptr, &identity_float);
    check(status == FloatLutStatus::ok, "build identity float LUT");

    alignas(U16LutPrepared) U16LutPrepared identity_u16{};
    status = imprint::acr::build_u16_lut_01300600(identity_float, &identity_u16);
    check(status == FloatLutStatus::ok, "build identity uint16 LUT");
    bool identity_matches_domain = true;
    std::size_t identity_mismatches = 0;
    for (std::size_t i = 0; i < imprint::acr::kU16LutEntryCount; ++i) {
        if (identity_u16.values[i] != static_cast<std::uint16_t>(i)) {
            identity_matches_domain = false;
            if (identity_mismatches < 8) {
                std::fprintf(stderr, "identity mismatch at %zu: got %u\n", i,
                             static_cast<unsigned>(identity_u16.values[i]));
            }
            ++identity_mismatches;
        }
    }
    check(identity_matches_domain,
          "identity LUT maps every uint16 domain value to itself");

    alignas(FloatLutPrepared) FloatLutPrepared polynomial_float{};
    status = imprint::acr::build_float_lut_01300490(
        polynomial, nullptr, &polynomial_float);
    check(status == FloatLutStatus::ok, "build polynomial float LUT");
    alignas(U16LutPrepared) U16LutPrepared polynomial_u16{};
    status = imprint::acr::build_u16_lut_01300600(
        polynomial_float, &polynomial_u16);
    check(status == FloatLutStatus::ok &&
              polynomial_u16.values.front() == 13107 &&
              polynomial_u16.values.back() == 37355,
          "polynomial LUT produces expected endpoint quantization");

    alignas(FloatLutPrepared) FloatLutPrepared negative_float{};
    fill_constant(&negative_float, -0.25f);
    alignas(U16LutPrepared) U16LutPrepared negative_u16{};
    status = imprint::acr::build_u16_lut_01300600(negative_float,
                                                 &negative_u16);
    check(status == FloatLutStatus::ok && all_u16(
              negative_u16.values.data(), negative_u16.values.size(), 49153),
          "negative table values wrap to the low 16 bits");

    alignas(U16LutPrepared) U16LutPrepared failed_output{};
    constexpr std::uint16_t kSentinel = 0xa55a;
    for (auto& value : failed_output.values) value = kSentinel;
    alignas(FloatLutPrepared) FloatLutPrepared bad_float{};
    fill_constant(&bad_float, 0.0f);
    bad_float.values[123] = std::numeric_limits<float>::quiet_NaN();
    status = imprint::acr::build_u16_lut_01300600(bad_float, &failed_output);
    check(status == FloatLutStatus::invalid_parameters &&
              all_u16(failed_output.values.data(), failed_output.values.size(),
                      kSentinel),
          "NaN source table is rejected atomically");

    fill_constant(&bad_float, 0.0f);
    bad_float.values[imprint::acr::kFloatLutSegments + 1] = 1.0f;
    status = imprint::acr::build_u16_lut_01300600(bad_float, &failed_output);
    check(status == FloatLutStatus::invalid_parameters &&
              all_u16(failed_output.values.data(), failed_output.values.size(),
                      kSentinel),
          "mismatched guard is rejected atomically");

    fill_constant(&bad_float, 40000.0f);
    status = imprint::acr::build_u16_lut_01300600(bad_float, &failed_output);
    check(status == FloatLutStatus::evaluation_failure &&
              all_u16(failed_output.values.data(), failed_output.values.size(),
                      kSentinel),
          "out-of-int32 result is rejected atomically");

    status = imprint::acr::build_u16_lut_01300600(
        identity_float, reinterpret_cast<U16LutPrepared*>(&identity_float));
    check(status == FloatLutStatus::overlap,
          "builder rejects overlapping input and output spans");

    alignas(U16LutPrepared) U16LutPrepared arbitrary_lut{};
    for (std::size_t i = 0; i < arbitrary_lut.values.size(); ++i) {
        arbitrary_lut.values[i] = static_cast<std::uint16_t>(65535U - i);
    }
    const std::uint16_t samples[] = {0, 65535, 42, 123, 512};
    std::uint16_t padded[] = {9, 9, 9, 9, 0xbeef, 0xcafe};
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, samples, 5, padded, 6, 4);
    check(status == FloatLutStatus::ok && padded[0] == 65535 &&
              padded[1] == 0 && padded[2] == 65493 && padded[3] == 65412 &&
              padded[4] == 0xbeef && padded[5] == 0xcafe,
          "row lookup processes width and leaves padding untouched");

    std::uint16_t in_place[] = {5, 10, 20, 30, 0xbeef};
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, in_place, 5, in_place, 5, 4);
    check(status == FloatLutStatus::ok && in_place[0] == 65530 &&
              in_place[1] == 65525 && in_place[2] == 65515 &&
              in_place[3] == 65505 && in_place[4] == 0xbeef,
          "in-place row lookup leaves padding untouched");

    std::uint16_t partial_alias[] = {1, 2, 3, 4, 5};
    const std::uint16_t partial_before[] = {1, 2, 3, 4, 5};
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, partial_alias, 4, partial_alias + 1, 4, 3);
    check(status == FloatLutStatus::overlap &&
              std::memcmp(partial_alias, partial_before,
                          sizeof(partial_alias)) == 0,
          "partial row overlap is rejected atomically");

    std::uint16_t one_sample = 7;
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, &one_sample, 1, arbitrary_lut.values.data(),
        arbitrary_lut.values.size(), 1);
    check(status == FloatLutStatus::overlap,
          "row lookup rejects destination overlap with its table");

    std::uint16_t overflow_source = 0;
    std::uint16_t overflow_destination = 0x1234;
    const std::size_t overflow_width =
        static_cast<std::size_t>(std::numeric_limits<std::ptrdiff_t>::max()) /
            sizeof(std::uint16_t) +
        1;
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, &overflow_source, overflow_width,
        &overflow_destination, overflow_width, overflow_width);
    check(status == FloatLutStatus::invalid_argument &&
              overflow_destination == 0x1234,
          "row byte-span overflow is rejected before access");

    const auto near_address_end = reinterpret_cast<const std::uint16_t*>(
        std::numeric_limits<std::uintptr_t>::max() - 1U);
    status = imprint::acr::apply_u16_lut_row_0132d484(
        arbitrary_lut, near_address_end, 2, &overflow_destination, 2, 2);
    check(status == FloatLutStatus::invalid_argument &&
              overflow_destination == 0x1234,
          "row address-span wrap is rejected before access");

    alignas(U16LutPrepared) U16LutPrepared bridge_output{};
    int bridge_status = acr_test_u16_lut_build(
        identity_float.values.data(), identity_float.values.size(),
        bridge_output.values.data(), bridge_output.values.size());
    check(bridge_status == static_cast<int>(FloatLutStatus::ok) &&
              bridge_output.values[0] == 0 &&
              bridge_output.values[32768] == 32768 &&
              bridge_output.values[65535] == 65535,
          "bridge builds and copies the fixed uint16 table");

    std::uint16_t bridge_destination[] = {7, 7, 0xbeef};
    bridge_status = acr_test_u16_lut_apply(
        arbitrary_lut.values.data(), arbitrary_lut.values.size(), samples, 5,
        bridge_destination, 3, 2);
    check(bridge_status == static_cast<int>(FloatLutStatus::ok) &&
              bridge_destination[0] == 65535 && bridge_destination[1] == 0 &&
              bridge_destination[2] == 0xbeef,
          "bridge applies lookup and preserves destination padding");

    bridge_status = acr_test_u16_lut_build(
        identity_float.values.data(), identity_float.values.size(),
        reinterpret_cast<std::uint16_t*>(identity_float.values.data()),
        imprint::acr::kU16LutEntryCount);
    check(bridge_status == static_cast<int>(FloatLutStatus::overlap),
          "bridge rejects real input/output alias before copying");

    bridge_status = acr_test_u16_lut_apply(
        arbitrary_lut.values.data(), arbitrary_lut.values.size(), samples, 1,
        arbitrary_lut.values.data(), arbitrary_lut.values.size(), 1);
    check(bridge_status == static_cast<int>(FloatLutStatus::overlap),
          "bridge rejects table/destination alias before copying");

    if (failures != 0) {
        std::fprintf(stderr, "%d uint16 LUT validation(s) failed\n", failures);
        return 1;
    }
    std::puts("uint16 LUT validation passed");
    return 0;
}
