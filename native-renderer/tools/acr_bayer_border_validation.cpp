#include "../src/acr_bayer_border.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <vector>

extern "C" int acr_test_bayer_repeat2(
    const std::uint16_t *source,
    std::size_t source_sample_count,
    std::size_t source_row_stride,
    const std::int32_t *source_rect4,
    const std::int32_t *request_rect4,
    std::uint16_t *destination,
    std::size_t destination_sample_count,
    std::size_t destination_row_stride) noexcept;

namespace {

using imprint::acr::BayerBorderStatus;
using imprint::acr::BayerRepeatRect;

int failures = 0;

void check(bool condition, const char *message) {
    if (condition) return;
    std::fprintf(stderr, "FAIL: %s\n", message);
    ++failures;
}

void check_repeat_pattern_and_bridge() {
    constexpr std::size_t kSourceStride = 6;
    constexpr std::size_t kDestinationStride = 10;
    const BayerRepeatRect source_rect{5, 7, 9, 11};
    const BayerRepeatRect request_rect{3, 5, 11, 13};
    std::vector<std::uint16_t> source(4 * kSourceStride, 0xEEEE);
    for (std::size_t y = 0; y < 4; ++y) {
        for (std::size_t x = 0; x < 4; ++x) {
            source[y * kSourceStride + x] =
                static_cast<std::uint16_t>(100 * y + x + 1);
        }
    }
    const std::vector<std::uint16_t> source_before = source;
    std::vector<std::uint16_t> destination(8 * kDestinationStride, 0xABCD);

    const BayerBorderStatus status =
        imprint::acr::copy_bayer_repeat2_u16(
            source.data(), source.size(), kSourceStride, source_rect,
            request_rect, destination.data(), destination.size(),
            kDestinationStride);
    check(status == BayerBorderStatus::ok,
          "repeat2 4x4 source to 8x8 request status");

    const std::size_t axis_map[8] = {0, 1, 0, 1, 2, 3, 2, 3};
    for (std::size_t y = 0; y < 8; ++y) {
        for (std::size_t x = 0; x < 8; ++x) {
            const std::uint16_t expected = static_cast<std::uint16_t>(
                100 * axis_map[y] + axis_map[x] + 1);
            check(destination[y * kDestinationStride + x] == expected,
                  "repeat2 corner/edge/interior sample mapping");
        }
        for (std::size_t x = 8; x < kDestinationStride; ++x) {
            check(destination[y * kDestinationStride + x] == 0xABCD,
                  "destination row padding preserved");
        }
    }
    check(source == source_before, "source samples and row padding preserved");

    // Exercise the ABI bridge with the same non-zero odd-origin geometry.
    const std::int32_t source_values[4] = {5, 7, 9, 11};
    const std::int32_t request_values[4] = {3, 5, 11, 13};
    std::vector<std::uint16_t> bridged(8 * kDestinationStride, 0xABCD);
    const int bridge_status = acr_test_bayer_repeat2(
        source.data(), source.size(), kSourceStride, source_values,
        request_values, bridged.data(), bridged.size(), kDestinationStride);
    check(bridge_status == static_cast<int>(BayerBorderStatus::ok),
          "bridge repeat2 status");
    check(bridged == destination, "bridge output matches CPU operator");
}

void expect_failure(BayerBorderStatus expected,
                    const std::uint16_t *source,
                    std::size_t source_count,
                    std::size_t source_stride,
                    BayerRepeatRect source_rect,
                    BayerRepeatRect request_rect,
                    std::uint16_t *destination,
                    std::size_t destination_count,
                    std::size_t destination_stride,
                    const std::vector<std::uint16_t> &before,
                    const char *message) {
    const BayerBorderStatus status = imprint::acr::copy_bayer_repeat2_u16(
        source, source_count, source_stride, source_rect, request_rect,
        destination, destination_count, destination_stride);
    check(status == expected, message);
    if (destination != nullptr) {
        check(std::equal(before.begin(), before.end(), destination),
              "destination remains byte-identical on failure");
    }
}

void check_failures_are_atomic() {
    const std::vector<std::uint16_t> source(4 * 6, 17);
    const BayerRepeatRect source_rect{0, 0, 4, 4};
    const BayerRepeatRect request_rect{0, 0, 4, 4};

    std::vector<std::uint16_t> destination(4 * 6, 0xCAFE);
    const std::vector<std::uint16_t> destination_before = destination;
    expect_failure(BayerBorderStatus::unsupported_domain,
                   source.data(), source.size(), 6,
                   {0, 0, 4, 1}, request_rect,
                   destination.data(), destination.size(), 6,
                   destination_before, "source width one rejected");

    expect_failure(BayerBorderStatus::unsupported_domain,
                   source.data(), source.size(), 6,
                   {std::numeric_limits<std::int32_t>::min(), 0,
                    std::numeric_limits<std::int32_t>::max(), 4},
                   {0, 0, 2, 2}, destination.data(), destination.size(), 6,
                   destination_before, "INT32_MIN..MAX dimension rejected");

    expect_failure(BayerBorderStatus::invalid_parameters,
                   source.data(), std::numeric_limits<std::size_t>::max(),
                   std::numeric_limits<std::size_t>::max(), source_rect,
                   request_rect, destination.data(), destination.size(), 6,
                   destination_before, "SIZE_MAX source stride rejected");

    expect_failure(BayerBorderStatus::invalid_parameters,
                   source.data(), std::numeric_limits<std::size_t>::max(), 4,
                   source_rect, request_rect, destination.data(),
                   destination.size(), 6, destination_before,
                   "SIZE_MAX source capacity rejected");

    const std::size_t over_ptrdiff_capacity =
        static_cast<std::size_t>(
            std::numeric_limits<std::ptrdiff_t>::max() / 2) +
        1;
    expect_failure(BayerBorderStatus::invalid_parameters,
                   source.data(), over_ptrdiff_capacity, 6, source_rect,
                   request_rect, destination.data(), destination.size(), 6,
                   destination_before,
                   "source capacity above PTRDIFF_MAX bytes rejected");

    expect_failure(BayerBorderStatus::invalid_parameters,
                   source.data(), source.size(), 6, source_rect, request_rect,
                   destination.data(), std::numeric_limits<std::size_t>::max(),
                   4, destination_before,
                   "SIZE_MAX destination capacity rejected");

    const std::uintptr_t near_address_end =
        std::numeric_limits<std::uintptr_t>::max() - 1;
    const auto *wrapping_source =
        reinterpret_cast<const std::uint16_t *>(near_address_end);
    expect_failure(BayerBorderStatus::invalid_parameters,
                   wrapping_source, 2, 2, source_rect, request_rect,
                   destination.data(), destination.size(), 6,
                   destination_before,
                   "source capacity address wrap rejected");

    auto *wrapping_destination = reinterpret_cast<std::uint16_t *>(
        std::numeric_limits<std::uintptr_t>::max() - 1);
    const BayerBorderStatus wrapping_destination_status =
        imprint::acr::copy_bayer_repeat2_u16(
            source.data(), source.size(), 6, source_rect, request_rect,
            wrapping_destination, 2, 2);
    check(wrapping_destination_status == BayerBorderStatus::invalid_parameters,
          "destination capacity address wrap rejected");
    check(destination == destination_before,
          "destination wrap rejection leaves regular buffer unchanged");

    // Declared capacities overlap, but the actual accessed spans are disjoint.
    // The operator's alias rule is based on those accessed spans, including
    // row gaps, rather than on unused capacity beyond the operation.
    std::vector<std::uint16_t> shared(32, 0xD00D);
    shared[0] = 11;
    shared[1] = 12;
    shared[4] = 21;
    shared[5] = 22;
    const BayerRepeatRect small_rect{0, 0, 2, 2};
    const BayerBorderStatus capacity_overlap_status =
        imprint::acr::copy_bayer_repeat2_u16(
            shared.data(), 20, 4, small_rect, small_rect,
            shared.data() + 8, 20, 4);
    check(capacity_overlap_status == BayerBorderStatus::ok,
          "unused overlapping capacities do not widen access-span alias check");
    check(shared[8] == 11 && shared[9] == 12 && shared[12] == 21 &&
              shared[13] == 22,
          "disjoint accessed spans copy successfully");
    check(shared[10] == 0xD00D && shared[11] == 0xD00D,
          "inter-row padding stays unchanged for disjoint accessed spans");

    expect_failure(BayerBorderStatus::invalid_parameters,
                   source.data(), 21, 6, source_rect, request_rect,
                   destination.data(), destination.size(), 6,
                   destination_before, "short source last row rejected");

    std::vector<std::uint16_t> aliased(4 * 4, 0xA55A);
    const std::vector<std::uint16_t> aliased_before = aliased;
    const BayerBorderStatus alias_status =
        imprint::acr::copy_bayer_repeat2_u16(
            aliased.data(), aliased.size(), 4, source_rect, request_rect,
            aliased.data(), aliased.size(), 4);
    check(alias_status == BayerBorderStatus::overlap,
          "source and destination overlap rejected");
    check(aliased == aliased_before, "overlap rejection leaves bytes unchanged");

    // The bridge must reject an output that would overwrite the rect metadata
    // it has just consumed; the array must remain byte-identical.
    const std::int32_t bridge_source_rect[4] = {0, 0, 4, 4};
    std::int32_t bridge_request_rect[4] = {0, 0, 2, 2};
    const std::int32_t request_before[4] = {0, 0, 2, 2};
    const int metadata_alias_status = acr_test_bayer_repeat2(
        source.data(), source.size(), 6, bridge_source_rect,
        bridge_request_rect,
        reinterpret_cast<std::uint16_t *>(bridge_request_rect), 8, 2);
    check(metadata_alias_status == static_cast<int>(BayerBorderStatus::overlap),
          "bridge rejects output overlapping rectangle metadata");
    for (std::size_t index = 0; index < 4; ++index) {
        check(bridge_request_rect[index] == request_before[index],
              "metadata overlap rejection leaves rect bytes unchanged");
    }

    const int null_metadata_status = acr_test_bayer_repeat2(
        source.data(), source.size(), 6, nullptr, request_before,
        destination.data(), destination.size(), 6);
    check(null_metadata_status ==
              static_cast<int>(BayerBorderStatus::invalid_argument),
          "bridge rejects null metadata array");
    check(destination == destination_before,
          "bridge metadata failure leaves destination unchanged");
}

}  // namespace

int main() {
    check_repeat_pattern_and_bridge();
    check_failures_are_atomic();
    if (failures != 0) {
        std::fprintf(stderr, "%d Bayer border validation failure(s)\n", failures);
        return 1;
    }
    std::puts("Bayer border CPU validation passed");
    return 0;
}
