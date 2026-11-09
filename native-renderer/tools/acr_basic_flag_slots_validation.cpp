#include "acr_basic_flag_slots.hpp"
#include "imprint_renderer.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void validate() {
    using namespace imprint::acr;
    BasicFlagSlotsInput input{};
    input.current.fill(-999999);
    input.header.fill(1);
    BasicFlagSlotsResult output{};
    normalize_basic_flag_slots(input, output);
    require(output.values == input.current, "enabled groups changed");
    require(std::all_of(output.reset_mask.begin(), output.reset_mask.end(),
                        [](auto value) { return value == 0; }), "unexpected store bits");
    for (unsigned byte = 0; byte != 256; ++byte) {
        input.header[7] = static_cast<std::uint8_t>(byte);
        normalize_basic_flag_slots(input, output);
        for (unsigned id = 0; id != 132; ++id) {
            const bool reset = byte == 0 && id >= 43 && id <= 66;
            require(output.values[id] == (reset ? 0 : -999999), "HSL group gate mismatch");
            require(((output.reset_mask[id/32] >> (id%32)) & 1u) == reset,
                    "HSL reset mask mismatch");
        }
        require((output.reset_mask[4] & ~15u) == 0, "unused mask bits set");
    }
    input.header.fill(0);
    normalize_basic_flag_slots(input, output);
    for (unsigned id = 0; id != 132; ++id) {
        const bool mapped = id >= 18 && id <= 118;
        require(((output.reset_mask[id/32] >> (id%32)) & 1u) == mapped,
                "mapped range mismatch");
        if (!mapped) require(output.values[id] == -999999, "unmapped ID defaulted");
    }
    for (const auto pair : {std::array<int,2>{22,25}, {23,50}, {24,75}, {26,10},
                           {87,30}, {88,70}, {90,40}, {91,60}, {100,100}, {110,1}})
        require(output.values[pair[0]] == pair[1], "nonzero default mismatch");

    std::array<std::int32_t,132> destination{};
    std::array<std::uint32_t,5> mask{};
    destination.fill(-123456);
    mask.fill(0xa5a5a5a5);
    const auto saved_destination = destination;
    const auto saved_mask = mask;
    require(im_native_acr_basic_flag_slots(input.current.data(),131,input.header.data(),16,
                destination.data(),132,mask.data(),5) == IM_STATUS_INVALID_ARGUMENT,
            "bad count accepted");
    require(destination == saved_destination && mask == saved_mask, "partial failure commit");
    require(im_native_acr_basic_flag_slots(input.current.data(),132,input.header.data(),16,
                input.current.data(),132,mask.data(),5) == IM_STATUS_INVALID_ARGUMENT,
            "input/output overlap accepted");
    require(mask == saved_mask, "overlap changed mask");
    require(im_native_acr_basic_flag_slots(input.current.data(),132,input.header.data(),16,
                destination.data(),132,mask.data(),5) == IM_STATUS_OK, "formal ABI failed");
    require(destination == output.values && mask == output.reset_mask, "ABI/internal mismatch");
}
}  // namespace

int main() {
    try {
        validate();
        std::cout << "Basic flag prefix: 256 byte gates, 132 IDs, defaults, mask and ABI atomicity passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
