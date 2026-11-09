#include "acr_basic_flag_slots.hpp"

#include <cstddef>

namespace imprint::acr {
namespace {

struct FlagGroup {
    std::size_t first;
    std::size_t last;
    std::size_t header_index;
};

// b5e1f0's supported groups, expressed as indices into bytes P+0x4..P+0x13.
// Every other Basic ID maps to -1000 and does not read an activation byte.
constexpr std::array<FlagGroup, 9> kFlagGroups{{
    {18, 24, 4},    // P+0x8
    {25, 34, 8},    // P+0xc
    {35, 42, 6},    // P+0xa
    {43, 66, 7},    // P+0xb
    {67, 80, 5},    // P+0x9
    {81, 94, 9},    // P+0xd
    {95, 102, 10},  // P+0xe
    {103, 111, 11}, // P+0xf
    {112, 118, 12}, // P+0x10
}};

// Complete b5df18(ID, 2) table, retained for explicit ID provenance. Only
// mapped IDs 18..118 are consumed here: these entries do NOT imply that this
// operator resets any of IDs 0..17 or 119..131 to their table defaults.
constexpr std::array<std::int32_t, 132> kMode2Defaults{{
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       // IDs 0..11
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 25, 50,     // IDs 12..23
    75, 0, 10, 25, 0, 0, 50, 0, 0, 50, 50, 0, // IDs 24..35
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       // IDs 36..47
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       // IDs 48..59
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,       // IDs 60..71
    0, 0, 0, 0, 0, 50, 0, 0, 0, 0, 0, 0,      // IDs 72..83
    0, 50, 0, 30, 70, 0, 40, 60, 0, 0, 0, 0,   // IDs 84..95
    0, 0, 0, 0, 100, 0, 0, 0, 25, 50, 0, 50,   // IDs 96..107
    50, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,      // IDs 108..119
    400, 0, 0, 0, 0, 0, 0, 0, 100, 0, 0, 0,   // IDs 120..131
}};

}  // namespace

void normalize_basic_flag_slots(const BasicFlagSlotsInput& input,
                                BasicFlagSlotsResult& result) noexcept {
    const BasicFlagSlotsInput snapshot = input;
    BasicFlagSlotsResult candidate{};
    candidate.values = snapshot.current;
    for (const auto& group : kFlagGroups) {
        if (snapshot.header[group.header_index] != 0) continue;
        for (std::size_t id = group.first; id <= group.last; ++id) {
            candidate.values[id] = kMode2Defaults[id];
            candidate.reset_mask[id / 32] |= std::uint32_t{1} << (id % 32);
        }
    }
    result = candidate;
}

}  // namespace imprint::acr
