#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace imprint::acr {

constexpr std::size_t kMaxCpuPipeStages = 100;

// Internal prepared-state subset, not the layout of the original ACR objects.
// bool members intentionally restrict this contract to canonical boolean flags.
struct CpuPipeStageFlags {
    bool cpu_type1 = false;
    bool cpu_type2 = false;
    bool generic = false;
    bool encoding_c = false;
    bool encoding_d = false;
    std::uint32_t generic_count = 0;
};

struct CpuPipeFormatState {
    std::size_t stage_count = 0;
    std::array<CpuPipeStageFlags, kMaxCpuPipeStages> stages{};
    std::array<std::uint32_t, kMaxCpuPipeStages> selected{};
    std::array<std::uint8_t, kMaxCpuPipeStages> encoding{};
};

enum class FirstStagePreference : int {
    unknown = 0,
    type1 = 1,
    type2 = 2,
};

enum class CpuPipeFormatStatus : int {
    ok = 0,
    invalid_argument = 1,
    unsupported_domain = 2,
    unresolved_first_stage = 3,
};

// Recovered CPU selection branches of 00bbf644. Choices must be 0, 1, or 2;
// active selected entries must be 0 (unresolved), 1, or 2. Existing 1/2 entries
// are preserved without inspecting their capability flags. A generic-only
// first stage needs an explicit preference in place of the original virtual
// call. An explicit second_choice can select a type not advertised by a stage:
// this retains the original branch contract, not a capability-enforcing policy.
// No GPU, guide preparation, callbacks, or pixel conversion are performed.
// Failure leaves the entire state unchanged; success writes only unresolved
// active selected entries. Unused array tails and encoding are never written.
CpuPipeFormatStatus select_cpu_buffer_types_00bbf644(
    CpuPipeFormatState &state, std::uint32_t first_choice,
    std::uint32_t second_choice, FirstStagePreference preference) noexcept;

// Recovered reverse/forward encoding passes of 00bbfb30, restricted to resolved
// CPU types 1/2. Zero is invalid_argument; other active types are unsupported.
// The original (type & ~2) == 1 mask also accepts 3, but GPU types 3/4 are outside
// this contract. Failure leaves the entire state unchanged; success overwrites
// only active encoding entries and never touches unused tails or selected.
CpuPipeFormatStatus propagate_cpu_encoding_00bbfb30(
    CpuPipeFormatState &state) noexcept;

}  // namespace imprint::acr
