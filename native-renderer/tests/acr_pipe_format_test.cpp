#include "acr_pipe_format.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {

using imprint::acr::CpuPipeFormatState;
using imprint::acr::CpuPipeFormatStatus;
using imprint::acr::CpuPipeStageFlags;
using imprint::acr::FirstStagePreference;
using imprint::acr::kMaxCpuPipeStages;
using imprint::acr::propagate_cpu_encoding_00bbfb30;
using imprint::acr::select_cpu_buffer_types_00bbf644;

int failures = 0;
int checks = 0;

void check(bool condition, const char *message) {
    ++checks;
    if (!condition) {
        ++failures;
        std::cerr << "FAIL: " << message << '\n';
    }
}

using StateBytes = std::array<unsigned char, sizeof(CpuPipeFormatState)>;

StateBytes snapshot(const CpuPipeFormatState &state) {
    StateBytes bytes{};
    std::memcpy(bytes.data(), &state, bytes.size());
    return bytes;
}

void check_unchanged(const CpuPipeFormatState &state, const StateBytes &before,
                     const char *message) {
    check(std::memcmp(&state, before.data(), before.size()) == 0, message);
}

bool same_flags(const CpuPipeStageFlags &a, const CpuPipeStageFlags &b) {
    return a.cpu_type1 == b.cpu_type1 && a.cpu_type2 == b.cpu_type2 &&
           a.generic == b.generic && a.encoding_c == b.encoding_c &&
           a.encoding_d == b.encoding_d && a.generic_count == b.generic_count;
}

CpuPipeFormatState sentinel_state(std::size_t count) {
    CpuPipeFormatState state{};
    state.stage_count = count;
    for (std::size_t i = 0; i < kMaxCpuPipeStages; ++i) {
        state.selected[i] = 1000u + static_cast<std::uint32_t>(i);
        state.encoding[i] = static_cast<std::uint8_t>(100u + i);
        state.stages[i] = {i % 2 == 0, i % 3 == 0, i % 5 == 0,
                           i % 7 == 0, i % 11 == 0,
                           100u + static_cast<std::uint32_t>(i)};
    }
    return state;
}

void check_selection_write_scope(const CpuPipeFormatState &state,
                                 const CpuPipeFormatState &before) {
    check(state.stage_count == before.stage_count,
          "selection preserves stage_count");
    check(state.encoding == before.encoding, "selection preserves encoding");
    for (std::size_t i = 0; i < kMaxCpuPipeStages; ++i) {
        check(same_flags(state.stages[i], before.stages[i]),
              "selection preserves all flags");
        if (i >= state.stage_count || before.selected[i] != 0) {
            check(state.selected[i] == before.selected[i],
                  "selection preserves resolved entries and unused tail");
        }
    }
}

void check_encoding_write_scope(const CpuPipeFormatState &state,
                                const CpuPipeFormatState &before) {
    check(state.stage_count == before.stage_count,
          "encoding preserves stage_count");
    check(state.selected == before.selected, "encoding preserves selection");
    for (std::size_t i = 0; i < kMaxCpuPipeStages; ++i) {
        check(same_flags(state.stages[i], before.stages[i]),
              "encoding preserves all flags");
        if (i >= state.stage_count) {
            check(state.encoding[i] == before.encoding[i],
                  "encoding preserves unused sentinel tail");
        }
    }
}

void test_empty_and_stage_limits() {
    auto empty = sentinel_state(0);
    const auto empty_before = snapshot(empty);
    check(select_cpu_buffer_types_00bbf644(
              empty, 0, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "empty selection succeeds without first-stage preference");
    check_unchanged(empty, empty_before, "empty selection writes nothing");
    check(propagate_cpu_encoding_00bbfb30(empty) == CpuPipeFormatStatus::ok,
          "empty encoding succeeds");
    check_unchanged(empty, empty_before, "empty encoding writes nothing");

    auto full = sentinel_state(kMaxCpuPipeStages);
    for (std::size_t i = 0; i < kMaxCpuPipeStages; ++i) {
        full.selected[i] = 0;
        full.stages[i] = {true, true, false, true, i != 0, 0};
    }
    const auto before_selection = full;
    check(select_cpu_buffer_types_00bbf644(
              full, 0, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "100-stage selection succeeds");
    for (const auto selected : full.selected) {
        check(selected == 1, "100-stage selection fills every active entry");
    }
    check_selection_write_scope(full, before_selection);
    const auto before_encoding = full;
    check(propagate_cpu_encoding_00bbfb30(full) == CpuPipeFormatStatus::ok,
          "100-stage encoding succeeds");
    for (const auto encoding : full.encoding) {
        check(encoding == 1, "100-stage forward encoding reaches final entry");
    }
    check_encoding_write_scope(full, before_encoding);

    for (const auto count : {std::size_t{101},
                             std::numeric_limits<std::size_t>::max()}) {
        auto invalid = sentinel_state(count);
        const auto before = snapshot(invalid);
        check(select_cpu_buffer_types_00bbf644(
                  invalid, 0, 0, FirstStagePreference::unknown) ==
                  CpuPipeFormatStatus::invalid_argument,
              "selection rejects stage count above 100");
        check_unchanged(invalid, before, "invalid count selection is atomic");
        check(propagate_cpu_encoding_00bbfb30(invalid) ==
                  CpuPipeFormatStatus::invalid_argument,
              "encoding rejects stage count above 100");
        check_unchanged(invalid, before, "invalid count encoding is atomic");
    }
}

void test_invalid_selection_arguments() {
    const std::array<std::uint32_t, 3> invalid_values = {
        3, 4, std::numeric_limits<std::uint32_t>::max()};
    for (const auto value : invalid_values) {
        for (const bool invalid_first : {false, true}) {
            auto state = sentinel_state(1);
            state.selected[0] = 0;
            state.stages[0] = {true, true, false, false, false, 0};
            const auto before = snapshot(state);
            check(select_cpu_buffer_types_00bbf644(
                      state, invalid_first ? value : 0,
                      invalid_first ? 0 : value,
                      FirstStagePreference::unknown) ==
                      CpuPipeFormatStatus::invalid_argument,
                  "selection rejects choices outside 0/1/2");
            check_unchanged(state, before, "invalid choice writes nothing");
        }
    }

    auto state = sentinel_state(1);
    state.selected[0] = 0;
    state.stages[0] = {true, false, false, false, false, 0};
    const auto before = snapshot(state);
    check(select_cpu_buffer_types_00bbf644(
              state, 0, 0, static_cast<FirstStagePreference>(99)) ==
              CpuPipeFormatStatus::invalid_argument,
          "selection rejects an invalid preference enumerator");
    check_unchanged(state, before, "invalid preference writes nothing");
}

void test_existing_selection_and_atomic_failure() {
    auto preserved = sentinel_state(3);
    preserved.selected[0] = 1;
    preserved.selected[1] = 2;
    preserved.selected[2] = 0;
    preserved.stages[0] = {};
    preserved.stages[1] = {};
    preserved.stages[2] = {true, false, false, false, false, 0};
    const auto before = preserved;
    check(select_cpu_buffer_types_00bbf644(
              preserved, 2, 2, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "resolved entries do not require capabilities");
    check(preserved.selected[0] == 1 && preserved.selected[1] == 2 &&
              preserved.selected[2] == 1,
          "selection preserves resolved entries while filling an unresolved one");
    check_selection_write_scope(preserved, before);

    for (const auto unsupported :
         {3u, 4u, std::numeric_limits<std::uint32_t>::max()}) {
        auto state = sentinel_state(3);
        state.selected[0] = 0;
        state.stages[0] = {true, false, false, false, false, 0};
        state.selected[1] = 1;
        state.selected[2] = unsupported;
        const auto bytes = snapshot(state);
        check(select_cpu_buffer_types_00bbf644(
                  state, 0, 0, FirstStagePreference::unknown) ==
                  CpuPipeFormatStatus::unsupported_domain,
              "selection rejects unsupported active types");
        check_unchanged(state, bytes,
                        "unsupported active type does not partially fill prefix");
    }

    auto later_failure = sentinel_state(3);
    for (std::size_t i = 0; i < 3; ++i) {
        later_failure.selected[i] = 0;
        later_failure.stages[i] = {true, false, false, false, false, 0};
    }
    later_failure.stages[2] = {};
    const auto bytes = snapshot(later_failure);
    check(select_cpu_buffer_types_00bbf644(
              later_failure, 0, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::unsupported_domain,
          "selection rejects a late stage without any usable capability");
    check_unchanged(later_failure, bytes,
                    "late capability failure does not write valid prefix");
}

void test_generic_threshold_and_first_stage_fallback() {
    for (const auto preference : {FirstStagePreference::unknown,
                                  FirstStagePreference::type1,
                                  FirstStagePreference::type2}) {
        auto state = sentinel_state(1);
        state.selected[0] = 0;
        state.stages[0] = {false, false, true, false, false, 4};
        const auto bytes = snapshot(state);
        const auto before = state;
        const auto status = select_cpu_buffer_types_00bbf644(
            state, 2, 1, preference);
        if (preference == FirstStagePreference::unknown) {
            check(status == CpuPipeFormatStatus::unresolved_first_stage,
                  "generic-only first stage rejects unknown preference");
            check_unchanged(state, bytes,
                            "unresolved first-stage fallback writes nothing");
        } else {
            check(status == CpuPipeFormatStatus::ok,
                  "generic count four permits explicit first-stage fallback");
            check(state.selected[0] ==
                      (preference == FirstStagePreference::type1 ? 1u : 2u),
                  "first-stage preference overrides first/second choices");
            check_selection_write_scope(state, before);
        }
    }

    for (const auto count : {5u, std::numeric_limits<std::uint32_t>::max()}) {
        auto state = sentinel_state(1);
        state.selected[0] = 0;
        state.stages[0] = {false, false, true, false, false, count};
        const auto before = snapshot(state);
        check(select_cpu_buffer_types_00bbf644(
                  state, 0, 0, FirstStagePreference::type1) ==
                  CpuPipeFormatStatus::unsupported_domain,
              "generic-only count at least five is unsupported");
        check_unchanged(state, before, "generic threshold failure is atomic");
    }

    auto resolved_generic = sentinel_state(1);
    resolved_generic.selected[0] = 2;
    resolved_generic.stages[0] = {false, false, true, false, false, 4};
    const auto resolved_before = snapshot(resolved_generic);
    check(select_cpu_buffer_types_00bbf644(
              resolved_generic, 0, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "already resolved generic first stage does not need fallback");
    check_unchanged(resolved_generic, resolved_before,
                    "resolved generic first stage is untouched");

    for (const auto first : {0u, 1u, 2u}) {
        auto state = sentinel_state(2);
        state.selected[0] = 0;
        state.selected[1] = 0;
        state.stages[0] = {false, false, true, false, false, 4};
        state.stages[1] = state.stages[0];
        const auto preference = first < 2 ? FirstStagePreference::type1
                                         : FirstStagePreference::type2;
        const auto before = state;
        check(select_cpu_buffer_types_00bbf644(state, first, 0, preference) ==
                  CpuPipeFormatStatus::ok,
              "generic-only first and later stages select successfully");
        check(state.selected[0] == (first < 2 ? 1u : 2u) &&
                  state.selected[1] == (first < 2 ? 2u : 1u),
              "only the first generic-only stage uses explicit fallback");
        check_selection_write_scope(state, before);
    }
}

void test_selection_branch_table() {
    struct BranchCase {
        CpuPipeStageFlags flags;
        // first choice 0, 1, 2; within each row second choice 0, 1, 2.
        std::array<std::array<std::uint32_t, 3>, 3> expected;
    };
    const std::array<BranchCase, 11> cases = {{
        {{true, false, false, false, false, 0},
         {{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}}},
        {{false, true, false, false, false, 0},
         {{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}}},
        {{true, true, false, false, false, 0},
         {{{1, 1, 1}, {1, 1, 1}, {2, 2, 2}}}},
        {{false, false, true, false, false, 4},
         {{{2, 1, 2}, {2, 1, 2}, {1, 1, 2}}}},
        {{true, false, true, false, false, 4},
         {{{1, 1, 1}, {1, 1, 1}, {1, 1, 2}}}},
        {{false, true, true, false, false, 4},
         {{{2, 1, 2}, {2, 1, 2}, {2, 2, 2}}}},
        {{true, true, true, false, false, 4},
         {{{1, 1, 1}, {1, 1, 1}, {2, 2, 2}}}},
        {{true, false, true, false, false, 5},
         {{{1, 1, 1}, {1, 1, 1}, {1, 1, 1}}}},
        {{false, true, true, false, false, 5},
         {{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}}},
        {{true, true, true, false, false, 5},
         {{{1, 1, 1}, {1, 1, 1}, {2, 2, 2}}}},
        {{false, true, false, false, false, 0xffffffffu},
         {{{2, 2, 2}, {2, 2, 2}, {2, 2, 2}}}},
    }};

    for (const auto &test : cases) {
        for (std::uint32_t first = 0; first <= 2; ++first) {
            for (std::uint32_t second = 0; second <= 2; ++second) {
                // A generic-only first stage is tested separately; every other
                // branch must behave identically at the first and later stages.
                const bool generic_only = test.flags.generic &&
                    !test.flags.cpu_type1 && !test.flags.cpu_type2;
                for (const std::size_t index : {std::size_t{0},
                                                std::size_t{1}}) {
                    if (index == 0 && generic_only) continue;
                    auto state = sentinel_state(index + 1);
                    state.selected[0] = index == 0 ? 0 : 2;
                    state.stages[0] = {};
                    state.selected[index] = 0;
                    state.stages[index] = test.flags;
                    const auto before = state;
                    check(select_cpu_buffer_types_00bbf644(
                              state, first, second,
                              FirstStagePreference::unknown) ==
                              CpuPipeFormatStatus::ok,
                          "selection branch-table case succeeds");
                    check(state.selected[index] == test.expected[first][second],
                          "selection matches independent branch-table values");
                    check_selection_write_scope(state, before);
                }
            }
        }
    }
}

void test_prefix_preserved_when_extending_state() {
    auto state = sentinel_state(2);
    state.selected[0] = 0;
    state.selected[1] = 0;
    state.stages[0] = {true, true, false, false, false, 0};
    state.stages[1] = state.stages[0];
    const auto before_first = state;
    check(select_cpu_buffer_types_00bbf644(
              state, 0, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "first selection fills initial prefix");
    check(state.selected[0] == 1 && state.selected[1] == 1,
          "initial prefix uses first choice zero as type one");
    check_selection_write_scope(state, before_first);

    state.stage_count = 4;
    state.stages[0] = {};
    state.stages[1] = {};
    state.stages[2] = {true, true, false, false, false, 0};
    state.stages[3] = {false, false, true, false, false, 4};
    state.selected[2] = 0;
    state.selected[3] = 0;
    const auto before_second = state;
    check(select_cpu_buffer_types_00bbf644(
              state, 2, 0, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "second selection fills newly added stages");
    check(state.selected[0] == 1 && state.selected[1] == 1 &&
              state.selected[2] == 2 && state.selected[3] == 1,
          "second call preserves prefix and applies new choices only to zeros");
    check_selection_write_scope(state, before_second);
    const auto final_bytes = snapshot(state);
    check(select_cpu_buffer_types_00bbf644(
              state, 1, 2, FirstStagePreference::unknown) ==
              CpuPipeFormatStatus::ok,
          "fully resolved state can be selected repeatedly");
    check_unchanged(state, final_bytes, "resolved repeated selection is a no-op");
}

template <std::size_t N>
void encoding_case(const std::array<std::uint32_t, N> &selected,
                   const std::array<CpuPipeStageFlags, N> &flags,
                   const std::array<std::uint8_t, N> &expected,
                   const char *message) {
    auto state = sentinel_state(N);
    for (std::size_t i = 0; i < N; ++i) {
        state.selected[i] = selected[i];
        state.stages[i] = flags[i];
    }
    const auto before = state;
    check(propagate_cpu_encoding_00bbfb30(state) == CpuPipeFormatStatus::ok,
          message);
    for (std::size_t i = 0; i < N; ++i) {
        check(state.encoding[i] == expected[i], message);
    }
    check_encoding_write_scope(state, before);
    const auto bytes = snapshot(state);
    check(propagate_cpu_encoding_00bbfb30(state) == CpuPipeFormatStatus::ok,
          "repeated encoding succeeds");
    check_unchanged(state, bytes, "encoding propagation is idempotent");
}

void test_encoding_single_stage_flags() {
    for (const auto selected : {1u, 2u}) {
        for (unsigned int bits = 0; bits < 8; ++bits) {
            const bool a = (bits & 1u) != 0;
            const bool c = (bits & 2u) != 0;
            const bool d = (bits & 4u) != 0;
            const std::uint8_t expected = selected == 1 && a && c && !d ? 1 : 0;
            encoding_case<1>(
                {selected}, {{{a, true, true, c, d, 4}}}, {expected},
                "single-stage encoding requires type1, A, C, and not D");
        }
    }
}

void test_encoding_bidirectional_propagation() {
    const CpuPipeStageFlags seed{true, false, false, true, false, 0};
    const CpuPipeStageFlags relay{true, false, false, true, true, 0};
    const CpuPipeStageFlags no_a{false, true, true, true, true, 4};
    const CpuPipeStageFlags no_c{true, false, false, false, true, 0};

    encoding_case<4>({1, 1, 1, 1}, {relay, relay, relay, seed}, {1, 1, 1, 1},
                     "reverse pass inherits a later seed");
    encoding_case<4>({1, 1, 1, 1}, {seed, relay, relay, relay}, {1, 1, 1, 1},
                     "forward pass inherits earlier seed including last stage");
    encoding_case<4>({1, 1, 1, 1}, {relay, seed, relay, relay}, {1, 1, 1, 1},
                     "middle seed propagates in both directions");
    encoding_case<3>({1, 1, 1}, {relay, relay, relay}, {0, 0, 0},
                     "terminal relay starts at zero without a seed");
    encoding_case<5>({1, 1, 2, 1, 1}, {seed, relay, seed, relay, relay},
                     {1, 1, 0, 0, 0}, "type2 blocks forward propagation");
    encoding_case<5>({1, 1, 2, 1, 1}, {relay, relay, seed, relay, seed},
                     {0, 0, 0, 1, 1}, "type2 blocks reverse propagation");
    encoding_case<3>({1, 1, 1}, {relay, no_a, seed}, {0, 0, 1},
                     "missing type1 capability blocks reverse inheritance");
    encoding_case<3>({1, 1, 1}, {seed, no_c, relay}, {1, 0, 0},
                     "missing encoding C blocks forward inheritance");
    encoding_case<2>({1, 1}, {no_a, seed}, {0, 1},
                     "forward pass overrides a reverse-inherited relay");
    encoding_case<3>({1, 1, 1}, {no_a, relay, seed}, {0, 0, 1},
                     "forward relay copies earlier zero over reverse one");
    encoding_case<2>({2, 1}, {seed, relay}, {0, 0},
                     "encoding requires actual selected type1 not capabilities");
}

void test_encoding_invalid_selection_is_atomic() {
    for (const auto value : {0u, 3u, 4u,
                             std::numeric_limits<std::uint32_t>::max()}) {
        auto state = sentinel_state(4);
        for (std::size_t i = 0; i < 4; ++i) {
            state.selected[i] = 1;
            state.stages[i] = {true, false, false, true, false, 0};
        }
        state.selected[2] = value;
        const auto before = snapshot(state);
        const auto expected = value == 0
            ? CpuPipeFormatStatus::invalid_argument
            : CpuPipeFormatStatus::unsupported_domain;
        check(propagate_cpu_encoding_00bbfb30(state) == expected,
              "encoding rejects unresolved or unsupported active types");
        check_unchanged(state, before,
                        "encoding validates all active types before any writes");
    }
}

}  // namespace

int main() {
    static_assert(noexcept(select_cpu_buffer_types_00bbf644(
        *static_cast<CpuPipeFormatState *>(nullptr), 0, 0,
        FirstStagePreference::unknown)));
    static_assert(noexcept(propagate_cpu_encoding_00bbfb30(
        *static_cast<CpuPipeFormatState *>(nullptr))));

    test_empty_and_stage_limits();
    test_invalid_selection_arguments();
    test_existing_selection_and_atomic_failure();
    test_generic_threshold_and_first_stage_fallback();
    test_selection_branch_table();
    test_prefix_preserved_when_extending_state();
    test_encoding_single_stage_flags();
    test_encoding_bidirectional_propagation();
    test_encoding_invalid_selection_is_atomic();
    if (failures != 0) {
        std::cerr << failures << " of " << checks << " checks failed\n";
        return 1;
    }
    std::cout << "acr_pipe_format_test: " << checks << " checks passed\n";
    return 0;
}
