#include "acr_pipe_format.hpp"

namespace imprint::acr {

CpuPipeFormatStatus select_cpu_buffer_types_00bbf644(
    CpuPipeFormatState &state, std::uint32_t first_choice,
    std::uint32_t second_choice, FirstStagePreference preference) noexcept {
    if (state.stage_count > kMaxCpuPipeStages || first_choice > 2 ||
        second_choice > 2) {
        return CpuPipeFormatStatus::invalid_argument;
    }
    if (preference != FirstStagePreference::unknown &&
        preference != FirstStagePreference::type1 &&
        preference != FirstStagePreference::type2) {
        return CpuPipeFormatStatus::invalid_argument;
    }
    for (std::size_t i = 0; i < state.stage_count; ++i) {
        if (state.selected[i] > 2) {
            return CpuPipeFormatStatus::unsupported_domain;
        }
    }

    const std::uint32_t normal = first_choice < 2 ? 1 : first_choice;
    const std::uint32_t alternate =
        second_choice != 0 ? second_choice : (first_choice < 2 ? 2 : 1);
    // Stage all choices before committing, so a later unsupported stage cannot
    // leave an otherwise valid prefix partially updated.
    std::array<std::uint32_t, kMaxCpuPipeStages> pending{};
    for (std::size_t i = 0; i < state.stage_count; ++i) {
        if (state.selected[i] != 0) continue;

        const auto &stage = state.stages[i];
        const bool a = stage.cpu_type1;
        const bool b = stage.cpu_type2;
        const bool g = stage.generic && stage.generic_count < 5;
        if (!a && !b && !g) {
            return CpuPipeFormatStatus::unsupported_domain;
        }

        if (i == 0 && g && !a && !b) {
            if (preference == FirstStagePreference::unknown) {
                return CpuPipeFormatStatus::unresolved_first_stage;
            }
            pending[i] = preference == FirstStagePreference::type1 ? 1 : 2;
        } else if (b || g) {
            if (a || g) {
                const bool choose_normal =
                    (first_choice <= 1 && a) || (normal == 2 && b);
                pending[i] = choose_normal ? normal : alternate;
            } else {
                pending[i] = 2;
            }
        } else {
            pending[i] = 1;
        }
    }

    for (std::size_t i = 0; i < state.stage_count; ++i) {
        if (state.selected[i] == 0) state.selected[i] = pending[i];
    }
    return CpuPipeFormatStatus::ok;
}

CpuPipeFormatStatus propagate_cpu_encoding_00bbfb30(
    CpuPipeFormatState &state) noexcept {
    if (state.stage_count > kMaxCpuPipeStages) {
        return CpuPipeFormatStatus::invalid_argument;
    }
    for (std::size_t i = 0; i < state.stage_count; ++i) {
        if (state.selected[i] == 0) {
            return CpuPipeFormatStatus::invalid_argument;
        }
        if (state.selected[i] > 2) {
            return CpuPipeFormatStatus::unsupported_domain;
        }
    }

    for (std::size_t end = state.stage_count; end != 0; --end) {
        const std::size_t i = end - 1;
        state.encoding[i] = 0;
        const auto &stage = state.stages[i];
        if (state.selected[i] == 1 && stage.cpu_type1 && stage.encoding_c) {
            if (!stage.encoding_d) {
                state.encoding[i] = 1;
            } else if (end < state.stage_count) {
                state.encoding[i] = state.encoding[end];
            }
        }
    }

    for (std::size_t i = 1; i < state.stage_count; ++i) {
        const auto &stage = state.stages[i];
        if (state.selected[i] == 1 && state.selected[i - 1] == 1 &&
            stage.cpu_type1 && stage.encoding_c && stage.encoding_d) {
            state.encoding[i] = state.encoding[i - 1];
        }
    }
    return CpuPipeFormatStatus::ok;
}

}  // namespace imprint::acr
