#pragma once
#include "acr_stage3_balance3_finish.hpp"

namespace imprint::acr {
enum class Balance3ControlsStatus : int {
    ok = 0,
    invalid_argument = 1,
    invalid_range = 2,
    unsupported_basic_branch = 3,
    unsupported_domain = 4,
};

// Only the eligible no-matrix bounded branch of 007ed860. Raw (0,0) selects
// a different original basic branch and is rejected here. `direct` is the
// extracted object byte+0x28, not an inferred camera property. Output must
// have natural object alignment; all failures leave it unchanged.
Balance3ControlsStatus prepare_balance3_finish_limits_007ed860(
    double lower, double upper, bool direct, Balance3FinishLimits *out) noexcept;
}
