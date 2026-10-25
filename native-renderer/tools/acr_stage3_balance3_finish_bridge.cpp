// Test-only ABI adapter. Row arithmetic stays in the recovered native kernels.
#include "../src/acr_stage3_balance3_finish.hpp"

#include <cstddef>

extern "C" int acr_test_balance3_basic(
    const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    float *dst, std::size_t dst_count) noexcept {
  return static_cast<int>(imprint::acr::balance3_basic_prepared_row_007ed860(
      source, stride, diff, diff_count, width, phase, row_even, dst, dst_count));
}

extern "C" int acr_test_balance3_finish(
    int branch, const float *source, std::size_t stride, const float *diff,
    std::size_t diff_count, std::size_t width, int phase, int row_even,
    const float *flat_limits, float *dst, std::size_t dst_count) noexcept {
  using imprint::acr::Balance3FinishLimits;
  using imprint::acr::Balance3FinishStatus;
  using imprint::acr::balance3_finish_prepared_row_0081403c;
  using imprint::acr::balance3_finish_prepared_row_013312fc;

  if (flat_limits == nullptr) {
    return static_cast<int>(Balance3FinishStatus::invalid_argument);
  }
  if (branch != 0 && branch != 1) {
    return static_cast<int>(Balance3FinishStatus::invalid_argument);
  }

  // Copy controls before entering the API; the public API also documents this
  // copy-before-write behavior for callers whose limits alias destination.
  const Balance3FinishLimits limits{
      flat_limits[0], flat_limits[1], flat_limits[2], flat_limits[3]};
  const auto status = branch == 0
      ? balance3_finish_prepared_row_013312fc(
            source, stride, diff, diff_count, width, phase, row_even, limits,
            dst, dst_count)
      : balance3_finish_prepared_row_0081403c(
            source, stride, diff, diff_count, width, phase, row_even, limits,
            dst, dst_count);
  return static_cast<int>(status);
}
