#include "../src/acr_source_log.hpp"

#include <cstddef>

extern "C" int acr_test_source_log(const float *input, float *output,
                                   std::size_t count, double epsilon,
                                   int pin_overrange) {
    if (pin_overrange != 0 && pin_overrange != 1) {
        return static_cast<int>(imprint::acr::SourceLogStatus::invalid_input);
    }
    return static_cast<int>(imprint::acr::apply_acr_source_log(
        input, output, count, epsilon, pin_overrange != 0));
}
