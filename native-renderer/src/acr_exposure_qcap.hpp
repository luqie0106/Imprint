#pragma once

namespace imprint::acr {

struct ExposureQcapContext {
    double exposure98;
    double curve_a0;
    double whites_b0;
    double whites80;
    double filtered_p1;
    double filtered_p99;
    double filtered_p0001;
    double high690;
};

struct ExposureQcapResult {
    double anchor_v;
    double response;
    double q;
};

enum class ExposureQcapStatus { ok = 0, invalid_argument = 1, curve_failure = 2 };

// Supported prepared domain, not original request validation: exposure98 and
// curve_a0 in [-120,120], whites_b0/whites80 in [-2,2], finite statistics and
// computed v in [-120,120]. Does not infer profile mode, k, or the log domain.
// Both operations commit output only on success. Input/output references must
// denote valid C++ objects; C API buffer validation belongs to the wrapper.
ExposureQcapStatus evaluate_exposure_context_response(
    const ExposureQcapContext& context, double input, double& output) noexcept;
ExposureQcapStatus prepare_exposure_qcap(
    const ExposureQcapContext& context, ExposureQcapResult& output) noexcept;

}  // namespace imprint::acr
