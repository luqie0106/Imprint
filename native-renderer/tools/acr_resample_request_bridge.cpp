#include "acr_resample_request.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {

struct AxisArgs {
    int kind;
    std::int32_t integer_offset;
    std::int64_t origin;
    std::int64_t step;
    std::int64_t lower;
    std::int64_t upper;
    std::size_t tap_count;
    const float *coefficients;
    std::size_t coefficient_count;
    std::size_t stride;
};

struct RectArgs {
    std::int32_t top;
    std::int32_t left;
    std::size_t height;
    std::size_t width;
};

struct RequestArgs {
    RectArgs destination;
    AxisArgs vertical;
    AxisArgs horizontal;
};

struct ResultArgs {
    RectArgs source;
    RectArgs intermediate;
};

imprint::acr::DestinationRect make_rectangle(const RectArgs &args) noexcept {
    return {args.top, args.left, args.height, args.width};
}

RectArgs copy_rectangle(const imprint::acr::DestinationRect &rect) noexcept {
    return {rect.top, rect.left, rect.height, rect.width};
}

void prepare_axis(const AxisArgs &args, imprint::acr::ResampleAxisPlan &axis,
                  imprint::acr::ResampleCoefficientTable &coefficients) {
    axis = {static_cast<imprint::acr::ResampleAxisKind>(args.kind),
            args.origin, args.step, args.lower, args.upper,
            args.integer_offset};
    coefficients.tap_count = args.tap_count;
    coefficients.stride = args.stride;
    if (args.coefficient_count != 0) {
        if (!args.coefficients)
            throw 1;
        coefficients.coefficients.assign(
            args.coefficients, args.coefficients + args.coefficient_count);
    }
}

}  // namespace

extern "C" int acr_test_prepare_ordinary_source_request(
    const RequestArgs *args, ResultArgs *output) noexcept {
    using namespace imprint::acr;
    if (!args || !output) return -1;

    try {
        OrdinaryResamplePlan plan{};
        prepare_axis(args->vertical, plan.vertical,
                     plan.vertical_coefficients);
        prepare_axis(args->horizontal, plan.horizontal,
                     plan.horizontal_coefficients);

        OrdinarySourceRequest request_output{
            make_rectangle(output->source),
            make_rectangle(output->intermediate),
        };
        std::array<unsigned char, sizeof(request_output)> before{};
        std::memcpy(before.data(), &request_output, before.size());

        const auto status = prepare_ordinary_source_request(
            plan, make_rectangle(args->destination), request_output);
        if (status != OrdinarySourceRequestStatus::ok &&
            std::memcmp(before.data(), &request_output, before.size()) != 0)
            return 99;
        if (status == OrdinarySourceRequestStatus::ok) {
            output->source = copy_rectangle(request_output.source);
            output->intermediate = copy_rectangle(request_output.intermediate);
        }
        return static_cast<int>(status);
    } catch (...) {
        return -2;
    }
}

