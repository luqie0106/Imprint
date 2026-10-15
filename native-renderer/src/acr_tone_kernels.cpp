#include "acr_tone_kernels.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace imprint::acr {
namespace {
constexpr std::int64_t kQ = std::int64_t{1} << 32;
constexpr double kDefaultEpsilon = 0x1p-12;
constexpr std::size_t kMaxSamples = std::size_t{1} << 26;

std::size_t extent(std::uint32_t width, std::uint32_t height) {
    const auto size = std::uint64_t{width} * height;
    if (!width || !height || width > 65535 || height > 65535 || size > kMaxSamples) {
        throw std::invalid_argument("invalid gray image extent");
    }
    return static_cast<std::size_t>(size);
}

void finite_pixels(const float *pixels, std::size_t count) {
    if (!pixels) throw std::invalid_argument("null image input");
    for (std::size_t i = 0; i < count; ++i) {
        if (!std::isfinite(pixels[i])) throw std::invalid_argument("non-finite image input");
    }
}

void finite_result(const GrayImage &image) {
    finite_pixels(image.pixels.data(), image.pixels.size());
}

struct Coordinate { std::int64_t start; unsigned phase; };
struct AxisPlan {
    unsigned radius;
    std::vector<Coordinate> coordinates;
    std::array<std::vector<float>, 128> weights;
};

std::int64_t floor_q(std::int64_t coordinate) {
    return coordinate >= 0 ? coordinate/kQ : -((-coordinate+kQ-1)/kQ);
}

AxisPlan axis_plan(std::uint32_t source_size, std::uint32_t output_size) {
    const double ratio = static_cast<double>(source_size)/output_size;
    const auto step = static_cast<std::int64_t>(ratio*static_cast<double>(kQ)+0.5);
    const double scale = std::min(1.0/(static_cast<double>(step)/kQ), 1.0);
    const auto radius = static_cast<unsigned>(2.0/scale+0.9999);
    const std::int64_t lower = (1-static_cast<std::int64_t>(radius))*kQ;
    const std::int64_t upper = (static_cast<std::int64_t>(source_size)-radius)*kQ;
    const std::int64_t offset = step/2+lower-0x7f000000;
    AxisPlan result{radius, {}, {}};
    result.coordinates.reserve(output_size);
    for (std::uint32_t i = 0; i < output_size; ++i) {
        const auto coordinate = std::clamp(offset+step*i, lower, upper);
        const auto phase = static_cast<unsigned>((static_cast<std::uint64_t>(coordinate)>>25)&127);
        result.coordinates.push_back({floor_q(coordinate), phase});
        auto &weights = result.weights[phase];
        if (!weights.empty()) continue;
        weights.resize(2*radius);
        float sum = 0.0f;
        for (std::size_t j = 0; j < weights.size(); ++j) {
            const double x = scale*(static_cast<double>(j)-radius+1.0-static_cast<double>(phase)/128.0);
            weights[j] = x < 2.0 ? static_cast<float>(std::exp(-2.0*x*x)) : 0.0f;
            sum += weights[j];
        }
        const float norm = 1.0f/sum;
        for (float &weight : weights) weight *= norm;
    }
    return result;
}

float filter_line(const float *source, std::size_t stride, std::uint32_t size,
                  const AxisPlan &plan, std::uint32_t position) {
    const auto &coordinate = plan.coordinates[position];
    const auto &weights = plan.weights[coordinate.phase];
    float result = 0.0f;
    for (std::size_t j = 0; j < weights.size(); ++j) {
        const auto index = std::clamp(coordinate.start+static_cast<std::int64_t>(j),
                                      std::int64_t{0}, static_cast<std::int64_t>(size)-1);
        result += source[static_cast<std::size_t>(index)*stride]*weights[j];
    }
    return result;
}
}  // namespace

LogSceneStats build_log_scene_stats(const float *pixels, std::size_t count, double epsilon) {
    if (!count || count > kMaxSamples || !std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("invalid log statistics input");
    }
    finite_pixels(pixels, count);
    std::vector<float> sorted(pixels, pixels+count);
    std::sort(sorted.begin(), sorted.end());
    auto raw_q = [&](double p) {
        return static_cast<double>(sorted[std::min(count-1, static_cast<std::size_t>((count-1)*p))]);
    };
    const double lower = std::log(epsilon)/0.6931471805599453+1.0;
    const double threshold = std::max(raw_q(.99)-.5, (raw_q(.01)+raw_q(.99))*.5);
    std::vector<double> effective;
    effective.reserve(count);
    double previous = -1000.0;
    for (const float sample : sorted) {
        const double x = sample;
        if (x >= lower && (x != previous || x < threshold)) effective.push_back(x);
        previous = x;
    }
    const bool fallback = effective.empty();
    if (fallback) effective.push_back(std::log(epsilon)/0.6931471805599453);
    const auto m = effective.size();
    const auto lo = std::min(m-1, static_cast<std::size_t>(m*.0001));
    const auto hi = std::min(m-1, static_cast<std::size_t>(m*.9999));
    auto linear = [&](double x) {
        return std::max(0.0, std::exp2(std::clamp(x,-120.0,120.0))-epsilon);
    };
    const double left = linear(effective[lo]), right = linear(effective[hi]);
    double representative = left;
    if (left != right) {
        double sum = 0.0;
        for (std::size_t i = lo; i <= hi; ++i) {
            const double x = std::clamp((linear(effective[i])-left)/(right-left),0.0,1.0);
            sum += x <= .0031308 ? 12.92*x : 1.055*std::pow(x,1.0/2.4)-.055;
        }
        const double mean = sum/static_cast<double>(hi-lo+1);
        const double decoded = mean <= .040449936 ? mean*.07739938080495357
                             : std::pow((mean+.055)*.9478672985781991,2.4);
        representative = left+decoded*(right-left);
    }
    const double center = std::log2(std::clamp(representative,0.0,0x1p120)+epsilon);
    if (!std::isfinite(center)) throw std::invalid_argument("invalid log statistics result");
    return {center, effective[lo], effective[hi],
            (effective[(m-1)/2]+effective[m/2])*.5, m, fallback};
}

GrayImage resize_gray(const float *pixels, std::uint32_t width,
                      std::uint32_t height, std::uint32_t output_width,
                      std::uint32_t output_height) {
    const auto input_size = extent(width, height);
    const auto output_size = extent(output_width, output_height);
    // Also bound the intermediate vertical stage, not just the endpoints.
    const auto vertical_size = extent(width, output_height);
    finite_pixels(pixels, input_size);
    GrayImage vertical{width, output_height, std::vector<float>(vertical_size)};
    if (height == output_height) {
        std::copy(pixels, pixels+input_size, vertical.pixels.begin());
    } else {
        const auto plan = axis_plan(height, output_height);
        for (std::uint32_t y = 0; y < output_height; ++y) {
            for (std::uint32_t x = 0; x < width; ++x) {
                vertical.pixels[static_cast<std::size_t>(y)*width+x] =
                    filter_line(pixels+x, width, height, plan, y);
            }
        }
    }
    if (width == output_width) { finite_result(vertical); return vertical; }
    GrayImage output{output_width, output_height, std::vector<float>(output_size)};
    const auto plan = axis_plan(width, output_width);
    for (std::uint32_t y = 0; y < output_height; ++y) {
        for (std::uint32_t x = 0; x < output_width; ++x) {
            output.pixels[static_cast<std::size_t>(y)*output_width+x] =
                filter_line(vertical.pixels.data()+static_cast<std::size_t>(y)*width,
                            1, width, plan, x);
        }
    }
    finite_result(output);
    return output;
}

double calculate_flare_for_log(const float *pixels, std::size_t pixel_count,
                                std::uint32_t channels) {
    if (!pixel_count || !channels || channels > 4 || pixel_count > kMaxSamples/channels) {
        throw std::invalid_argument("invalid flare image extent");
    }
    finite_pixels(pixels, pixel_count*channels);
    double average = 0.0;
    // Per-channel double accumulation, then channel-average, as in 00c107f8
    // called by 00aef2fc. Actual Adobe tile traversal can change sum rounding.
    for (std::uint32_t c = 0; c < channels; ++c) {
        double sum = 0.0;
        for (std::size_t i = 0; i < pixel_count; ++i) sum += pixels[i*channels+c];
        average += sum/static_cast<double>(pixel_count);
    }
    average /= channels;
    return std::clamp(average*0x1p-20, 0.0, kDefaultEpsilon);
}

double select_log_epsilon(int source_mode, double stored_flare) {
    if (!std::isfinite(stored_flare)) throw std::invalid_argument("invalid stored flare");
    if (source_mode == 0 && stored_flare > 0.0) return std::min(stored_flare, kDefaultEpsilon);
    return kDefaultEpsilon;
}

GrayImage make_gray_log(const float *rgb, std::uint32_t width,
                        std::uint32_t height, double gain_ev, double epsilon) {
    const auto count = extent(width, height);
    // extent() bounds gray pixels, not RGB channel values. Dividing its cap
    // by three would reject ordinary 6064x4040 camera inputs.
    if (!std::isfinite(gain_ev) || std::abs(gain_ev) > 120.0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0 || epsilon > kDefaultEpsilon ||
        static_cast<float>(epsilon) == 0.0f) {
        throw std::invalid_argument("invalid gray log parameters");
    }
    finite_pixels(rgb, count*3);
    const float gain = static_cast<float>(std::exp2(gain_ev));
    const float e = static_cast<float>(epsilon);
    GrayImage output{width, height, std::vector<float>(count)};
    for (std::size_t i = 0; i < count; ++i) {
        const float gray = std::max(0.0f, (rgb[3*i+1]*0.59f+rgb[3*i]*0.3f)+rgb[3*i+2]*0.110000014f);
        const float scaled = gray*gain;
        if (!std::isfinite(gray) || !std::isfinite(scaled)) {
            throw std::invalid_argument("gray log intermediate overflow");
        }
        output.pixels[i] = std::log2(e+std::min(scaled, 0x1p120f));
    }
    finite_result(output);
    return output;
}

GrayImage local_laplacian_gray(const float *pixels, std::uint32_t width,
                              std::uint32_t height, double low, double high,
                              double alpha, double beta, double sigma,
                              LogSceneStats *collapse_stats,
                              std::uint32_t statistics_level, double epsilon) {
    const auto count = extent(width, height);
    if (!std::isfinite(low) || !std::isfinite(high) || high < low || high-low > 40.001 ||
        !std::isfinite(alpha) || alpha <= 0.0 || alpha > 8.0 ||
        !std::isfinite(beta) || beta < 0.0 || beta > 8.0 ||
        !std::isfinite(sigma) || sigma <= 0.0 || sigma > 128.0) {
        throw std::invalid_argument("invalid local remap parameters");
    }
    if (collapse_stats && (!std::isfinite(epsilon) || epsilon <= 0.0)) {
        throw std::invalid_argument("invalid collapse statistics epsilon");
    }
    const double float_max = static_cast<double>(std::numeric_limits<float>::max());
    if (low < -float_max || low > float_max || high < -float_max || high > float_max) {
        throw std::invalid_argument("local remap bounds exceed float32 range");
    }
    const float lower = static_cast<float>(low), upper = static_cast<float>(high);
    if (!std::isfinite(lower) || !std::isfinite(upper) || upper < lower) {
        throw std::invalid_argument("local remap bounds are not representable as float32");
    }
    if (high > low && upper == lower) {
        throw std::invalid_argument("local remap span is not representable as float32");
    }
    const double wide_range = static_cast<double>(upper)-static_cast<double>(lower);
    if (!std::isfinite(wide_range) || wide_range < 0.0 || wide_range > 40.001 ||
        wide_range >= static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("invalid float32 local remap range");
    }
    const float range = static_cast<float>(wide_range);
    if (!std::isfinite(range) || (wide_range > 0.0 && range == 0.0f) ||
        static_cast<double>(range) >= static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("local remap range is not representable as float32");
    }
    const int intervals = std::max(1, static_cast<int>(range));
    const float step = range/static_cast<float>(intervals);
    if (!std::isfinite(step) || step < 0.0f || (range > 0.0f && step == 0.0f)) {
        throw std::invalid_argument("local remap step is not representable as float32");
    }
    const double wide_inverse_step = step == 0.0f ? 0.0 : 1.0/static_cast<double>(step);
    if (!std::isfinite(wide_inverse_step) ||
        wide_inverse_step > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument("local remap inverse step exceeds float32 range");
    }
    const float inverse_step = static_cast<float>(wide_inverse_step);
    const float a = static_cast<float>(alpha), b = static_cast<float>(beta);
    const float s = static_cast<float>(sigma);
    if (!std::isfinite(a) || a <= 0.0f || !std::isfinite(b) || b < 0.0f ||
        !std::isfinite(s) || s <= 0.0f || !std::isfinite(inverse_step)) {
        throw std::invalid_argument("local remap parameters are not representable as float32");
    }
    finite_pixels(pixels, count);
    GrayImage input{width, height, std::vector<float>(pixels, pixels+count)};
    std::vector<GrayImage> guide;
    guide.push_back(input);
    while (guide.size() < 16 && std::max(guide.back().width, guide.back().height) >= 2) {
        const auto &last = guide.back();
        guide.push_back(resize_gray(last.pixels.data(), last.width, last.height,
                                    (last.width+1)/2, (last.height+1)/2));
    }
    if (guide.size() == 1) {
        if (collapse_stats) throw std::invalid_argument("no collapsed level for statistics callback");
        return input;
    }
    if (collapse_stats && statistics_level >= guide.size()-1) {
        throw std::invalid_argument("statistics callback must address a collapsed level");
    }
    LogSceneStats selected_stats{};
    std::vector<GrayImage> residuals;
    residuals.reserve(guide.size()-1);
    for (std::size_t level = 0; level+1 < guide.size(); ++level) {
        residuals.push_back({guide[level].width, guide[level].height,
                            std::vector<float>(guide[level].pixels.size(), 0.0f)});
    }
    if (alpha == 1.0 && beta == 1.0) {
        for (std::size_t level = 0; level < residuals.size(); ++level) {
            const auto &next = guide[level+1];
            const auto expanded = resize_gray(next.pixels.data(), next.width, next.height,
                                               guide[level].width, guide[level].height);
            for (std::size_t i = 0; i < residuals[level].pixels.size(); ++i) {
                residuals[level].pixels[i] = guide[level].pixels[i]-expanded.pixels[i];
            }
        }
        GrayImage output = guide.back();
        for (std::size_t level = residuals.size(); level-- > 0;) {
            auto expanded = resize_gray(output.pixels.data(), output.width, output.height,
                                         residuals[level].width, residuals[level].height);
            for (std::size_t i = 0; i < expanded.pixels.size(); ++i) expanded.pixels[i] += residuals[level].pixels[i];
            output = std::move(expanded);
            if (collapse_stats && level == statistics_level) {
                selected_stats = build_log_scene_stats(output.pixels.data(), output.pixels.size(), epsilon);
            }
        }
        finite_result(output);
        if (collapse_stats) *collapse_stats = selected_stats;
        return output;
    }
    for (int sample = 0; sample <= intervals; ++sample) {
        const float anchor = lower+static_cast<float>(sample)*step;
        GrayImage remapped{width, height, std::vector<float>(count)};
        for (std::size_t i = 0; i < count; ++i) {
            if (a == 1.0f && b == 0.0f) {
                remapped.pixels[i] = std::clamp(pixels[i], anchor-s, anchor+s);
            } else {
                const float delta = pixels[i]-anchor;
                const float magnitude = std::abs(delta);
                const float mapped = magnitude <= s ? s*std::pow(magnitude/s, a)
                                                     : s+b*(magnitude-s);
                remapped.pixels[i] = anchor+(delta < 0.0f ? -mapped : mapped);
            }
        }
        for (std::size_t level = 0; level < residuals.size(); ++level) {
            const auto &g = guide[level];
            const auto &next_g = guide[level+1];
            auto reduced = resize_gray(remapped.pixels.data(), remapped.width, remapped.height,
                                        next_g.width, next_g.height);
            const auto expanded = resize_gray(reduced.pixels.data(), reduced.width, reduced.height,
                                               g.width, g.height);
            auto &r = residuals[level].pixels;
            for (std::size_t i = 0; i < r.size(); ++i) {
                const float x = g.pixels[i];
                float weight = 0.0f;
                if (sample == 0) {
                    if (x < anchor) weight = 1.0f;
                    else if (x < anchor+step) weight = (x-(anchor+step))*(-inverse_step);
                } else if (sample == intervals) {
                    if (x >= anchor) weight = 1.0f;
                    else if (x >= anchor-step) weight = inverse_step*(x-(anchor-step));
                } else if (x >= anchor-step) {
                    if (x < anchor) weight = inverse_step*(x-(anchor-step));
                    else if (x < anchor+step) weight = (x-(anchor+step))*(-inverse_step);
                }
                if (weight != 0.0f) r[i] += weight*(remapped.pixels[i]-expanded.pixels[i]);
            }
            remapped = std::move(reduced);
        }
    }
    // 00a0aa9c(...,1) retains the original coarsest Gaussian; only the
    // detail bands are remapped. Then 00a0abe0 expands/adds toward level 0.
    GrayImage output = guide.back();
    for (std::size_t level = residuals.size(); level-- > 0;) {
        auto expanded = resize_gray(output.pixels.data(), output.width, output.height,
                                     residuals[level].width, residuals[level].height);
        for (std::size_t i = 0; i < expanded.pixels.size(); ++i) {
            expanded.pixels[i] += residuals[level].pixels[i];
        }
        output = std::move(expanded);
        if (collapse_stats && level == statistics_level) {
            selected_stats = build_log_scene_stats(output.pixels.data(), output.pixels.size(), epsilon);
        }
    }
    finite_result(output);
    if (collapse_stats) *collapse_stats = selected_stats;
    return output;
}

GrayImage build_delta_mask(const float *original, const float *collapsed,
                          std::uint32_t width, std::uint32_t height,
                          double original_center, double mask_center) {
    const auto count = extent(width, height);
    if (!std::isfinite(original_center) || !std::isfinite(mask_center)) {
        throw std::invalid_argument("invalid mask center");
    }
    finite_pixels(original, count);
    finite_pixels(collapsed, count);
    const float shift = static_cast<float>(original_center)-static_cast<float>(mask_center);
    GrayImage result{width, height, std::vector<float>(count)};
    for (std::size_t i = 0; i < count; ++i) {
        const float delta = (shift+collapsed[i])-original[i];
        if (!std::isfinite(delta)) throw std::invalid_argument("mask delta overflow");
        result.pixels[i] = std::min(delta, 4.0f);
    }
    return result;
}

LocalToneMask build_local_tone_mask(const float *log_gray, std::uint32_t width,
                                    std::uint32_t height, double epsilon, int sigma_mode) {
    const auto count = extent(width, height);
    if (count < 2 || !std::isfinite(epsilon) || epsilon <= 0.0 || epsilon > kDefaultEpsilon ||
        sigma_mode < 0 || sigma_mode > 2) {
        throw std::invalid_argument("invalid local tone mask parameters");
    }
    finite_pixels(log_gray, count);
    GrayImage statistics_image{width, height, std::vector<float>(log_gray,log_gray+count)};
    std::uint32_t statistics_level = 0;
    // Select the level BEFORE the next level's long side becomes <128.
    while (statistics_level < 15 && std::max(statistics_image.width,statistics_image.height) >= 2) {
        const auto next_width = (statistics_image.width+1)/2;
        const auto next_height = (statistics_image.height+1)/2;
        if (std::max(next_width,next_height) < 128) break;
        statistics_image = resize_gray(statistics_image.pixels.data(),statistics_image.width,
                                       statistics_image.height,next_width,next_height);
        ++statistics_level;
    }
    const auto source_stats = build_log_scene_stats(statistics_image.pixels.data(),
                                                    statistics_image.pixels.size(),epsilon);
    constexpr std::array<double,6> ranges{2.0,3.5,5.0,6.5,12.0,14.0};
    constexpr std::array<double,6> sigmas{1.4,1.6,2.3,2.5,2.5,2.0};
    const double range = source_stats.high-source_stats.low;
    double linear_sigma = sigmas.back();
    if (range <= ranges.front()) linear_sigma = sigmas.front();
    else {
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            if (range <= ranges[i]) {
                linear_sigma = sigmas[i-1]+(sigmas[i]-sigmas[i-1])*
                               ((range-ranges[i-1])/(ranges[i]-ranges[i-1]));
                break;
            }
        }
    }
    if (sigma_mode > 0) {
        const double log_sigma = std::log(linear_sigma)/0.6931471805599453;
        linear_sigma = std::exp2(log_sigma+(0.4854268271702417-log_sigma)*sigma_mode*.5);
    }
    const double sigma = std::log2(epsilon+linear_sigma);
    // The anchor endpoints are evaluated in f32 by 00a0c1c8/00a0b650.
    const float median = static_cast<float>(source_stats.median);
    const float low = std::max(static_cast<float>(source_stats.low),median-20.0f);
    const float high = std::min(static_cast<float>(source_stats.high),median+20.0f);
    LogSceneStats mask_stats{};
    const auto collapsed = local_laplacian_gray(log_gray,width,height,low,high,1.0,0.0,sigma,
                                                &mask_stats,statistics_level,epsilon);
    auto delta = build_delta_mask(log_gray,collapsed.pixels.data(),width,height,
                                  source_stats.center,mask_stats.center);
    const double shift = source_stats.center-mask_stats.center;
    auto aligned = mask_stats;
    aligned.center += shift;
    aligned.low += shift;
    aligned.high += shift;
    return {std::move(delta),source_stats,mask_stats,aligned,sigma,statistics_level};
}

}  // namespace imprint::acr
