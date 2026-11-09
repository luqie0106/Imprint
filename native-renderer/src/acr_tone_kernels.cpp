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
        // 012e3c28..48: store float, promote to double for the sum, then
        // round the double reciprocal to a float normalizer.
        double sum = 0.0;
        for (std::size_t j = 0; j < weights.size(); ++j) {
            const double x = scale*(static_cast<double>(j)-radius+1.0-static_cast<double>(phase)/128.0);
            weights[j] = x < 2.0 ? static_cast<float>(std::exp(-2.0*x*x)) : 0.0f;
            sum += static_cast<double>(weights[j]);
        }
        const float norm = static_cast<float>(1.0/sum);
        for (float &weight : weights) weight *= norm;
    }
    return result;
}

float filter_line(const float *source, std::size_t stride, std::uint32_t size,
                  const AxisPlan &plan, std::uint32_t position, bool vertical) {
    const auto &coordinate = plan.coordinates[position];
    const auto &weights = plan.weights[coordinate.phase];
    auto sample = [&](std::size_t j) {
        const auto index = std::clamp(coordinate.start+static_cast<std::int64_t>(j),
                                      std::int64_t{0}, static_cast<std::int64_t>(size)-1);
        return source[static_cast<std::size_t>(index)*stride];
    };
    if (vertical) {
        // 01341a30 baseline: +0 initialization, skip zero weights, SIMD FMLA.
        float result = 0.0f;
        for (std::size_t j = 0; j < weights.size(); ++j)
            if (weights[j] != 0.0f) result = std::fma(sample(j),weights[j],result);
        return result;
    }
    // Same arithmetic as ResampleRowMode::matrix_horizontal_baseline, using
    // logical border-clamped samples instead of allocating a padded image.
    // 0134147c: only 4/6/8 taps use tap1-product, tap0-FMA, tap2..N-1 FMA.
    if (weights.size() == 4 || weights.size() == 6 || weights.size() == 8) {
        float result = std::fma(weights[0],sample(0),weights[1]*sample(1));
        for (std::size_t j = 2; j < weights.size(); ++j)
            result = std::fma(weights[j],sample(j),result);
        return result;
    }
    float result = weights[0]*sample(0);
    const auto block_end = (weights.size()-1) & ~std::size_t{15};
    for (std::size_t j = 1; j <= block_end; ++j) {
        const float product = weights[j]*sample(j);
        result += product;
    }
    for (std::size_t j = block_end+1; j < weights.size(); ++j)
        result = std::fma(weights[j],sample(j),result);
    return result;
}
}  // namespace

LocalToneGeometry prepare_local_tone_geometry(std::uint32_t width, std::uint32_t height) {
    (void)extent(width,height);
    LocalToneGeometry plan{};
    plan.levels[0] = {width,height};
    plan.level_count = 1;
    while (plan.level_count < plan.levels.size() && std::max(width,height) >= 2) {
        width = (width+1)/2;
        height = (height+1)/2;
        plan.levels[plan.level_count++] = {width,height};
    }
    for (std::uint32_t level = 1; level < plan.level_count; ++level) {
        const auto next = plan.levels[level];
        if (std::max(next.width,next.height) < 128) break;
        plan.statistics_level = level;
    }
    return plan;
}

LogSceneStats build_log_scene_stats(const float *pixels, std::size_t count, double epsilon) {
    if (!count || count > kMaxSamples || !std::isfinite(epsilon) || epsilon <= 0.0) {
        throw std::invalid_argument("invalid log statistics input");
    }
    finite_pixels(pixels, count);
    std::vector<float> sorted(pixels, pixels+count);
    std::sort(sorted.begin(), sorted.end());
    // Raw low quantiles round down; raw high quantiles round up. In
    // particular 009d74c4 FRINTP supplies .99 to the repeat threshold.
    const double raw_low = sorted[static_cast<std::size_t>(std::floor((count-1)*.01))];
    const double raw_high = sorted[static_cast<std::size_t>(std::ceil((count-1)*.99))];
    const double lower = std::log(epsilon)/0.6931471805599453+1.0;
    const double threshold = std::max(raw_high-.5, std::fma(raw_high-raw_low,.5,raw_low));
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
    struct Group { double log; std::uint32_t count; };
    std::vector<Group> groups;
    for (double value : effective) {
        if (!groups.empty() && groups.back().log == value) ++groups.back().count;
        else groups.push_back({value,1});
    }
    std::size_t left_index = 0, cumulative = 0;
    for (; left_index+1 < groups.size(); ++left_index) {
        cumulative += groups[left_index].count;
        if (cumulative > lo) break;
    }
    std::size_t right_index = groups.size()-1;
    std::uint32_t right_count = 1;
    auto rank = static_cast<std::uint32_t>(m)-groups.back().count;
    for (std::size_t index = groups.size(); index-- > 0;) {
        if (rank <= hi) {
            right_index = index;
            right_count = left_index == index ? 0 : static_cast<std::uint32_t>(hi)-rank+1;
            break;
        }
        // 009d8670 loads counts[current] with post-decrement, then subtracts
        // in uint32. This differs from a conventional prefix-rank lookup when
        // the high rank excludes the last group; retain the observed order.
        rank -= groups[index].count;
    }
    const double left = linear(groups[left_index].log), right = linear(groups[right_index].log);
    const double inverse = left == right ? 0.0 : -1.0/(left-right);
    double sum = left_index == right_index ? 0.0 : static_cast<double>(right_count);
    for (std::size_t i = left_index+1; i < right_index; ++i) {
        const double x = std::clamp(std::fma(inverse,linear(groups[i].log)-left,0.0),0.0,1.0);
        const double encoded = x <= .0031308 ? 12.92*x
            : std::fma(std::pow(x,1.0/2.4),1.055,-.055);
        sum = std::fma(static_cast<double>(groups[i].count),encoded,sum);
    }
    const double mean = sum/static_cast<double>(hi-lo+1);
    const double decoded = mean <= .040449936 ? mean*.07739938080495357
                         : std::pow((mean+.055)*.9478672985781991,2.4);
    // 009d8770 FMSUB computes left - (left-right)*decoded with one rounding.
    const double representative = std::fma(-(left-right),decoded,left);
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
                    filter_line(pixels+x, width, height, plan, y, true);
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
                            1, width, plan, x, false);
        }
    }
    finite_result(output);
    return output;
}

GrayImage swap_left_right_gray(const float *pixels, std::uint32_t width,
                               std::uint32_t height, bool inverse) {
    const auto count = extent(width,height);
    if (width < 2) throw std::invalid_argument("swap requires width >=2");
    finite_pixels(pixels,count);
    const auto shift = inverse ? width/2 : (width+1)/2;
    GrayImage output{width,height,std::vector<float>(count)};
    for (std::uint32_t y = 0; y < height; ++y) {
        const auto row = static_cast<std::size_t>(y)*width;
        std::copy_n(pixels+row+shift,width-shift,output.pixels.data()+row);
        std::copy_n(pixels+row,shift,output.pixels.data()+row+width-shift);
    }
    return output;
}

GrayImage blend_horizontal_gray(const float *first, const float *second,
                                std::uint32_t width, std::uint32_t height) {
    const auto count = extent(width,height);
    if (width < 2) throw std::invalid_argument("blend requires width >=2");
    finite_pixels(first,count);
    finite_pixels(second,count);
    const float inverse_width = 1.0f/static_cast<float>(width-1);
    GrayImage output{width,height,std::vector<float>(count)};
    for (std::uint32_t x = 0; x < width; ++x) {
        const float t = std::clamp(static_cast<float>(x)*inverse_width,0.0f,1.0f);
        const float a = std::abs(std::fma(t,2.0f,-1.0f));
        const double a64 = a;
        // 00a094dc..508 / 00a09588..5a4: widen before the square and
        // double FMADD, multiply in double, then round the weight to float.
        const float weight = static_cast<float>((a64*a64)*std::fma(-2.0,a64,3.0));
        for (std::uint32_t y = 0; y < height; ++y) {
            const auto i = static_cast<std::size_t>(y)*width+x;
            const float difference = second[i]-first[i];
            output.pixels[i] = std::fma(weight,difference,first[i]);
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
        // 00889448/50/5c and the SIMD/tail paths: FMUL G, FMADD R,
        // FMADD B, FMAXNM +0. These are not the Q14 output-gray weights.
        const float green = rgb[3*i+1] * 0x1.2e147ap-1f;
        const float red_green = std::fma(rgb[3*i], 0x1.333334p-2f, green);
        const float gray = std::max(0.0f,
            std::fma(rgb[3*i+2], 0x1.c28f6p-4f, red_green));
        // 00889fe4 skips the gain stage for both signed zeros of EV90.
        const float scaled = gain_ev == 0.0 ? gray : gray*gain;
        if (!std::isfinite(gray) || !std::isfinite(scaled)) {
            throw std::invalid_argument("gray log intermediate overflow");
        }
        output.pixels[i] = std::log2(e+std::min(scaled, 0x1p120f));
    }
    finite_result(output);
    return output;
}

LocalRemapSampling prepare_local_remap_sampling(double low, double high) {
    if (!std::isfinite(low) || !std::isfinite(high) || high < low || high-low > 40.001)
        throw std::invalid_argument("invalid local remap endpoints");
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
    const float range = upper-lower; // 00a0b954: binary32 FSUB.
    if (!std::isfinite(range) || (wide_range > 0.0 && range == 0.0f) ||
        static_cast<double>(range) >= static_cast<double>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("local remap range is not representable as float32");
    }
    const int intervals = std::max(1, static_cast<int>(std::ceil(range)));
    const float step = range/static_cast<float>(intervals);
    if (!std::isfinite(step) || step < 0.0f || (range > 0.0f && step == 0.0f)) {
        throw std::invalid_argument("local remap step is not representable as float32");
    }
    const double wide_inverse_step = step == 0.0f ? 0.0 : 1.0/static_cast<double>(step);
    if (!std::isfinite(wide_inverse_step) ||
        wide_inverse_step > static_cast<double>(std::numeric_limits<float>::max())) {
        throw std::invalid_argument("local remap inverse step exceeds float32 range");
    }
    // Keep the wide guard, but execute the saved consumer's binary32 FDIV.
    const float inverse_step = step == 0.0f ? 0.0f : 1.0f/step;
    return {lower, upper, static_cast<std::uint32_t>(intervals), step, inverse_step};
}

GrayImage local_laplacian_gray(const float *pixels, std::uint32_t width,
                              std::uint32_t height, double low, double high,
                              double alpha, double beta, double sigma,
                              LogSceneStats *collapse_stats,
                              std::uint32_t statistics_level, double epsilon) {
    const auto count = extent(width, height);
    const auto sampling = prepare_local_remap_sampling(low, high);
    const float lower = sampling.low, step = sampling.step;
    const float inverse_step = sampling.inverse_step;
    const int intervals = static_cast<int>(sampling.intervals);
    if (!std::isfinite(alpha) || alpha <= 0.0 || alpha > 8.0 ||
        !std::isfinite(beta) || beta < 0.0 || beta > 8.0 ||
        !std::isfinite(sigma) || sigma <= 0.0 || sigma > 128.0) {
        throw std::invalid_argument("invalid local remap parameters");
    }
    if (collapse_stats && (!std::isfinite(epsilon) || epsilon <= 0.0)) {
        throw std::invalid_argument("invalid collapse statistics epsilon");
    }
    const float a = static_cast<float>(alpha), b = static_cast<float>(beta);
    const float s = static_cast<float>(sigma);
    if (!std::isfinite(a) || a <= 0.0f || !std::isfinite(b) || b < 0.0f ||
        !std::isfinite(s) || s <= 0.0f || !std::isfinite(inverse_step)) {
        throw std::invalid_argument("local remap parameters are not representable as float32");
    }
    finite_pixels(pixels, count);
    GrayImage input{width, height, std::vector<float>(pixels, pixels+count)};
    const auto geometry = prepare_local_tone_geometry(width,height);
    std::vector<GrayImage> guide;
    guide.push_back(input);
    for (std::uint32_t level = 1; level < geometry.level_count; ++level) {
        const auto &last = guide.back();
        const auto next = geometry.levels[level];
        guide.push_back(resize_gray(last.pixels.data(), last.width, last.height,
                                    next.width, next.height));
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
        // Shared by 00a09938 and 00a0bc48 (including their SIMD paths).
        const float anchor = std::fma(step, static_cast<float>(sample), lower);
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
                const float residual = remapped.pixels[i]-expanded.pixels[i];
                if (sample == 0) {
                    // First sample initializes the accumulator (FMUL only).
                    if (x < anchor) r[i] = residual;
                    else if (x < anchor+step)
                        r[i] = ((x-(anchor+step))*(-inverse_step))*residual;
                    else r[i] = 0.0f;
                } else if (sample == intervals) {
                    if (x >= anchor) r[i] += residual;
                    else if (x >= anchor-step)
                        r[i] = std::fma(residual, inverse_step*(x-(anchor-step)), r[i]);
                } else if (x >= anchor-step) {
                    if (x < anchor)
                        r[i] = std::fma(residual, inverse_step*(x-(anchor-step)), r[i]);
                    else if (x < anchor+step)
                        r[i] = std::fma(residual, (x-(anchor+step))*(-inverse_step), r[i]);
                }
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

double prepare_local_mask_sigma(double low, double high, double epsilon, int sigma_mode) {
    if (!std::isfinite(low) || !std::isfinite(high) || high < low ||
        !std::isfinite(high-low) || !std::isfinite(epsilon) || epsilon <= 0.0 ||
        epsilon > kDefaultEpsilon || sigma_mode < 0 || sigma_mode > 2)
        throw std::invalid_argument("invalid mask sigma parameters");
    constexpr std::array<double,6> ranges{2.0,3.5,5.0,6.5,12.0,14.0};
    constexpr std::array<double,6> sigmas{1.4,1.6,2.3,2.5,2.5,2.0};
    const double range = high-low;
    double linear_sigma = sigmas.back();
    if (range <= ranges.front()) linear_sigma = sigmas.front();
    else {
        for (std::size_t i = 1; i < ranges.size(); ++i) {
            if (range <= ranges[i]) {
                linear_sigma = std::fma((range-ranges[i-1])/(ranges[i]-ranges[i-1]),
                                       sigmas[i]-sigmas[i-1], sigmas[i-1]);
                break;
            }
        }
    }
    if (sigma_mode > 0) {
        const double log_sigma = std::log(linear_sigma)/0.6931471805599453;
        linear_sigma = std::exp2(std::fma(static_cast<double>(sigma_mode)*0.5,
            0.4854268271702417-log_sigma, log_sigma));
    }
    return std::log2(epsilon+linear_sigma);
}

LocalToneMask build_local_tone_mask(const float *log_gray, std::uint32_t width,
                                    std::uint32_t height, double epsilon, int sigma_mode,
                                    std::uint32_t boundary_mode) {
    const auto count = extent(width, height);
    if (count < 2 || !std::isfinite(epsilon) || epsilon <= 0.0 || epsilon > kDefaultEpsilon ||
        sigma_mode < 0 || sigma_mode > 2 || (boundary_mode != 2 && boundary_mode != 4)) {
        throw std::invalid_argument("invalid local tone mask parameters");
    }
    finite_pixels(log_gray, count);
    const auto geometry = prepare_local_tone_geometry(width,height);
    struct PassResult {
        GrayImage collapsed;
        LogSceneStats source_stats;
        LogSceneStats mask_stats;
        double sigma;
        std::uint32_t statistics_level;
    };
    const auto run_pass = [&](const float *pixels) {
        GrayImage statistics_image{width, height, std::vector<float>(pixels,pixels+count)};
        const auto statistics_level = geometry.statistics_level;
        for (std::uint32_t level = 1; level <= statistics_level; ++level) {
            const auto next = geometry.levels[level];
            statistics_image = resize_gray(statistics_image.pixels.data(),statistics_image.width,
                                           statistics_image.height,next.width,next.height);
        }
        const auto source_stats = build_log_scene_stats(statistics_image.pixels.data(),
                                                        statistics_image.pixels.size(),epsilon);
        const double sigma = prepare_local_mask_sigma(source_stats.low,source_stats.high,epsilon,sigma_mode);
        // The anchor endpoints are evaluated in f32 by 00a0c1c8/00a0b650.
        const float median = static_cast<float>(source_stats.median);
        const float low = std::max(static_cast<float>(source_stats.low),median-20.0f);
        const float high = std::min(static_cast<float>(source_stats.high),median+20.0f);
        LogSceneStats mask_stats{};
        auto collapsed = local_laplacian_gray(pixels,width,height,low,high,1.0,0.0,sigma,
                                              &mask_stats,statistics_level,epsilon);
        return PassResult{std::move(collapsed),source_stats,mask_stats,sigma,statistics_level};
    };
    auto pass = run_pass(log_gray);
    if (boundary_mode == 4 && width >= 2) {
        const auto swapped = swap_left_right_gray(log_gray,width,height);
        auto second = run_pass(swapped.pixels.data());
        const auto restored = swap_left_right_gray(second.collapsed.pixels.data(),width,height,true);
        second.collapsed = blend_horizontal_gray(pass.collapsed.pixels.data(),restored.pixels.data(),width,height);
        // 00a0855c calls 00a0889c twice with the same stats/remapper pointers.
        // Keep the second writes: neither the first stats nor blend stats.
        pass = std::move(second);
    }
    auto delta = build_delta_mask(log_gray,pass.collapsed.pixels.data(),width,height,
                                  pass.source_stats.center,pass.mask_stats.center);
    const double shift = pass.source_stats.center-pass.mask_stats.center;
    auto aligned = pass.mask_stats;
    aligned.center += shift;
    aligned.low += shift;
    aligned.high += shift;
    return {std::move(delta),pass.source_stats,pass.mask_stats,aligned,pass.sigma,pass.statistics_level};
}

}  // namespace imprint::acr
