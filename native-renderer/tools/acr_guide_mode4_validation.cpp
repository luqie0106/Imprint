#include "acr_tone_kernels.hpp"
#include "imprint_renderer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

float from_bits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

bool same_bits(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size()*sizeof(float)) == 0;
}

void saved_blend_witness() {
    // Saved original 00a09264 scalar/SIMD body, prepared two-plane buffer.
    // Endpoints deliberately differ by one ULP from copying the second input:
    // the original always subtracts and FMAs, even when the weight equals one.
    const std::array<std::uint32_t,5> first{
        1089165460,3229714479,1078113676,3234597402,1074455071};
    const std::array<std::uint32_t,5> second{
        3242708365,3230146121,1098490844,1099898319,3234786547};
    const std::array<std::uint32_t,5> expected{
        3242708366,3229930300,1078113676,1085814418,3234786546};
    std::vector<float> a,b,wanted;
    for (auto x:first) a.push_back(from_bits(x));
    for (auto x:second) b.push_back(from_bits(x));
    for (auto x:expected) wanted.push_back(from_bits(x));
    const auto actual=imprint::acr::blend_horizontal_gray(a.data(),b.data(),5,1);
    require(same_bits(actual.pixels,wanted),"saved mixed-precision blend witness mismatch");
}

void components() {
    using namespace imprint::acr;
    for (std::uint32_t width:{2u,3u,4u,5u,6u,7u,8u,9u,16u,17u,31u,128u,257u}) {
        std::vector<float> values(width*3),second(values.size());
        for (std::size_t i=0;i<values.size();++i) {
            values[i]=static_cast<float>(static_cast<int>(i%37)-18)*0.03125f;
            second[i]=static_cast<float>(static_cast<int>((i*13)%53)-26)*0.0625f;
        }
        values[0]=-0.0f;
        const auto old=values;
        const auto swapped=swap_left_right_gray(values.data(),width,3,false);
        const auto restored=swap_left_right_gray(swapped.pixels.data(),width,3,true);
        require(same_bits(restored.pixels,values),"swap inverse failed");
        for (std::uint32_t y=0;y<3;++y) for (std::uint32_t x=0;x<width;++x) {
            const auto source=values[y*width+(x+(width+1)/2)%width];
            require(std::memcmp(&source,&swapped.pixels[y*width+x],sizeof(float))==0,
                    "swap is not cyclic half-width shift");
        }
        const auto mixed=blend_horizontal_gray(values.data(),second.data(),width,3);
        const float reciprocal=1.0f/static_cast<float>(width-1);
        for (std::uint32_t x=0;x<width;++x) {
            const float t=std::clamp(static_cast<float>(x)*reciprocal,0.0f,1.0f);
            const double a=std::abs(std::fma(t,2.0f,-1.0f));
            const float weight=static_cast<float>((a*a)*std::fma(-2.0,a,3.0));
            for (std::uint32_t y=0;y<3;++y) {
                const auto i=y*width+x;
                const float want=std::fma(weight,second[i]-values[i],values[i]);
                require(std::memcmp(&want,&mixed.pixels[i],sizeof(float))==0,
                        "blend arithmetic order mismatch");
            }
        }
        require(same_bits(values,old),"component modified input");
    }
}

void masks() {
    using namespace imprint::acr;
    std::size_t different=0;
    for (std::uint32_t width:{2u,5u,6u,17u,257u}) {
        std::vector<float> log(width*3);
        for (std::uint32_t y=0;y<3;++y) for (std::uint32_t x=0;x<width;++x)
            log[y*width+x]=-10.0f+static_cast<float>((x*19+y*7)%113)*0.08f;
        const auto old=log;
        for (int sigma=0;sigma<3;++sigma) {
            const auto ordinary=build_local_tone_mask(log.data(),width,3,0x1p-12,sigma);
            const auto mode4=build_local_tone_mask(log.data(),width,3,0x1p-12,sigma,4);
            require(mode4.delta.pixels.size()==log.size(),"mask dimensions changed");
            require(mode4.statistics_level==ordinary.statistics_level,"mask statistics geometry changed");
            for (std::size_t i=0;i<log.size();++i) {
                require(std::isfinite(mode4.delta.pixels[i]) && mode4.delta.pixels[i]<=4.0f,
                        "invalid mask output");
                different += mode4.delta.pixels[i]!=ordinary.delta.pixels[i];
            }
        }
        require(same_bits(log,old),"mask modified input");
    }
    require(different>0,"mode4 was silently treated as ordinary mode");
    const float column[]{-10,-8,-4};
    const auto a=build_local_tone_mask(column,1,3,0x1p-12,0,2);
    const auto b=build_local_tone_mask(column,1,3,0x1p-12,0,4);
    require(same_bits(a.delta.pixels,b.delta.pixels),"width-one mode4 gate mismatch");
    for (std::uint32_t mode:{0u,1u,3u,5u,std::numeric_limits<std::uint32_t>::max()}) {
        bool rejected=false;
        try { (void)build_local_tone_mask(column,1,3,0x1p-12,0,mode); }
        catch (const std::invalid_argument&) { rejected=true; }
        require(rejected,"unsupported boundary accepted");
    }
}

void formal_abi() {
    std::array<float,15> source{};
    std::array<float,15> output{};
    std::array<double,20> diagnostics{};
    for (std::size_t i=0;i<source.size();++i) source[i]=-8.0f+static_cast<float>(i)*0.25f;
    output.fill(-9123.25f);
    diagnostics.fill(-1234.5);
    const auto old_output=output;
    const auto old_diagnostics=diagnostics;
    for (std::uint32_t boundary:{0u,1u,3u,5u}) {
        require(im_native_acr_guide_mask_prepared_f32(5,3,source.data(),15,0x1p-12,0,boundary,
            output.data(),15,diagnostics.data(),20)==IM_STATUS_INVALID_ARGUMENT,"ABI boundary accepted");
        require(output==old_output && diagnostics==old_diagnostics,"ABI failed partial commit");
    }
    require(im_native_acr_guide_mask_prepared_f32(5,3,source.data(),15,0x1p-12,0,4,
        source.data(),15,diagnostics.data(),20)==IM_STATUS_INVALID_ARGUMENT,"ABI overlap accepted");
    source.back()=std::numeric_limits<float>::quiet_NaN();
    require(im_native_acr_guide_mask_prepared_f32(5,3,source.data(),15,0x1p-12,0,4,
        output.data(),15,diagnostics.data(),20)==IM_STATUS_INVALID_ARGUMENT,"ABI late NaN accepted");
    require(output==old_output && diagnostics==old_diagnostics,"ABI late failure partial commit");
    source.back()=-4.5f;
    require(im_native_acr_guide_mask_prepared_f32(5,3,source.data(),15,0x1p-12,0,4,
        output.data(),15,diagnostics.data(),20)==IM_STATUS_OK,"ABI mode4 failed");
    for (auto value:output) require(std::isfinite(value) && value<=4.0f,"ABI invalid delta");
}
} // namespace

int main() {
    try {
        saved_blend_witness();
        components();
        masks();
        formal_abi();
        std::cout<<"mode4: saved-word blend, 13 swap/blend widths, 15 double-pass masks, narrow gate, ABI atomicity and invalid modes passed\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr<<e.what()<<'\n';
        return 1;
    }
}
