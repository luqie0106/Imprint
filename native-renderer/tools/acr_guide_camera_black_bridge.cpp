// Test-only C ABI for the prepared Camera -> post-camera-black composition.
// Pixel math and format selection remain in the native production sources.
#include "../src/acr_guide_camera_black.hpp"
#include "../src/acr_camera_prepare.hpp"
#include "../src/acr_tone_kernels.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <vector>

namespace {

using imprint::acr::GuideCameraBlackControls;
using imprint::acr::GuideCameraBlackPlan;
using imprint::acr::GuideCameraBlackResult;

struct Snapshot {
    std::size_t height;
    std::size_t width;
    std::uint32_t camera_type;
    std::uint32_t black_type;
    std::uint8_t black_encoding;
    std::uint8_t quantized_before_black;
    std::size_t rgb_size;
    std::uint32_t pin_type;
};

void export_result(const GuideCameraBlackResult& result, Snapshot* snapshot,
                   float* rgb, std::size_t rgb_capacity) noexcept {
    snapshot->height = result.height;
    snapshot->width = result.width;
    snapshot->camera_type = result.plan.camera_type;
    snapshot->black_type = result.plan.black_type;
    snapshot->black_encoding = result.plan.black_encoding;
    snapshot->quantized_before_black =
        result.plan.quantized_before_black ? 1u : 0u;
    snapshot->rgb_size = result.rgb.size();
    snapshot->pin_type = result.plan.pin_type;
    const std::size_t copied = std::min(rgb_capacity, result.rgb.size());
    if (copied != 0u && rgb != nullptr) {
        std::copy_n(result.rgb.data(), copied, rgb);
    }
}

}  // namespace

extern "C" int acr_test_guide_camera_black(
    const float* source, std::size_t source_count,
    std::size_t source_row_stride, std::size_t height, std::size_t width,
    const double* camera_values, const std::uint32_t* order,
    std::uint32_t camera_flags, std::uint32_t input_selector,
    std::size_t old_height, std::size_t old_width,
    std::uint32_t old_camera_type, std::uint32_t old_black_type,
    std::uint8_t old_black_encoding, std::uint8_t old_quantized_before_black,
    const float* old_rgb, std::size_t old_rgb_count,
    std::uint32_t source_is_old_rgb, Snapshot* snapshot, float* result_rgb,
    std::size_t result_rgb_capacity) noexcept {
    if (camera_values == nullptr || order == nullptr || snapshot == nullptr ||
        (old_rgb_count != 0u && old_rgb == nullptr)) {
        return static_cast<int>(
            imprint::acr::GuideCameraBlackStatus::invalid_argument);
    }

    try {
        GuideCameraBlackControls controls{};
        std::copy_n(camera_values, controls.camera.size(), controls.camera.begin());
        std::copy_n(order, controls.order.size(), controls.order.begin());
        controls.camera_flags = camera_flags;
        controls.input_selector = input_selector;

        GuideCameraBlackResult output{};
        output.height = old_height;
        output.width = old_width;
        output.plan = GuideCameraBlackPlan{
            old_camera_type, old_black_type, old_black_encoding,
            old_quantized_before_black != 0u};
        output.plan.pin_type = 0xC3D4E5F6u;
        if (old_rgb_count != 0u) {
            output.rgb.assign(old_rgb, old_rgb + old_rgb_count);
        }

        const float* effective_source = source;
        if (source_is_old_rgb != 0u) {
            effective_source = output.rgb.empty() ? nullptr : output.rgb.data();
        }
        const auto status = imprint::acr::apply_guide_camera_black_prepared(
            effective_source, source_count, source_row_stride, height, width,
            controls, output);
        export_result(output, snapshot, result_rgb, result_rgb_capacity);
        return static_cast<int>(status);
    } catch (const std::bad_alloc&) {
        return static_cast<int>(
            imprint::acr::GuideCameraBlackStatus::allocation_failure);
    } catch (...) {
        return static_cast<int>(
            imprint::acr::GuideCameraBlackStatus::allocation_failure);
    }
}

extern "C" int acr_test_bind_guide_camera_black(
    const float* matrix, const float* sorted_matrix, const std::uint32_t* gains,
    const float* gain_float, const std::uint32_t* order,
    const std::uint32_t* source_fields, double* output18) noexcept {
    imprint::acr::CameraStagePrepared camera{};
    std::copy_n(matrix, 9, camera.matrix.begin());
    std::copy_n(sorted_matrix, 9, camera.sorted_matrix.begin());
    for (std::size_t i = 0; i < 3; ++i) {
        if (gains[i] > 65535u || order[i] > 255u)
            return static_cast<int>(imprint::acr::GuideCameraBlackStatus::invalid_controls);
        camera.gain_q12[i] = static_cast<std::uint16_t>(gains[i]);
        camera.gain_float[i] = gain_float[i];
        camera.sorted_plane_indices[i] = static_cast<std::uint8_t>(order[i]);
    }
    // Deliberately distinct profile compensation must not become retained b.
    camera.black_compensation = {0.25, 0.125, 0.0625};
    const imprint::acr::GuideCameraBlackSource source{
        source_fields[0], source_fields[1], source_fields[2], source_fields[3]};
    GuideCameraBlackControls output{};
    std::copy_n(output18, 13, output.camera.begin());
    output.order = {2, 0, 1};
    output.camera_flags = 0xA1;
    output.input_selector = 0xB2;
    const auto before = output;
    const auto status = imprint::acr::bind_guide_camera_black_controls_prepared(camera, source, output);
    if (status != imprint::acr::GuideCameraBlackStatus::ok) {
        if (output.camera != before.camera || output.order != before.order ||
            output.camera_flags != before.camera_flags || output.input_selector != before.input_selector)
            return -100;
        return static_cast<int>(status);
    }
    std::copy(output.camera.begin(), output.camera.end(), output18);
    for (std::size_t i = 0; i < 3; ++i) output18[13+i] = output.order[i];
    output18[16] = output.camera_flags;
    output18[17] = output.input_selector;
    return 0;
}

extern "C" int acr_test_guide_suffix(const float* rgb, std::uint32_t width,
    std::uint32_t height, double ev90, double epsilon, float* output) noexcept {
    try {
        auto result = imprint::acr::make_gray_log(rgb,width,height,ev90,epsilon);
        std::copy(result.pixels.begin(),result.pixels.end(),output);
        return 0;
    } catch (...) { return 1; }
}

extern "C" int acr_test_guide_gray_log(
    const float* source, std::size_t count, std::size_t stride,
    std::size_t height, std::size_t width, const double* camera_values,
    const std::uint32_t* order, std::uint32_t flags, std::uint32_t selector,
    double ev90, double epsilon, Snapshot* snapshot, std::uint32_t* gain,
    float* pixels, std::size_t capacity, std::uint32_t reuse_old) noexcept {
    try {
        imprint::acr::GuideGrayLogControls controls{};
        std::copy_n(camera_values,13,controls.camera.camera.begin());
        std::copy_n(order,3,controls.camera.order.begin());
        controls.camera.camera_flags = flags;
        controls.camera.input_selector = selector;
        controls.ev90 = ev90;
        controls.source_epsilon = epsilon;
        imprint::acr::GuideGrayLogResult result{};
        result.height=91; result.width=87;
        result.camera_plan={31,32,33,true,34};
        result.gain_stage_present=true;
        result.log_gray={-1234.5f,-1234.5f,-1234.5f,-1234.5f,-1234.5f};
        if (reuse_old) result.log_gray.assign(source,source+count);
        const auto status=imprint::acr::apply_guide_gray_log_prepared(
            reuse_old ? result.log_gray.data() : source,count,stride,height,width,controls,result);
        snapshot->height=result.height;
        snapshot->width=result.width;
        snapshot->camera_type=result.camera_plan.camera_type;
        snapshot->black_type=result.camera_plan.black_type;
        snapshot->black_encoding=result.camera_plan.black_encoding;
        snapshot->quantized_before_black=result.camera_plan.quantized_before_black;
        snapshot->pin_type=result.camera_plan.pin_type;
        snapshot->rgb_size=result.log_gray.size();
        *gain=result.gain_stage_present;
        std::copy_n(result.log_gray.data(),std::min(capacity,result.log_gray.size()),pixels);
        return static_cast<int>(status);
    } catch (...) { return -1; }
}

namespace {
std::array<double,70> mask_metadata(const imprint::acr::GuideLocalMaskResult& r) {
    std::array<double,70> m{};
    m[0]=r.guide.height; m[1]=r.guide.width; m[2]=r.guide.gain_stage_present;
    m[3]=r.guide.camera_plan.camera_type; m[4]=r.guide.camera_plan.black_type;
    m[5]=r.guide.camera_plan.pin_type; m[6]=r.guide.camera_plan.quantized_before_black;
    m[7]=r.guide.camera_plan.black_encoding;
    m[8]=r.geometry.level_count; m[9]=r.geometry.statistics_level;
    m[10]=r.mask.remap_sigma; m[11]=r.mask.statistics_level;
    m[12]=r.request.guide.source_epsilon; m[13]=r.request.guide.ev90;
    m[14]=r.request.sigma_mode; m[15]=r.request.boundary_mode;
    for (std::size_t i=0;i<4;++i) m[16+i]=r.request.bounds[i];
    const imprint::acr::LogSceneStats stats[]{r.mask.source_stats,r.mask.mask_stats,r.mask.aligned_mask_stats};
    for (std::size_t i=0;i<3;++i) {
        const auto& s=stats[i];
        m[20+6*i]=s.center; m[21+6*i]=s.low; m[22+6*i]=s.high;
        m[23+6*i]=s.median; m[24+6*i]=s.effective_count; m[25+6*i]=s.fallback;
    }
    m[38]=r.guide.log_gray.size(); m[39]=r.mask.delta.pixels.size();
    m[40]=r.mask.delta.height; m[41]=r.mask.delta.width;
    for (std::size_t i=0;i<13;++i) m[42+i]=r.request.guide.camera.camera[i];
    for (std::size_t i=0;i<3;++i) m[55+i]=r.request.guide.camera.order[i];
    m[58]=r.request.guide.camera.camera_flags; m[59]=r.request.guide.camera.input_selector;
    return m;
}
bool same_pixels(const std::vector<float>& a,const std::vector<float>& b) {
    return a.size()==b.size() && (a.empty() || std::memcmp(a.data(),b.data(),a.size()*sizeof(float))==0);
}
}

extern "C" int acr_test_guide_local_mask(
    const float* source,std::size_t count,std::size_t stride,const std::int32_t* bounds,
    const double* camera_values,const std::uint32_t* order,std::uint32_t flags,std::uint32_t selector,
    double ev90,double epsilon,int sigma_mode,std::uint32_t boundary_mode,std::uint32_t reuse,
    double* metadata,std::uint32_t* levels,float* log,float* delta,std::size_t capacity) noexcept {
    try {
        imprint::acr::GuideLocalMaskRequest request{};
        std::copy_n(bounds,4,request.bounds.begin());
        std::copy_n(camera_values,13,request.guide.camera.camera.begin());
        std::copy_n(order,3,request.guide.camera.order.begin());
        request.guide.camera.camera_flags=flags; request.guide.camera.input_selector=selector;
        request.guide.ev90=ev90; request.guide.source_epsilon=epsilon;
        request.sigma_mode=sigma_mode; request.boundary_mode=boundary_mode;
        imprint::acr::GuideLocalMaskResult result{};
        result.request.bounds={3,4,5,6}; result.request.guide.ev90=87.;
        result.request.guide.source_epsilon=.25; result.request.sigma_mode=19;
        result.geometry.level_count=13; result.geometry.statistics_level=7;
        for (auto& level:result.geometry.levels) level={43,47};
        result.guide.height=91; result.guide.width=87;
        result.guide.camera_plan={31,32,33,true,34}; result.guide.gain_stage_present=true;
        result.guide.log_gray.assign(5,-1234.5f);
        result.mask.delta={7,11,std::vector<float>(5,-4321.5f)};
        result.mask.source_stats={-13.,-14.,-15.,-16.,37,true};
        result.mask.mask_stats=result.mask.source_stats; result.mask.aligned_mask_stats=result.mask.source_stats;
        result.mask.remap_sigma=91.; result.mask.statistics_level=12;
        if (reuse==1) result.guide.log_gray.assign(source,source+count);
        if (reuse==2) result.mask.delta.pixels.assign(source,source+count);
        const auto before=result;
        const auto previous_metadata=mask_metadata(result);
        const auto* old_log=result.guide.log_gray.data(); const auto* old_delta=result.mask.delta.pixels.data();
        const auto log_capacity=result.guide.log_gray.capacity(),delta_capacity=result.mask.delta.pixels.capacity();
        const auto* effective= reuse==1 ? old_log : reuse==2 ? old_delta : source;
        const auto status=imprint::acr::apply_guide_local_mask_prepared(effective,count,stride,request,result);
        const auto current_metadata=mask_metadata(result);
        if (status!=imprint::acr::GuideCameraBlackStatus::ok) {
            if (current_metadata!=previous_metadata || !same_pixels(result.guide.log_gray,before.guide.log_gray) ||
                !same_pixels(result.mask.delta.pixels,before.mask.delta.pixels) ||
                old_log!=result.guide.log_gray.data() || old_delta!=result.mask.delta.pixels.data() ||
                log_capacity!=result.guide.log_gray.capacity() || delta_capacity!=result.mask.delta.pixels.capacity()) return -100;
            for (std::size_t i=0;i<16;++i)
                if (result.geometry.levels[i].width!=before.geometry.levels[i].width ||
                    result.geometry.levels[i].height!=before.geometry.levels[i].height) return -101;
        }
        std::copy(current_metadata.begin(),current_metadata.end(),metadata);
        for (std::size_t i=0;i<16;++i) {
            levels[2*i]=result.geometry.levels[i].width; levels[2*i+1]=result.geometry.levels[i].height;
        }
        std::copy_n(result.guide.log_gray.data(),std::min(capacity,result.guide.log_gray.size()),log);
        std::copy_n(result.mask.delta.pixels.data(),std::min(capacity,result.mask.delta.pixels.size()),delta);
        return static_cast<int>(status);
    } catch (...) { return -1; }
}

extern "C" int acr_test_local_tone_geometry(std::uint32_t width,std::uint32_t height,
    std::uint32_t* levels,std::uint32_t* count,std::uint32_t* selected) noexcept {
    try {
        const auto plan=imprint::acr::prepare_local_tone_geometry(width,height);
        for (std::size_t i=0;i<16;++i) {
            levels[2*i]=plan.levels[i].width;levels[2*i+1]=plan.levels[i].height;
        }
        *count=plan.level_count;*selected=plan.statistics_level;
        return 0;
    } catch (...) { return 1; }
}

extern "C" int acr_test_guide_source_route(std::uint32_t profile_present,
    std::uint32_t ad9,std::uint32_t adb,std::uint32_t presence_mask,
    std::uint32_t strength,std::uint32_t cached_level,std::uint32_t request_level,
    std::uint32_t* output) noexcept {
    if (!output || ad9>255 || adb>255 || profile_present>1 || presence_mask>63) return 1;
    imprint::acr::GuideSourceLevelInput input{};
    input.profile_pair_present=profile_present!=0;
    input.field_ad9=static_cast<std::uint8_t>(ad9);
    input.field_adb=static_cast<std::uint8_t>(adb);
    std::array<int,6> tokens{};
    for (std::size_t i=0;i<6;++i) input.level_images[i]=(presence_mask&(1u<<i)) ? &tokens[i] : nullptr;
    imprint::acr::GuideSourceLevelPlan plan{91,87,0xa5,&tokens[5],true};
    const auto old=plan;
    const auto status=imprint::acr::prepare_guide_source_level(input,plan);
    if (status!=imprint::acr::GuideCameraBlackStatus::ok) {
        if (plan.initial_level!=old.initial_level || plan.selected_level!=old.selected_level ||
            plan.tested_level_mask!=old.tested_level_mask || plan.selected_image!=old.selected_image ||
            plan.available!=old.available) return -100;
        return static_cast<int>(status);
    }
    if (plan.selected_image!=input.level_images[plan.selected_level]) return -101;
    const std::uint32_t result[]{plan.initial_level,plan.selected_level,plan.tested_level_mask,
        plan.available ? 1u : 0u,imprint::acr::select_guide_base_cache_slot(strength),
        static_cast<std::uint32_t>(imprint::acr::classify_guide_mask_transfer(cached_level,request_level))};
    std::copy_n(result,6,output);
    return 0;
}
