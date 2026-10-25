cmake_minimum_required(VERSION 3.24)
project(imprint_native_renderer VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CTest)
find_package(Threads REQUIRED)

option(IMPRINT_ENABLE_CUDA "Build the CUDA GPU backend when a CUDA compiler is available" ON)

set(IMPRINT_RENDERER_SOURCES
    src/renderer.cpp
    src/dehaze_reference.cpp
    src/dense_kernels.cpp
    src/warp_kernels.cpp
    src/camera_profile_kernels.cpp
    src/acr_color_kernels.cpp
    src/acr_color_api.cpp
    src/acr_illuminant.cpp
    src/acr_illuminant_api.cpp
    src/acr_profile_tone.cpp
    src/acr_profile_table.cpp
    src/acr_profile_stage.cpp
    src/acr_profile_u16.cpp
    src/acr_profile_u16_api.cpp
    src/acr_profile_table_blend.cpp
    src/acr_huesatmap_prepare.cpp
    src/acr_huesatmap_prepare_api.cpp
    src/acr_huesatmap_gain_api.cpp
    src/acr_profile_encoding.cpp
    src/acr_profile_look_api.cpp
    src/acr_profile_tone_api.cpp
    src/acr_negative_curve.cpp
    src/acr_exposure.cpp
    src/acr_exposure_api.cpp
    src/acr_exposure_limit.cpp
    src/acr_exposure_limit_api.cpp
    src/acr_exposure_rgb.cpp
    src/acr_exposure_signed.cpp
    src/acr_exposure_signed_api.cpp
    src/acr_exposure_sdr_mask.cpp
    src/acr_exposure_sdr_mask_api.cpp
    src/acr_exposure_epsilon.cpp
    src/acr_exposure_epsilon_api.cpp
    src/acr_exposure_context_lut.cpp
    src/acr_exposure_context_lut_api.cpp
    src/acr_exposure_black_prepare.cpp
    src/acr_exposure_black_prepare_api.cpp
    src/acr_exposure_rgb_prepare.cpp
    src/acr_whites.cpp
    src/acr_exposure_anchor_api.cpp
    src/acr_source_log.cpp
    src/acr_gray.cpp
    src/acr_anchor_geometry.cpp
    src/acr_source_crop.cpp
    src/acr_source_crop_metadata.cpp
    src/acr_source_pyramid.cpp
    src/acr_fast_bayer.cpp
    src/acr_stage3_dispatch.cpp
    src/acr_stage3_core3.cpp
    src/acr_stage3_core3_colour.cpp
    src/acr_stage3_core3_green.cpp
    src/acr_stage3_core3_refine.cpp
    src/acr_stage3_core3_format.cpp
    src/acr_stage3_core3_prefix.cpp
    src/acr_stage3_core3_compose.cpp
    src/acr_stage3_balance3.cpp
    src/acr_stage3_balance3_finish.cpp
    src/acr_stage3_balance3_controls.cpp
    src/acr_bayer_gain.cpp
    src/acr_noise_transform.cpp
    src/acr_float_lut.cpp
    src/acr_profile_gain.cpp
    src/acr_profile_gain_map.cpp
    src/acr_profile_gain_cache.cpp
    src/acr_profile_gain_apply.cpp
    src/acr_profile_gain_coordinates.cpp
    src/acr_stage3_curve.cpp
    src/acr_hotpixel.cpp
    src/acr_bayer_border.cpp
    src/acr_stage3_prepared_image.cpp
    src/acr_stage3_prepared_image_api.cpp
    src/acr_gainmap.cpp
    src/acr_gainmap_api.cpp
    src/acr_warp.cpp
    src/acr_warp_api.cpp
    src/acr_buffer_format.cpp
    src/acr_buffer_format_api.cpp
    src/acr_camera_combined.cpp
    src/acr_camera_combined_api.cpp
    src/acr_postcamera_black.cpp
    src/acr_postcamera_black_api.cpp
    src/acr_u16_lut.cpp
    src/acr_source_linearization.cpp
    src/acr_resample_coefficients.cpp
    src/acr_resample_row.cpp
    src/acr_resample_vertical.cpp
    src/acr_resample_separable.cpp
    src/acr_resample_axis.cpp
    src/acr_resample_matrix.cpp
    src/acr_resample_plan.cpp
    src/acr_resample_request.cpp
    src/acr_source_resample.cpp
    src/acr_render_geometry.cpp
    src/acr_gray_log_api.cpp
    src/acr_u16_store.cpp
    src/acr_camera_prepare.cpp
    src/acr_camera_prepare_api.cpp
    src/acr_colorspec.cpp
    src/acr_colorspec_api.cpp
    src/acr_display.cpp
    src/acr_display_api.cpp
    src/acr_flare.cpp
    src/acr_flare_api.cpp
    src/acr_highlight_stat.cpp
    src/acr_post_tone_stats_api.cpp
    src/acr_post_tone_basic.cpp
    src/acr_post_tone_basic_api.cpp
    src/acr_exposure_qcap.cpp
    src/acr_exposure_qcap_api.cpp
    src/acr_exposure_optional_prepare.cpp
    src/acr_exposure_optional_prepare_api.cpp
    src/acr_tone_kernels.cpp
    src/acr_tone_mapping.cpp
    src/acr_camera_kernels.cpp
    src/acr_tone_pixels.cpp
    src/acr_tone_api.cpp
    src/backend_factory.cpp
)
set(IMPRINT_BACKEND_DEFINITIONS "")

if(APPLE)
    enable_language(OBJCXX)
    set(IMPRINT_EMBEDDED_SHADER "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_metal_shader.hpp")
    add_custom_command(
        OUTPUT "${IMPRINT_EMBEDDED_SHADER}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.metal"
                "-DOUTPUT_FILE=${IMPRINT_EMBEDDED_SHADER}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedMetalShader.cmake"
        DEPENDS src/renderer.metal cmake/EmbedMetalShader.cmake
        VERBATIM
    )
    add_custom_target(imprint_embedded_shader DEPENDS "${IMPRINT_EMBEDDED_SHADER}")
    list(APPEND IMPRINT_RENDERER_SOURCES src/metal_backend.mm)
    set(IMPRINT_BACKEND_DEFINITIONS IMPRINT_BACKEND_METAL=1)
elseif(WIN32)
    set(IMPRINT_EMBEDDED_HLSL "${CMAKE_CURRENT_BINARY_DIR}/generated/embedded_hlsl_shader.hpp")
    add_custom_command(
        OUTPUT "${IMPRINT_EMBEDDED_HLSL}"
        COMMAND "${CMAKE_COMMAND}" -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/generated"
        COMMAND "${CMAKE_COMMAND}"
                "-DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/renderer.hlsl"
                "-DOUTPUT_FILE=${IMPRINT_EMBEDDED_HLSL}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedHlslShader.cmake"
        DEPENDS src/renderer.hlsl cmake/EmbedHlslShader.cmake
        VERBATIM
    )
    add_custom_target(imprint_embedded_hlsl DEPENDS "${IMPRINT_EMBEDDED_HLSL}")
    list(APPEND IMPRINT_RENDERER_SOURCES src/d3d12_backend.cpp)
    set(IMPRINT_BACKEND_DEFINITIONS IMPRINT_BACKEND_D3D12=1)
elseif(IMPRINT_ENABLE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        set(CMAKE_CUDA_STANDARD 17)
        set(CMAKE_CUDA_STANDARD_REQUIRED ON)
        list(APPEND IMPRINT_RENDERER_SOURCES src/cuda_backend.cu)
        set(IMPRINT_BACKEND_DEFINITIONS IMPRINT_BACKEND_CUDA=1)
        message(STATUS "Building the CUDA backend with ${CMAKE_CUDA_COMPILER}")
    else()
        message(STATUS "CUDA compiler not found; building the shared ABI without a GPU backend")
    endif()
endif()

add_library(imprint_renderer SHARED ${IMPRINT_RENDERER_SOURCES})
target_compile_features(imprint_renderer PUBLIC cxx_std_17)
target_compile_definitions(imprint_renderer PRIVATE IMPRINT_RENDERER_BUILD ${IMPRINT_BACKEND_DEFINITIONS})
target_link_libraries(imprint_renderer PRIVATE Threads::Threads)
# Keep the dense CPU operators' float32 evaluation order aligned with NumPy.
if(MSVC)
    set_source_files_properties(src/acr_resample_axis.cpp src/acr_resample_matrix.cpp src/acr_resample_plan.cpp src/acr_render_geometry.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_stage3_prepared_image.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_gainmap.cpp src/acr_gainmap_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_warp.cpp src/acr_warp_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_buffer_format.cpp src/acr_buffer_format_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_profile_gain.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_profile_gain_map.cpp src/acr_profile_gain_cache.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_profile_gain_apply.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_profile_gain_coordinates.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_camera_combined.cpp src/acr_camera_combined_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_postcamera_black.cpp src/acr_postcamera_black_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_source_crop.cpp src/acr_source_pyramid.cpp src/acr_colorspec_api.cpp src/acr_source_linearization.cpp tools/acr_source_linearization_validation.cpp src/acr_stage3_core3.cpp tools/acr_stage3_core3_validation.cpp tools/acr_stage3_core3_bridge.cpp src/acr_stage3_core3_colour.cpp tools/acr_stage3_core3_colour_validation.cpp tools/acr_stage3_core3_colour_bridge.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/acr_exposure_black_prepare.cpp src/acr_exposure_context_lut.cpp src/acr_exposure_epsilon.cpp src/acr_exposure_sdr_mask.cpp src/acr_profile_u16.cpp src/acr_exposure_signed.cpp src/acr_exposure_limit.cpp src/acr_exposure_limit_api.cpp src/acr_exposure_rgb.cpp src/acr_exposure_rgb_prepare.cpp src/acr_whites.cpp src/acr_exposure_anchor_api.cpp src/acr_source_log.cpp src/acr_gray.cpp src/acr_anchor_geometry.cpp src/acr_resample_coefficients.cpp src/acr_resample_row.cpp src/acr_resample_vertical.cpp src/acr_resample_separable.cpp src/acr_gray_log_api.cpp src/acr_u16_store.cpp src/acr_illuminant.cpp src/acr_illuminant_api.cpp src/acr_profile_table.cpp src/acr_profile_stage.cpp src/acr_profile_table_blend.cpp src/acr_huesatmap_prepare.cpp src/acr_huesatmap_prepare_api.cpp src/acr_huesatmap_gain_api.cpp src/acr_profile_encoding.cpp src/acr_profile_look_api.cpp PROPERTIES COMPILE_OPTIONS "/fp:strict")
    set_source_files_properties(src/dense_kernels.cpp src/warp_kernels.cpp src/camera_profile_kernels.cpp src/acr_color_kernels.cpp src/acr_profile_tone.cpp src/acr_negative_curve.cpp src/acr_exposure.cpp src/acr_camera_prepare.cpp src/acr_colorspec.cpp src/acr_display.cpp src/acr_flare.cpp src/acr_highlight_stat.cpp src/acr_tone_kernels.cpp src/acr_tone_mapping.cpp src/acr_camera_kernels.cpp src/acr_tone_pixels.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(src/acr_resample_axis.cpp src/acr_resample_matrix.cpp src/acr_resample_plan.cpp src/acr_render_geometry.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_stage3_prepared_image.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_gainmap.cpp src/acr_gainmap_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_warp.cpp src/acr_warp_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_buffer_format.cpp src/acr_buffer_format_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_profile_gain.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_profile_gain_map.cpp src/acr_profile_gain_cache.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_profile_gain_apply.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_profile_gain_coordinates.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_camera_combined.cpp src/acr_camera_combined_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_postcamera_black.cpp src/acr_postcamera_black_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_source_crop.cpp src/acr_source_pyramid.cpp src/acr_colorspec_api.cpp src/acr_source_linearization.cpp tools/acr_source_linearization_validation.cpp src/acr_stage3_core3.cpp tools/acr_stage3_core3_validation.cpp tools/acr_stage3_core3_bridge.cpp src/acr_stage3_core3_colour.cpp tools/acr_stage3_core3_colour_validation.cpp tools/acr_stage3_core3_colour_bridge.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/acr_exposure_black_prepare.cpp src/acr_exposure_context_lut.cpp src/acr_exposure_epsilon.cpp src/acr_exposure_sdr_mask.cpp src/acr_profile_u16.cpp src/acr_exposure_signed.cpp src/acr_exposure_limit.cpp src/acr_exposure_limit_api.cpp src/acr_exposure_rgb.cpp src/acr_exposure_rgb_prepare.cpp src/acr_whites.cpp src/acr_exposure_anchor_api.cpp src/acr_source_log.cpp src/acr_gray.cpp src/acr_anchor_geometry.cpp src/acr_resample_coefficients.cpp src/acr_resample_row.cpp src/acr_resample_vertical.cpp src/acr_resample_separable.cpp src/acr_gray_log_api.cpp src/acr_u16_store.cpp src/acr_illuminant.cpp src/acr_illuminant_api.cpp src/acr_profile_table.cpp src/acr_profile_stage.cpp src/acr_profile_table_blend.cpp src/acr_huesatmap_prepare.cpp src/acr_huesatmap_prepare_api.cpp src/acr_huesatmap_gain_api.cpp src/acr_profile_encoding.cpp src/acr_profile_look_api.cpp PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
    set_source_files_properties(src/dense_kernels.cpp src/warp_kernels.cpp src/camera_profile_kernels.cpp src/acr_color_kernels.cpp src/acr_profile_tone.cpp src/acr_negative_curve.cpp src/acr_exposure.cpp src/acr_camera_prepare.cpp src/acr_colorspec.cpp src/acr_display.cpp src/acr_flare.cpp src/acr_highlight_stat.cpp src/acr_tone_kernels.cpp src/acr_tone_mapping.cpp src/acr_camera_kernels.cpp src/acr_tone_pixels.cpp PROPERTIES
        COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off"
    )
endif()
if(MSVC)
    set_source_files_properties(src/acr_stage3_core3_prefix.cpp
        src/acr_noise_transform.cpp
        tools/acr_noise_transform_validation.cpp
        src/acr_float_lut.cpp
        tools/acr_float_lut_validation.cpp
        src/acr_stage3_curve.cpp
        tools/acr_stage3_curve_validation.cpp
        src/acr_u16_lut.cpp
        tools/acr_u16_lut_validation.cpp
        src/acr_stage3_balance3_finish.cpp
        tools/acr_stage3_balance3_finish_validation.cpp
        src/acr_stage3_balance3_controls.cpp
        tools/acr_stage3_balance3_controls_validation.cpp
        src/acr_stage3_balance3.cpp
        tools/acr_stage3_balance3_validation.cpp
        src/acr_bayer_gain.cpp
        tools/acr_bayer_gain_validation.cpp
        src/acr_stage3_core3_compose.cpp
        tools/acr_stage3_core3_compose_validation.cpp
        src/acr_stage3_core3_format.cpp
        tools/acr_stage3_core3_format_validation.cpp
        src/acr_stage3_core3_refine.cpp
        tools/acr_stage3_core3_refine_validation.cpp
        src/acr_stage3_core3_green.cpp
        tools/acr_stage3_core3_green_validation.cpp
        tools/acr_stage3_core3_prefix_validation.cpp
        PROPERTIES COMPILE_OPTIONS "/fp:strict")
else()
    set_source_files_properties(src/acr_stage3_core3_prefix.cpp
        src/acr_noise_transform.cpp
        tools/acr_noise_transform_validation.cpp
        src/acr_float_lut.cpp
        tools/acr_float_lut_validation.cpp
        src/acr_stage3_curve.cpp
        tools/acr_stage3_curve_validation.cpp
        src/acr_u16_lut.cpp
        tools/acr_u16_lut_validation.cpp
        src/acr_stage3_balance3_finish.cpp
        tools/acr_stage3_balance3_finish_validation.cpp
        src/acr_stage3_balance3_controls.cpp
        tools/acr_stage3_balance3_controls_validation.cpp
        src/acr_stage3_balance3.cpp
        tools/acr_stage3_balance3_validation.cpp
        src/acr_bayer_gain.cpp
        tools/acr_bayer_gain_validation.cpp
        src/acr_stage3_core3_compose.cpp
        tools/acr_stage3_core3_compose_validation.cpp
        src/acr_stage3_core3_format.cpp
        tools/acr_stage3_core3_format_validation.cpp
        src/acr_stage3_core3_refine.cpp
        tools/acr_stage3_core3_refine_validation.cpp
        src/acr_stage3_core3_green.cpp
        tools/acr_stage3_core3_green_validation.cpp
        tools/acr_stage3_core3_prefix_validation.cpp
        PROPERTIES COMPILE_OPTIONS "-fno-fast-math;-ffp-contract=off")
endif()
target_include_directories(imprint_renderer
    PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    PRIVATE src
            "${CMAKE_CURRENT_BINARY_DIR}/generated"
)
set_target_properties(imprint_renderer PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION ${PROJECT_VERSION}
    SOVERSION 0
)

if(APPLE)
    add_dependencies(imprint_renderer imprint_embedded_shader)
    set_source_files_properties(src/metal_backend.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(imprint_renderer PRIVATE "-framework Foundation" "-framework Metal")
elseif(WIN32)
    add_dependencies(imprint_renderer imprint_embedded_hlsl)
    target_link_libraries(imprint_renderer PRIVATE d3d12 dxgi d3dcompiler)
elseif(CMAKE_CUDA_COMPILER)
    set_target_properties(imprint_renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTING)
    add_library(imprint_acr_color_test SHARED
        tools/acr_color_bridge.cpp
        src/acr_color_kernels.cpp
        src/acr_tone_kernels.cpp
        src/acr_tone_mapping.cpp
        src/acr_camera_kernels.cpp
        src/acr_tone_pixels.cpp
    )
    target_compile_features(imprint_acr_color_test PRIVATE cxx_std_17)
    target_compile_definitions(imprint_acr_color_test PRIVATE IMPRINT_RENDERER_BUILD)
    target_include_directories(imprint_acr_color_test PRIVATE include src)
    add_executable(imprint_acr_color_api_test tools/acr_color_api_validation.cpp)
    target_link_libraries(imprint_acr_color_api_test PRIVATE imprint_renderer)
    add_test(NAME native_acr_color_abi_validation COMMAND imprint_acr_color_api_test)

    add_library(imprint_dehaze_reference_test SHARED
        tests/dehaze_reference_bridge.cpp
        src/dehaze_reference.cpp
    )
    target_compile_features(imprint_dehaze_reference_test PRIVATE cxx_std_17)
    target_compile_definitions(imprint_dehaze_reference_test PRIVATE IMPRINT_RENDERER_BUILD)
    target_include_directories(imprint_dehaze_reference_test PRIVATE include src)

    add_executable(imprint_renderer_tests tests/renderer_tests.cpp)
    target_link_libraries(imprint_renderer_tests PRIVATE imprint_renderer)
    add_test(NAME native_renderer_abi_validation COMMAND imprint_renderer_tests --abi-only)
    add_test(NAME native_renderer_gpu_contract COMMAND imprint_renderer_tests)
    set_tests_properties(native_renderer_gpu_contract PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(imprint_renderer_benchmark tools/l2_benchmark.cpp)
    target_link_libraries(imprint_renderer_benchmark PRIVATE imprint_renderer)

    # Recovered aligned 2x2 fast-Bayer branch: integer accumulator, padding,
    # alias and failure-atomicity checks against an in-test reference. The
    # operator source is compiled into the test target directly because the
    # shared library keeps internal C++ symbols hidden; the Python validation
    # additionally links the official imprint_renderer object file.
    add_executable(imprint_fast_bayer_test
        tools/acr_fast_bayer_validation.cpp
        src/acr_fast_bayer.cpp
    )
    target_include_directories(imprint_fast_bayer_test PRIVATE src)
    target_compile_features(imprint_fast_bayer_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_fast_bayer_validation COMMAND imprint_fast_bayer_test)

    # Stage3 mode-1 dispatch and the closed pair-(2,2) preview branch. The
    # full-resolution pair-(1,1) branch stops explicitly at the unrecovered
    # 007f2124 cr_stage_bayer_core3 pipeline.
    add_executable(imprint_stage3_test
        tools/acr_stage3_validation.cpp
        src/acr_stage3_dispatch.cpp
        src/acr_fast_bayer.cpp
    )
    target_include_directories(imprint_stage3_test PRIVATE src)
    target_compile_features(imprint_stage3_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_validation COMMAND imprint_stage3_test)

    # Recovered cr_stage_bayer_core3 CPU row kernels FUN_0081376c (six
    # gradient signals) and FUN_00813368 (five-tap horizontal/vertical energy).
    # Full-resolution RGB composition is still unsupported; these targets only
    # validate the two recovered pixel consumers.
    add_executable(imprint_stage3_core3_test
        tools/acr_stage3_core3_validation.cpp
        src/acr_stage3_core3.cpp
    )
    target_include_directories(imprint_stage3_core3_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_validation COMMAND imprint_stage3_core3_test)

    add_library(imprint_stage3_core3_test_bridge SHARED
        tools/acr_stage3_core3_bridge.cpp
        src/acr_stage3_core3.cpp
    )
    target_include_directories(imprint_stage3_core3_test_bridge PRIVATE src)
    target_compile_features(imprint_stage3_core3_test_bridge PRIVATE cxx_std_17)

    # Recovered 008134b8 colour-difference row consumer. The native target
    # checks its status and scalar probes; the Python oracle links both this
    # source bridge and the official imprint_renderer object independently.
    add_executable(imprint_stage3_core3_colour_test
        tools/acr_stage3_core3_colour_validation.cpp
        src/acr_stage3_core3_colour.cpp
    )
    target_include_directories(imprint_stage3_core3_colour_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_colour_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_colour_validation
        COMMAND imprint_stage3_core3_colour_test)

    add_library(imprint_stage3_core3_colour_test_bridge SHARED
        tools/acr_stage3_core3_colour_bridge.cpp
        src/acr_stage3_core3_colour.cpp
    )
    target_include_directories(imprint_stage3_core3_colour_test_bridge PRIVATE src)
    target_compile_features(imprint_stage3_core3_colour_test_bridge PRIVATE cxx_std_17)

    # Recovered refinement and format conversion over prepared rows.
    add_executable(imprint_stage3_core3_format_test
        tools/acr_stage3_core3_format_validation.cpp
        src/acr_stage3_core3_format.cpp
    )
    target_include_directories(imprint_stage3_core3_format_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_format_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_format_validation
        COMMAND imprint_stage3_core3_format_test)

    add_executable(imprint_stage3_core3_refine_test
        tools/acr_stage3_core3_refine_validation.cpp
        src/acr_stage3_core3_refine.cpp
    )
    target_include_directories(imprint_stage3_core3_refine_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_refine_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_refine_validation
        COMMAND imprint_stage3_core3_refine_test)

    # Recovered green estimator over explicitly prepared float planes.
    add_executable(imprint_stage3_core3_green_test
        tools/acr_stage3_core3_green_validation.cpp
        src/acr_stage3_core3_green.cpp
    )
    target_include_directories(imprint_stage3_core3_green_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_green_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_green_validation
        COMMAND imprint_stage3_core3_green_test)

    # First two CPU consumers composed over a prepared float tile. The nine
    # workspace channels have independent plane and spatial-row strides.
    add_executable(imprint_stage3_core3_prefix_test
        tools/acr_stage3_core3_prefix_validation.cpp
        src/acr_stage3_core3_prefix.cpp
        src/acr_stage3_core3.cpp
    )
    target_include_directories(imprint_stage3_core3_prefix_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_prefix_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_prefix_validation
        COMMAND imprint_stage3_core3_prefix_test)

    # Separate recovered prepared-row balance3 and gain consumers.
    add_executable(imprint_stage3_balance3_test
        tools/acr_stage3_balance3_validation.cpp
        src/acr_stage3_balance3.cpp
    )
    target_include_directories(imprint_stage3_balance3_test PRIVATE src)
    target_compile_features(imprint_stage3_balance3_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_balance3_validation
        COMMAND imprint_stage3_balance3_test)

    add_executable(imprint_stage3_balance3_finish_test
        tools/acr_stage3_balance3_finish_validation.cpp
        src/acr_stage3_balance3_finish.cpp)
    target_include_directories(imprint_stage3_balance3_finish_test PRIVATE src)
    target_compile_features(imprint_stage3_balance3_finish_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_balance3_finish_validation
        COMMAND imprint_stage3_balance3_finish_test)

    add_executable(imprint_stage3_balance3_controls_test
        tools/acr_stage3_balance3_controls_validation.cpp
        src/acr_stage3_balance3_controls.cpp)
    target_include_directories(imprint_stage3_balance3_controls_test PRIVATE src)
    target_compile_features(imprint_stage3_balance3_controls_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_balance3_controls_validation
        COMMAND imprint_stage3_balance3_controls_test)

    add_executable(imprint_bayer_gain_test
        tools/acr_bayer_gain_validation.cpp
        src/acr_bayer_gain.cpp
    )
    target_include_directories(imprint_bayer_gain_test PRIVATE src)
    target_compile_features(imprint_bayer_gain_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_bayer_gain_validation
        COMMAND imprint_bayer_gain_test)

    add_executable(imprint_noise_transform_test
        tools/acr_noise_transform_validation.cpp
        tools/acr_noise_transform_bridge.cpp
        src/acr_noise_transform.cpp)
    target_include_directories(imprint_noise_transform_test PRIVATE src)
    target_compile_features(imprint_noise_transform_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_noise_transform_validation
        COMMAND imprint_noise_transform_test)

    add_executable(imprint_float_lut_test
        tools/acr_float_lut_validation.cpp
        tools/acr_float_lut_bridge.cpp
        src/acr_float_lut.cpp
        src/acr_noise_transform.cpp)
    target_include_directories(imprint_float_lut_test PRIVATE src)
    target_compile_features(imprint_float_lut_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_float_lut_validation COMMAND imprint_float_lut_test)

    add_executable(imprint_stage3_curve_test
        tools/acr_stage3_curve_validation.cpp
        tools/acr_stage3_curve_bridge.cpp
        src/acr_stage3_curve.cpp
        src/acr_float_lut.cpp)
    target_include_directories(imprint_stage3_curve_test PRIVATE src)
    target_compile_features(imprint_stage3_curve_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_curve_validation COMMAND imprint_stage3_curve_test)

    add_executable(imprint_hotpixel_test
        tools/acr_hotpixel_validation.cpp
        tools/acr_hotpixel_bridge.cpp
        src/acr_hotpixel.cpp)
    target_include_directories(imprint_hotpixel_test PRIVATE src)
    target_compile_features(imprint_hotpixel_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_hotpixel_validation COMMAND imprint_hotpixel_test)

    add_executable(imprint_bayer_border_test
        tools/acr_bayer_border_validation.cpp
        tools/acr_bayer_border_bridge.cpp
        src/acr_bayer_border.cpp)
    target_include_directories(imprint_bayer_border_test PRIVATE src)
    target_compile_features(imprint_bayer_border_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_bayer_border_validation COMMAND imprint_bayer_border_test)

    # Explicit prepared full-frame CPU composition, independent of host dispatch.
    add_executable(imprint_stage3_prepared_image_test
        tools/acr_stage3_prepared_image_validation.cpp
        tools/acr_stage3_prepared_image_bridge.cpp
        src/acr_stage3_prepared_image.cpp
        src/acr_bayer_border.cpp
        src/acr_bayer_gain.cpp
        src/acr_float_lut.cpp
        src/acr_u16_lut.cpp
        src/acr_stage3_curve.cpp
        src/acr_hotpixel.cpp
        src/acr_stage3_core3.cpp
        src/acr_stage3_core3_prefix.cpp
        src/acr_stage3_core3_green.cpp
        src/acr_stage3_core3_colour.cpp
        src/acr_stage3_core3_refine.cpp
        src/acr_stage3_core3_format.cpp
        src/acr_stage3_core3_compose.cpp)
    target_include_directories(imprint_stage3_prepared_image_test PRIVATE src)
    target_compile_features(imprint_stage3_prepared_image_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_prepared_image_validation COMMAND imprint_stage3_prepared_image_test)

    add_executable(imprint_gainmap_test
        tools/acr_gainmap_validation.cpp
        tools/acr_gainmap_bridge.cpp
        src/acr_gainmap.cpp)
    target_include_directories(imprint_gainmap_test PRIVATE src)
    target_compile_features(imprint_gainmap_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_gainmap_validation COMMAND imprint_gainmap_test)

    add_executable(imprint_warp_test
        tools/acr_warp_validation.cpp
        tools/acr_warp_bridge.cpp
        src/acr_warp.cpp)
    target_include_directories(imprint_warp_test PRIVATE src)
    target_compile_features(imprint_warp_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_warp_validation COMMAND imprint_warp_test)

    add_executable(imprint_buffer_format_test
        tools/acr_buffer_format_validation.cpp
        tools/acr_buffer_format_bridge.cpp
        src/acr_buffer_format.cpp)
    target_include_directories(imprint_buffer_format_test PRIVATE src)
    target_compile_features(imprint_buffer_format_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_buffer_format_validation COMMAND imprint_buffer_format_test)

    add_executable(imprint_camera_combined_test
        tools/acr_camera_combined_validation.cpp
        tools/acr_camera_combined_bridge.cpp
        src/acr_camera_combined.cpp)
    target_include_directories(imprint_camera_combined_test PRIVATE src)
    target_compile_features(imprint_camera_combined_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_camera_combined_validation COMMAND imprint_camera_combined_test)

    add_executable(imprint_postcamera_black_test
        tools/acr_postcamera_black_validation.cpp
        tools/acr_postcamera_black_bridge.cpp
        src/acr_postcamera_black.cpp
        src/acr_float_lut.cpp
        src/acr_u16_lut.cpp)
    target_include_directories(imprint_postcamera_black_test PRIVATE src)
    target_compile_features(imprint_postcamera_black_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_postcamera_black_validation COMMAND imprint_postcamera_black_test)

    add_executable(imprint_u16_lut_test
        tools/acr_u16_lut_validation.cpp
        tools/acr_u16_lut_bridge.cpp
        src/acr_float_lut.cpp
        src/acr_u16_lut.cpp)
    target_include_directories(imprint_u16_lut_test PRIVATE src)
    target_compile_features(imprint_u16_lut_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_u16_lut_validation COMMAND imprint_u16_lut_test)

    add_executable(imprint_bayer_gain_instruction_test
        tools/acr_bayer_gain_instruction_validation.cpp
        src/acr_bayer_gain.cpp
    )
    target_include_directories(imprint_bayer_gain_instruction_test PRIVATE src)
    target_compile_features(imprint_bayer_gain_instruction_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_bayer_gain_instruction_validation
        COMMAND imprint_bayer_gain_instruction_test)
    set_tests_properties(native_acr_bayer_gain_instruction_validation PROPERTIES
        SKIP_RETURN_CODE 77)

    # Explicit prepared-input composition; this does not enable full-res dispatch.
    add_executable(imprint_stage3_core3_compose_test
        tools/acr_stage3_core3_compose_validation.cpp
        src/acr_stage3_core3_compose.cpp
        src/acr_stage3_core3_format.cpp
        src/acr_stage3_core3_prefix.cpp
        src/acr_stage3_core3.cpp
        src/acr_stage3_core3_green.cpp
        src/acr_stage3_core3_colour.cpp
        src/acr_stage3_core3_refine.cpp
    )
    target_include_directories(imprint_stage3_core3_compose_test PRIVATE src)
    target_compile_features(imprint_stage3_core3_compose_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_stage3_core3_compose_validation
        COMMAND imprint_stage3_core3_compose_test)

    add_library(imprint_stage3_core3_prefix_test_bridge SHARED
        tools/acr_stage3_core3_prefix_bridge.cpp
        src/acr_stage3_core3_prefix.cpp
        src/acr_stage3_core3.cpp
    )
    target_include_directories(imprint_stage3_core3_prefix_test_bridge PRIVATE src)
    target_compile_features(imprint_stage3_core3_prefix_test_bridge PRIVATE cxx_std_17)

    add_library(imprint_stage3_test_bridge SHARED
        tools/acr_stage3_bridge.cpp
        src/acr_stage3_dispatch.cpp
        src/acr_fast_bayer.cpp
    )
    target_include_directories(imprint_stage3_test_bridge PRIVATE src)
    target_compile_features(imprint_stage3_test_bridge PRIVATE cxx_std_17)

    # Recovered dng_linearize_plane LUT build + Process for the uint16 CFA
    # Stage2 path. The operator is compiled into the test target directly (the
    # shipped library hides internal symbols); the Python oracle additionally
    # links the official imprint_renderer object file.
    add_executable(imprint_source_linearization_test
        tools/acr_source_linearization_validation.cpp
        src/acr_source_linearization.cpp
    )
    target_include_directories(imprint_source_linearization_test PRIVATE src)
    target_compile_features(imprint_source_linearization_test PRIVATE cxx_std_17)
    add_test(NAME native_acr_source_linearization_validation COMMAND imprint_source_linearization_test)
endif()

install(TARGETS imprint_renderer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/imprint_renderer.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
