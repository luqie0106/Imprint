cmake_minimum_required(VERSION 3.24)
project(imprint_native_renderer VERSION 0.1.0 LANGUAGES CXX)

include(CheckLanguage)
include(GNUInstallDirs)
include(CTest)

option(IMPRINT_ENABLE_CUDA "Build the CUDA GPU backend when a CUDA compiler is available" ON)

set(IMPRINT_RENDERER_SOURCES
    src/renderer.cpp
    src/dehaze_reference.cpp
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
elseif(CMAKE_CUDA_COMPILER)
    set_target_properties(imprint_renderer PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(BUILD_TESTING)
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
endif()

install(TARGETS imprint_renderer
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(FILES include/imprint_renderer.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
