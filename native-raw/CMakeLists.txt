cmake_minimum_required(VERSION 3.24)
project(imprint_native_raw VERSION 0.1.0 LANGUAGES CXX)
include(FetchContent)
find_package(Threads REQUIRED)
find_package(ZLIB QUIET)

# LibRaw supplies NEF metadata and RGB rendering. Imprint's own C++/Metal
# operators replace its TicoRAW entropy, inverse wavelet and CFA reconstruction.
set(IMPRINT_LIBRAW_SOURCE_DIR "" CACHE PATH "Optional preverified fixed LibRaw source tree")
if(NOT IMPRINT_LIBRAW_SOURCE_DIR)
    FetchContent_Declare(imprint_libraw_source
        URL https://codeload.github.com/pdcgomes/redlamp-libraw/tar.gz/4abfcd2ab82f2211b1f7e7c3cd83974c87e0d812
        URL_HASH SHA256=2ca1d9673531fccabcc7c8bce589444163868bcde625aa1f09eb442df1bcce2d
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
    )
    FetchContent_MakeAvailable(imprint_libraw_source)
    set(IMPRINT_LIBRAW_SOURCE_DIR "${imprint_libraw_source_SOURCE_DIR}")
endif()
if(NOT EXISTS "${IMPRINT_LIBRAW_SOURCE_DIR}/Makefile.dist")
    message(FATAL_ERROR "Fixed LibRaw source tree is missing Makefile.dist")
endif()

# Use upstream's library source manifest, without executing its build scripts.
file(STRINGS "${IMPRINT_LIBRAW_SOURCE_DIR}/Makefile.dist" make_lines)
set(object_manifest "")
set(in_objects FALSE)
foreach(line IN LISTS make_lines)
    if(line MATCHES "^LIB_OBJECTS=")
        set(in_objects TRUE)
    endif()
    if(in_objects)
        string(APPEND object_manifest " ${line}")
        if(NOT line MATCHES "\\\\$")
            set(in_objects FALSE)
        endif()
    endif()
endforeach()
string(REGEX MATCHALL "object/[A-Za-z0-9_]+\\.o" objects "${object_manifest}")
if(NOT objects)
    message(FATAL_ERROR "Could not read fixed LibRaw source manifest")
endif()
file(GLOB_RECURSE upstream_sources "${IMPRINT_LIBRAW_SOURCE_DIR}/src/*.cpp")
set(libraw_sources "")
foreach(object IN LISTS objects)
    string(REGEX REPLACE "object/(.*)\\.o" "\\1" basename "${object}")
    # Do not ship or call the public TicoRAW computation as the production path.
    if(basename MATCHES "^nikon_he_(decode|precinct_decode|gcli_decode|coefficient_decode|dequantize|tile|idwt_horizontal|idwt_vertical|bayer|predecessor)$")
        continue()
    endif()
    set(matches "")
    foreach(source IN LISTS upstream_sources)
        get_filename_component(source_basename "${source}" NAME_WE)
        if(source_basename STREQUAL basename)
            list(APPEND matches "${source}")
        endif()
    endforeach()
    list(LENGTH matches count)
    if(NOT count EQUAL 1)
        message(FATAL_ERROR "Expected one source for ${basename}, got ${count}")
    endif()
    list(APPEND libraw_sources ${matches})
endforeach()
add_library(imprint_libraw STATIC ${libraw_sources})
target_include_directories(imprint_libraw PUBLIC "${IMPRINT_LIBRAW_SOURCE_DIR}")
target_compile_features(imprint_libraw PUBLIC cxx_std_17)
target_compile_definitions(imprint_libraw PUBLIC LIBRAW_NODLL)
set_target_properties(imprint_libraw PROPERTIES POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN YES)
if(ZLIB_FOUND)
    target_compile_definitions(imprint_libraw PRIVATE USE_ZLIB)
    target_link_libraries(imprint_libraw PRIVATE ZLIB::ZLIB)
endif()

set(raw_sources src/raw_bridge.cpp src/ticoraw_decode.cpp src/ticoraw_lifting.cpp)
if(APPLE)
    enable_language(OBJCXX)
    set(shader_header "${CMAKE_CURRENT_BINARY_DIR}/generated/ticoraw_metal_shader.hpp")
    add_custom_command(OUTPUT "${shader_header}"
        COMMAND "${CMAKE_COMMAND}" -DINPUT_FILE=${CMAKE_CURRENT_SOURCE_DIR}/src/ticoraw.metal
            -DOUTPUT_FILE=${shader_header} -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/EmbedShader.cmake
        DEPENDS src/ticoraw.metal cmake/EmbedShader.cmake VERBATIM)
    list(APPEND raw_sources src/ticoraw_metal.mm "${shader_header}")
endif()
add_library(imprint_raw SHARED ${raw_sources})
target_compile_features(imprint_raw PRIVATE cxx_std_17)
target_compile_definitions(imprint_raw PRIVATE IMPRINT_RAW_BUILD)
target_include_directories(imprint_raw PUBLIC include PRIVATE src "${CMAKE_CURRENT_BINARY_DIR}"
    "${CMAKE_CURRENT_BINARY_DIR}/generated" "${IMPRINT_LIBRAW_SOURCE_DIR}/src"
    "${IMPRINT_LIBRAW_SOURCE_DIR}/src/decoders"
    "${IMPRINT_LIBRAW_SOURCE_DIR}/src/decoders/nikon_he")
target_link_libraries(imprint_raw PRIVATE imprint_libraw Threads::Threads)
set_target_properties(imprint_raw PROPERTIES CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES OBJCXX_STANDARD 17 OBJCXX_STANDARD_REQUIRED YES)
if(APPLE)
    set_source_files_properties(src/ticoraw_metal.mm PROPERTIES COMPILE_FLAGS "-fobjc-arc")
    target_link_libraries(imprint_raw PRIVATE "-framework Foundation" "-framework Metal")
endif()
install(TARGETS imprint_raw LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/imprint_raw.h DESTINATION include)
