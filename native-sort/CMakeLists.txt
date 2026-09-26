cmake_minimum_required(VERSION 3.16)
project(imprint_sort VERSION 0.1.0 LANGUAGES CXX)

include(CTest)

add_library(imprint_sort SHARED src/sort.cpp)
target_compile_features(imprint_sort PUBLIC cxx_std_17)
target_include_directories(imprint_sort PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(imprint_sort PRIVATE IMPRINT_SORT_BUILD)
set_target_properties(imprint_sort PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(BUILD_TESTING)
    add_executable(imprint_sort_tests tests/sort_tests.cpp)
    target_compile_features(imprint_sort_tests PRIVATE cxx_std_17)
    target_link_libraries(imprint_sort_tests PRIVATE imprint_sort)
    add_test(NAME native_sort_contract COMMAND imprint_sort_tests)
endif()
