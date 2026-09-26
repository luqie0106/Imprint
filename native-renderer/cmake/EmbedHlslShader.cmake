if(NOT DEFINED INPUT_FILE OR NOT DEFINED OUTPUT_FILE)
    message(FATAL_ERROR "INPUT_FILE and OUTPUT_FILE are required")
endif()

file(READ "${INPUT_FILE}" shader_source)
file(WRITE "${OUTPUT_FILE}"
    "#pragma once\nnamespace imprint {\n"
    "static constexpr const char *kHlslShaderSource = R\"IMHLSL(${shader_source})IMHLSL\";\n"
    "}\n")
