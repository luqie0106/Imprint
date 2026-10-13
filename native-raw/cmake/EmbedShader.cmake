file(READ "${INPUT_FILE}" shader)
get_filename_component(directory "${OUTPUT_FILE}" DIRECTORY)
file(MAKE_DIRECTORY "${directory}")
file(WRITE "${OUTPUT_FILE}" "#pragma once\nstatic const char* imprint_ticoraw_metal_source = R\"IMPRINT_TICO(\n${shader}\n)IMPRINT_TICO\";\n")
