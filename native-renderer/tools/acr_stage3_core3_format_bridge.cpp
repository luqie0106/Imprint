#include "acr_stage3_core3_format.hpp"
extern "C" int core3_format_u16_f32(const std::uint16_t* source, std::size_t source_count,
 float* destination, std::size_t destination_count, std::size_t width) {
 return static_cast<int>(imprint::acr::core3_u16_to_f32_prepared_row_008096e4(
 source,source_count,destination,destination_count,width));
}
extern "C" int core3_format_f32_u16(const float* source, std::size_t source_count,
 std::uint16_t* destination, std::size_t destination_count, std::size_t width) {
 return static_cast<int>(imprint::acr::core3_f32_to_u16_prepared_row_008103dc(
 source,source_count,destination,destination_count,width));
}
