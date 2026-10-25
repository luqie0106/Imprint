#include "acr_stage3_core3_green.hpp"
extern "C" int core3_green_bridge(
 const float* source, const float* p0, const float* p1, const float* p2,
 const float* p5, const float* p7, const float* p8,
 std::size_t source_stride, std::size_t workspace_stride, std::size_t width,
 std::uint32_t phase, const float* controls, float* green, float* auxiliary) {
 if (!controls) return 1;
 return static_cast<int>(imprint::acr::core3_green_prepared_float_row_0080f2b8(
 source,p0,p1,p2,p5,p7,p8,source_stride,workspace_stride,width,phase,
 {controls[0],controls[1],controls[2],controls[3]},green,auxiliary));
}
