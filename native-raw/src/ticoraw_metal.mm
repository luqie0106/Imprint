#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "ticoraw_metal.hpp"
#include "generated/ticoraw_metal_shader.hpp"

#include <algorithm>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>

namespace {

struct MetalParams {
    uint32_t rows;
    uint32_t cols;
    uint32_t lut_size;
};
static_assert(sizeof(MetalParams) == sizeof(uint32_t) * 3,
              "Metal and C++ parameter layouts must match");

struct MetalContext {
    __strong id<MTLDevice> device = nil;
    __strong id<MTLCommandQueue> queue = nil;
    __strong id<MTLLibrary> library = nil;
    __strong id<MTLComputePipelineState> step1_pipeline = nil;
    __strong id<MTLComputePipelineState> step2_pipeline = nil;
};

std::once_flag g_context_once;
std::unique_ptr<MetalContext> g_context;

bool initialize_context() {
    auto context = std::make_unique<MetalContext>();
    context->device = MTLCreateSystemDefaultDevice();
    if (!context->device) return false;

    context->queue = [context->device newCommandQueue];
    if (!context->queue) return false;

    NSString *source = [NSString stringWithUTF8String:imprint_ticoraw_metal_source];
    if (!source) return false;

    NSError *native_error = nil;
    context->library = [context->device newLibraryWithSource:source
                                                    options:nil
                                                      error:&native_error];
    if (!context->library) return false;

    id<MTLFunction> step1 = [context->library newFunctionWithName:@"step1_merge_4_to_2"];
    id<MTLFunction> step2 = [context->library newFunctionWithName:@"step2_bayer_rows"];
    if (!step1 || !step2) return false;

    context->step1_pipeline = [context->device newComputePipelineStateWithFunction:step1
                                                                             error:&native_error];
    if (!context->step1_pipeline) return false;
    context->step2_pipeline = [context->device newComputePipelineStateWithFunction:step2
                                                                             error:&native_error];
    if (!context->step2_pipeline) return false;

    g_context = std::move(context);
    return true;
}

MetalContext *get_context() {
    std::call_once(g_context_once, [] { (void)initialize_context(); });
    return g_context.get();
}

id<MTLBuffer> make_shared_buffer(id<MTLDevice> device, size_t bytes) {
    if (!device || bytes == 0) return nil;
    return [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
}

bool copy_plane_rows(id<MTLBuffer> destination,
                     const int32_t *source,
                     int rows,
                     int cols,
                     int stride) {
    if (!destination || !source || !destination.contents) return false;
    auto *dst = static_cast<int32_t *>(destination.contents);
    const size_t row_bytes = size_t(cols) * sizeof(int32_t);
    for (int r = 0; r < rows; ++r) {
        const size_t source_offset = size_t(r) * size_t(stride);
        std::memcpy(dst + size_t(r) * size_t(cols), source + source_offset, row_bytes);
    }
    return true;
}

void dispatch_1d(id<MTLComputeCommandEncoder> encoder,
                 id<MTLComputePipelineState> pipeline,
                 size_t item_count) {
    [encoder setComputePipelineState:pipeline];
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger threads = std::min<NSUInteger>(256, max_threads);
    const NSUInteger groups = (NSUInteger(item_count) + threads - 1) / threads;
    [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
             threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

bool reconstruct(const int32_t *ll,
                 const int32_t *lh,
                 const int32_t *hh,
                 const int32_t *hl,
                 int rows,
                 int cols,
                 int stride,
                 const int32_t *lut,
                 int lut_size,
                 uint16_t *out) {
    if (!ll || !lh || !hh || !hl || !lut || !out || rows <= 0 || cols <= 0 ||
        stride < cols || lut_size < kTicoRawLutSize) {
        return false;
    }

    const size_t row_count = size_t(rows);
    const size_t col_count = size_t(cols);
    if (row_count > std::numeric_limits<size_t>::max() / col_count) return false;
    const size_t count = row_count * col_count;
    // The shader uses uint indexing for both the input grid and 2x output.
    if (count > std::numeric_limits<uint32_t>::max() / 4u) return false;
    if (count > std::numeric_limits<size_t>::max() / (4u * sizeof(uint16_t)) ||
        count > std::numeric_limits<size_t>::max() / sizeof(int32_t)) {
        return false;
    }

    MetalContext *context = get_context();
    if (!context) return false;

    const size_t plane_bytes = count * sizeof(int32_t);
    const size_t output_bytes = count * 4u * sizeof(uint16_t);
    const size_t lut_bytes = size_t(kTicoRawLutSize) * sizeof(int32_t);

    id<MTLBuffer> ll_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> lh_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> hh_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> hl_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> lut_buffer = make_shared_buffer(context->device, lut_bytes);
    id<MTLBuffer> l_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> h_buffer = make_shared_buffer(context->device, plane_bytes);
    id<MTLBuffer> output_buffer = make_shared_buffer(context->device, output_bytes);
    if (!ll_buffer || !lh_buffer || !hh_buffer || !hl_buffer || !lut_buffer ||
        !l_buffer || !h_buffer || !output_buffer || !lut_buffer.contents ||
        !output_buffer.contents) {
        return false;
    }

    if (!copy_plane_rows(ll_buffer, ll, rows, cols, stride) ||
        !copy_plane_rows(lh_buffer, lh, rows, cols, stride) ||
        !copy_plane_rows(hh_buffer, hh, rows, cols, stride) ||
        !copy_plane_rows(hl_buffer, hl, rows, cols, stride)) {
        return false;
    }
    std::memcpy(lut_buffer.contents, lut, lut_bytes);

    const MetalParams params{uint32_t(rows), uint32_t(cols), uint32_t(kTicoRawLutSize)};
    id<MTLCommandBuffer> command_buffer = [context->queue commandBuffer];
    if (!command_buffer) return false;

    id<MTLComputeCommandEncoder> step1 = [command_buffer computeCommandEncoder];
    if (!step1) return false;
    [step1 setBuffer:ll_buffer offset:0 atIndex:0];
    [step1 setBuffer:lh_buffer offset:0 atIndex:1];
    [step1 setBuffer:hh_buffer offset:0 atIndex:2];
    [step1 setBuffer:hl_buffer offset:0 atIndex:3];
    [step1 setBuffer:l_buffer offset:0 atIndex:4];
    [step1 setBuffer:h_buffer offset:0 atIndex:5];
    [step1 setBytes:&params length:sizeof(params) atIndex:6];
    dispatch_1d(step1, context->step1_pipeline, count);
    [step1 endEncoding];

    id<MTLComputeCommandEncoder> step2 = [command_buffer computeCommandEncoder];
    if (!step2) return false;
    [step2 setBuffer:ll_buffer offset:0 atIndex:0];
    [step2 setBuffer:l_buffer offset:0 atIndex:1];
    [step2 setBuffer:h_buffer offset:0 atIndex:2];
    [step2 setBuffer:hl_buffer offset:0 atIndex:3];
    [step2 setBuffer:lut_buffer offset:0 atIndex:4];
    [step2 setBuffer:output_buffer offset:0 atIndex:5];
    [step2 setBytes:&params length:sizeof(params) atIndex:6];
    dispatch_1d(step2, context->step2_pipeline, count);
    [step2 endEncoding];

    [command_buffer commit];
    [command_buffer waitUntilCompleted];
    if (command_buffer.status != MTLCommandBufferStatusCompleted) return false;

    std::memcpy(out, output_buffer.contents, output_bytes);
    return true;
}

}  // namespace

extern "C" bool imprint_tico_reconstruct_metal(
    const int32_t *ll,
    const int32_t *lh,
    const int32_t *hh,
    const int32_t *hl,
    int rows,
    int cols,
    int stride,
    const int32_t *lut,
    int lut_size,
    uint16_t *out) {
    bool result = false;
    @autoreleasepool {
        @try {
            try {
                result = reconstruct(ll, lh, hh, hl, rows, cols, stride,
                                     lut, lut_size, out);
            } catch (...) {
                result = false;
            }
        } @catch (NSException *exception) {
            (void)exception;
            result = false;
        }
    }
    return result;
}
