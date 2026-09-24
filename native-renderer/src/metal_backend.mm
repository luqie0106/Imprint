#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "backend.hpp"
#include "embedded_metal_shader.hpp"

#include <array>
#include <cstring>
#include <sstream>

namespace imprint {

class MetalBackend final : public Backend {
public:
    explicit MetalBackend(std::string &error) {
        device_ = MTLCreateSystemDefaultDevice();
        if (!device_) {
            error = "No Metal-capable GPU is available";
            return;
        }
        queue_ = [device_ newCommandQueue];
        if (!queue_) {
            error = "Could not create a Metal command queue";
            return;
        }
        NSError *nativeError = nil;
        NSString *source = [NSString stringWithUTF8String:kMetalShaderSource];
        library_ = [device_ newLibraryWithSource:source options:nil error:&nativeError];
        if (!library_) {
            error = describe_error("Could not compile the Metal shader at runtime", nativeError);
            return;
        }
        id<MTLFunction> function = [library_ newFunctionWithName:@"render_kernel"];
        if (!function) {
            error = "Compiled Metal library does not contain render_kernel";
            return;
        }
        pipeline_ = [device_ newComputePipelineStateWithFunction:function error:&nativeError];
        if (!pipeline_) error = describe_error("Could not create the Metal compute pipeline", nativeError);
    }

    bool ready() const { return pipeline_ != nil; }
    const char *name() const override { return "Metal"; }

    bool set_images(const std::array<ImageLevel, 3> &levels, std::string &error) override {
        std::array<id<MTLBuffer>, 3> buffers{};
        for (size_t i = 0; i < levels.size(); ++i) {
            const size_t bytes = levels[i].pixels.size() * sizeof(uint16_t);
            if (!bytes) {
                error = "Preview pyramid contains an empty level";
                return false;
            }
            buffers[i] = [device_ newBufferWithBytes:levels[i].pixels.data()
                                               length:bytes
                                              options:MTLResourceStorageModeShared];
            if (!buffers[i]) {
                error = "Could not upload preview image to Metal";
                return false;
            }
        }
        source_buffers_ = buffers;
        levels_ = levels;
        return true;
    }

    bool set_filter(const im_filter_params &params,
                    const std::vector<uint16_t> &curve,
                    const std::vector<uint16_t> &lut,
                    uint32_t lut_edge,
                    std::string &error) override {
        const uint16_t dummy[] = {0, 0, 0};
        const uint16_t *lut_data = lut.empty() ? dummy : lut.data();
        const size_t lut_bytes = (lut.empty() ? 3 : lut.size()) * sizeof(uint16_t);
        id<MTLBuffer> curve_buffer = [device_ newBufferWithBytes:curve.data()
                                                          length:curve.size() * sizeof(uint16_t)
                                                         options:MTLResourceStorageModeShared];
        id<MTLBuffer> lut_buffer = [device_ newBufferWithBytes:lut_data
                                                        length:lut_bytes
                                                       options:MTLResourceStorageModeShared];
        if (!curve_buffer || !lut_buffer) {
            error = "Could not upload filter curve or LUT to Metal";
            return false;
        }
        curve_buffer_ = curve_buffer;
        lut_buffer_ = lut_buffer;
        filter_ = params;
        lut_edge_ = lut_edge;
        return true;
    }

    bool render_cached(unsigned level, const im_dehaze_params &dehaze, const im_basic_params &basic,
                       const ImageStats &stats,
                       uint16_t *destination, size_t destination_values, std::string &error) override {
        if (level >= levels_.size() || !source_buffers_[level]) {
            error = "Requested preview level has not been uploaded";
            return false;
        }
        return dispatch(source_buffers_[level], levels_[level], stats, dehaze, basic,
                        destination, destination_values, error);
    }

    bool render_full(const ImageLevel &source, const ImageStats &stats,
                     const im_dehaze_params &dehaze, const im_basic_params &basic,
                     uint16_t *destination, size_t destination_values, std::string &error) override {
        const size_t bytes = source.pixels.size() * sizeof(uint16_t);
        id<MTLBuffer> source_buffer = [device_ newBufferWithBytes:source.pixels.data()
                                                           length:bytes
                                                          options:MTLResourceStorageModeShared];
        if (!source_buffer) {
            error = "Could not upload full-resolution input to Metal";
            return false;
        }
        return dispatch(source_buffer, source, stats, dehaze, basic,
                        destination, destination_values, error);
    }

private:
    static std::string describe_error(const char *prefix, NSError *error) {
        std::ostringstream stream;
        stream << prefix;
        if (error) stream << ": " << [[error localizedDescription] UTF8String];
        return stream.str();
    }

    bool dispatch(id<MTLBuffer> source_buffer, const ImageLevel &source, const ImageStats &stats,
                  const im_dehaze_params &dehaze, const im_basic_params &basic,
                  uint16_t *destination, size_t destination_values, std::string &error) {
        const size_t pixel_count = static_cast<size_t>(source.width) * source.height;
        const size_t expected_values = pixel_count * 3;
        if (destination_values != expected_values || !destination || !curve_buffer_ || !lut_buffer_) {
            error = "Metal render buffers or output dimensions are invalid";
            return false;
        }
        id<MTLBuffer> output_buffer = [device_ newBufferWithLength:expected_values * sizeof(uint16_t)
                                                            options:MTLResourceStorageModeShared];
        if (!output_buffer) {
            error = "Could not allocate Metal output buffer";
            return false;
        }
        id<MTLCommandBuffer> command = [queue_ commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
        if (!command || !encoder) {
            error = "Could not create a Metal compute command";
            return false;
        }
        const uint32_t edge = lut_edge_;
        const uint32_t count = static_cast<uint32_t>(pixel_count);
        [encoder setComputePipelineState:pipeline_];
        [encoder setBuffer:source_buffer offset:0 atIndex:0];
        [encoder setBuffer:output_buffer offset:0 atIndex:1];
        [encoder setBytes:&dehaze length:sizeof(dehaze) atIndex:2];
        [encoder setBytes:&basic length:sizeof(basic) atIndex:3];
        [encoder setBytes:&filter_ length:sizeof(filter_) atIndex:4];
        [encoder setBytes:&stats length:sizeof(stats) atIndex:5];
        [encoder setBuffer:curve_buffer_ offset:0 atIndex:6];
        [encoder setBuffer:lut_buffer_ offset:0 atIndex:7];
        [encoder setBytes:&edge length:sizeof(edge) atIndex:8];
        [encoder setBytes:&count length:sizeof(count) atIndex:9];
        const NSUInteger width = pipeline_.threadExecutionWidth;
        const NSUInteger threads = width ? width : 256;
        [encoder dispatchThreadgroups:MTLSizeMake((pixel_count + threads - 1) / threads, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        [command commit];
        [command waitUntilCompleted];
        if (command.status != MTLCommandBufferStatusCompleted) {
            error = describe_error("Metal render failed", command.error);
            return false;
        }
        std::memcpy(destination, output_buffer.contents, expected_values * sizeof(uint16_t));
        return true;
    }

    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    id<MTLLibrary> library_ = nil;
    id<MTLComputePipelineState> pipeline_ = nil;
    std::array<id<MTLBuffer>, 3> source_buffers_{};
    id<MTLBuffer> curve_buffer_ = nil;
    id<MTLBuffer> lut_buffer_ = nil;
    std::array<ImageLevel, 3> levels_;
    im_filter_params filter_{};
    uint32_t lut_edge_ = 0;
};

std::unique_ptr<Backend> create_metal_backend(std::string &error) {
    auto backend = std::make_unique<MetalBackend>(error);
    if (!backend->ready()) return nullptr;
    return backend;
}

} // namespace imprint
