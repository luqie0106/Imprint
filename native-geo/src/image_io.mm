#import <CoreGraphics/CoreGraphics.h>
#import <CoreImage/CoreImage.h>
#import <Foundation/Foundation.h>
#import <ImageIO/ImageIO.h>

#include "image_io.h"

#include <cstdio>
#include <cmath>
#include <cstring>
#include <cctype>
#include <limits>

namespace {
void set_error(char *error, uint32_t capacity, const char *message) {
    if (error == nullptr || capacity == 0) return;
    std::snprintf(error, capacity, "%s", message);
}

CGImageSourceRef make_source(const char *path, char *error, uint32_t capacity) {
    if (path == nullptr || path[0] == '\0') {
        set_error(error, capacity, "Image path is empty.");
        return nullptr;
    }
    @autoreleasepool {
        NSString *file_path = [NSString stringWithUTF8String:path];
        if (file_path == nil) {
            set_error(error, capacity, "Image path is not valid UTF-8.");
            return nullptr;
        }
        NSURL *url = [NSURL fileURLWithPath:file_path];
        CGImageSourceRef source = CGImageSourceCreateWithURL((__bridge CFURLRef)url, nullptr);
        if (source == nullptr || CGImageSourceGetCount(source) == 0) {
            if (source != nullptr) CFRelease(source);
            set_error(error, capacity, "ImageIO could not read an image from this file.");
            return nullptr;
        }
        return source;
    }
}

bool source_dimensions(CGImageSourceRef source, size_t *image_index,
                      uint32_t *width, uint32_t *height) {
    if (source == nullptr || width == nullptr || height == nullptr) return false;
    bool found = false;
    uint64_t largest_pixels = 0;
    for (size_t index = 0; index < CGImageSourceGetCount(source); ++index) {
        CFDictionaryRef properties = CGImageSourceCopyPropertiesAtIndex(source, index, nullptr);
        if (properties == nullptr) continue;
        const CFNumberRef width_value = static_cast<CFNumberRef>(
            CFDictionaryGetValue(properties, kCGImagePropertyPixelWidth));
        const CFNumberRef height_value = static_cast<CFNumberRef>(
            CFDictionaryGetValue(properties, kCGImagePropertyPixelHeight));
        int raw_width = 0;
        int raw_height = 0;
        const bool valid = width_value != nullptr && height_value != nullptr &&
            CFNumberGetValue(width_value, kCFNumberIntType, &raw_width) &&
            CFNumberGetValue(height_value, kCFNumberIntType, &raw_height) &&
            raw_width > 0 && raw_height > 0;
        if (valid) {
            const uint64_t pixels = static_cast<uint64_t>(raw_width) *
                                    static_cast<uint64_t>(raw_height);
            if (!found || pixels > largest_pixels) {
                found = true;
                largest_pixels = pixels;
                *width = static_cast<uint32_t>(raw_width);
                *height = static_cast<uint32_t>(raw_height);
                if (image_index != nullptr) *image_index = index;
            }
        }
        CFRelease(properties);
    }
    return found;
}

bool path_has_dng_extension(const char *path) {
    if (path == nullptr) return false;
    const char *dot = std::strrchr(path, '.');
    if (dot == nullptr || std::strlen(dot) != 4) return false;
    return std::tolower(static_cast<unsigned char>(dot[1])) == 'd' &&
           std::tolower(static_cast<unsigned char>(dot[2])) == 'n' &&
           std::tolower(static_cast<unsigned char>(dot[3])) == 'g' &&
           dot[4] == '\0';
}

bool raw_dimensions(const char *path, uint32_t *width, uint32_t *height,
                    char *error, uint32_t capacity) {
    if (@available(macOS 12.0, *)) {
        @autoreleasepool {
            NSString *file_path = [NSString stringWithUTF8String:path];
            if (file_path == nil) {
                set_error(error, capacity, "Image path is not valid UTF-8.");
                return false;
            }
            NSURL *url = [NSURL fileURLWithPath:file_path];
            CIRAWFilter *filter = [CIRAWFilter filterWithImageURL:url];
            if (filter == nil) {
                set_error(error, capacity, "Core Image could not decode this DNG RAW source.");
                return false;
            }
            const CGSize size = filter.nativeSize;
            if (size.width <= 0 || size.height <= 0 ||
                size.width > std::numeric_limits<uint32_t>::max() ||
                size.height > std::numeric_limits<uint32_t>::max()) {
                set_error(error, capacity, "Core Image returned invalid full-resolution DNG dimensions.");
                return false;
            }
            *width = static_cast<uint32_t>(std::llround(size.width));
            *height = static_cast<uint32_t>(std::llround(size.height));
            return *width > 0 && *height > 0;
        }
    }
    set_error(error, capacity, "Full-resolution DNG decoding requires macOS 12 or later.");
    return false;
}

bool decode_dng_rgba8_srgb(const char *path, uint8_t *rgba, uint32_t width,
                           uint32_t height, char *error, uint32_t capacity) {
    if (@available(macOS 12.0, *)) {
        @autoreleasepool {
            NSString *file_path = [NSString stringWithUTF8String:path];
            if (file_path == nil) {
                set_error(error, capacity, "Image path is not valid UTF-8.");
                return false;
            }
            NSURL *url = [NSURL fileURLWithPath:file_path];
            CIRAWFilter *filter = [CIRAWFilter filterWithImageURL:url];
            if (filter == nil) {
                set_error(error, capacity, "Core Image could not decode this DNG RAW source.");
                return false;
            }
            filter.scaleFactor = 1.0f;
            filter.draftModeEnabled = NO;
            // Orientation is applied once in Rust using the source EXIF value.
            filter.orientation = kCGImagePropertyOrientationUp;
            CIImage *image = filter.outputImage;
            if (image == nil) {
                set_error(error, capacity, "Core Image did not render a DNG RAW image.");
                return false;
            }
            const CGRect extent = image.extent;
            const uint32_t actual_width = static_cast<uint32_t>(std::llround(extent.size.width));
            const uint32_t actual_height = static_cast<uint32_t>(std::llround(extent.size.height));
            if (actual_width != width || actual_height != height) {
                set_error(error, capacity, "Core Image did not return the full-resolution DNG image.");
                return false;
            }
            CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
            if (color_space == nullptr) {
                set_error(error, capacity, "Could not create an sRGB color space.");
                return false;
            }
            NSDictionary *options = @{
                kCIContextWorkingColorSpace: (__bridge id)color_space,
                kCIContextOutputColorSpace: (__bridge id)color_space,
                kCIContextOutputPremultiplied: @NO,
                kCIContextUseSoftwareRenderer: @YES,
            };
            CIContext *context = [CIContext contextWithOptions:options];
            [context render:image
                   toBitmap:rgba
                  rowBytes:static_cast<ptrdiff_t>(width) * 4
                    bounds:extent
                    format:kCIFormatRGBA8
                 colorSpace:color_space];
            CGColorSpaceRelease(color_space);
            return true;
        }
    }
    set_error(error, capacity, "Full-resolution DNG decoding requires macOS 12 or later.");
    return false;
}
} // namespace

extern "C" int32_t imprint_imageio_dimensions(const char *path, uint32_t *width,
                                                uint32_t *height, char *error,
                                                uint32_t error_capacity) {
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    @autoreleasepool {
        if (path_has_dng_extension(path)) {
            return raw_dimensions(path, width, height, error, error_capacity) ? 0 : 1;
        }
        CGImageSourceRef source = make_source(path, error, error_capacity);
        if (source == nullptr) return 1;
        const bool valid = source_dimensions(source, nullptr, width, height);
        CFRelease(source);
        if (!valid) {
            set_error(error, error_capacity, "ImageIO could not read image dimensions.");
            return 2;
        }
        return 0;
    }
}

extern "C" int32_t imprint_imageio_decode_rgba8_srgb(
    const char *path, uint8_t *rgba, uint32_t width, uint32_t height,
    uint64_t capacity, char *error, uint32_t error_capacity) {
    if (error != nullptr && error_capacity > 0) error[0] = '\0';
    if (rgba == nullptr || width == 0 || height == 0 ||
        static_cast<uint64_t>(width) * static_cast<uint64_t>(height) >
            std::numeric_limits<uint64_t>::max() / 4 ||
        capacity < static_cast<uint64_t>(width) * height * 4) {
        set_error(error, error_capacity, "ImageIO output buffer is too small.");
        return 1;
    }

    @autoreleasepool {
        if (path_has_dng_extension(path)) {
            return decode_dng_rgba8_srgb(path, rgba, width, height, error,
                                         error_capacity) ? 0 : 3;
        }
        CGImageSourceRef source = make_source(path, error, error_capacity);
        if (source == nullptr) return 2;
        size_t image_index = 0;
        uint32_t source_width = 0;
        uint32_t source_height = 0;
        if (!source_dimensions(source, &image_index, &source_width, &source_height) ||
            source_width != width || source_height != height) {
            CFRelease(source);
            set_error(error, error_capacity, "ImageIO selected an image that is not the largest full-resolution frame.");
            return 3;
        }
        CGImageRef image = CGImageSourceCreateImageAtIndex(source, image_index, nullptr);
        CFRelease(source);
        if (image == nullptr) {
            set_error(error, error_capacity, "ImageIO could not decode the full-resolution image.");
            return 3;
        }
        if (CGImageGetWidth(image) != width || CGImageGetHeight(image) != height) {
            CFRelease(image);
            set_error(error, error_capacity, "ImageIO decoded dimensions do not match the source dimensions.");
            return 4;
        }

        CIImage *ci_image = [CIImage imageWithCGImage:image];
        const CGRect extent = ci_image.extent;
        if (std::llround(extent.size.width) != width ||
            std::llround(extent.size.height) != height) {
            CFRelease(image);
            set_error(error, error_capacity, "ImageIO decoded dimensions do not match the source dimensions.");
            return 4;
        }
        CGColorSpaceRef color_space = CGColorSpaceCreateWithName(kCGColorSpaceSRGB);
        if (color_space == nullptr) {
            CFRelease(image);
            set_error(error, error_capacity, "Could not create an sRGB color space.");
            return 5;
        }
        NSDictionary *options = @{
            kCIContextWorkingColorSpace: (__bridge id)color_space,
            kCIContextOutputColorSpace: (__bridge id)color_space,
            kCIContextOutputPremultiplied: @NO,
        };
        CIContext *context = [CIContext contextWithOptions:options];
        [context render:ci_image
               toBitmap:rgba
              rowBytes:static_cast<ptrdiff_t>(width) * 4
                bounds:extent
                format:kCIFormatRGBA8
             colorSpace:color_space];
        CGColorSpaceRelease(color_space);
        CFRelease(image);
        return 0;
    }
}
