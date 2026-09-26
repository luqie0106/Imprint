#define NOMINMAX
#include <windows.h>
#include <d3d12.h>
#include <d3dcompiler.h>
#include <dxgi1_4.h>
#include <wrl/client.h>

#include "backend.hpp"
#include "embedded_hlsl_shader.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace imprint {
namespace {

using Microsoft::WRL::ComPtr;

std::string hresult_text(const char *operation, HRESULT result) {
    std::ostringstream stream;
    stream << operation << " failed (HRESULT 0x" << std::hex
           << static_cast<unsigned long>(result) << ")";
    return stream.str();
}

bool record_failure(std::string &error, const char *operation, HRESULT result) {
    error = hresult_text(operation, result);
    return false;
}

UINT64 aligned_buffer_size(size_t bytes) {
    if (bytes > std::numeric_limits<UINT64>::max() - 3) return 0;
    const UINT64 aligned = (static_cast<UINT64>(bytes) + 3u) & ~UINT64(3u);
    return std::max<UINT64>(aligned, 4u);
}

D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES properties{};
    properties.Type = type;
    properties.CPUPageProperty = D3D12_CPU_PAGE_PROPERTY_UNKNOWN;
    properties.MemoryPoolPreference = D3D12_MEMORY_POOL_UNKNOWN;
    properties.CreationNodeMask = 1;
    properties.VisibleNodeMask = 1;
    return properties;
}

D3D12_RESOURCE_DESC buffer_description(UINT64 bytes, D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC description{};
    description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    description.Alignment = 0;
    description.Width = bytes;
    description.Height = 1;
    description.DepthOrArraySize = 1;
    description.MipLevels = 1;
    description.Format = DXGI_FORMAT_UNKNOWN;
    description.SampleDesc.Count = 1;
    description.SampleDesc.Quality = 0;
    description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    description.Flags = flags;
    return description;
}

struct alignas(16) RenderConstants {
    float dehaze0[4];
    float dehaze1[4];
    float dehaze2[4];
    float basic0[4];
    float basic1[4];
    float filter0[4];
    float filter1[4];
    float stats0[4];
    float stats1[4];
    uint32_t dispatch[4];
};

static_assert(sizeof(RenderConstants) == 40 * sizeof(uint32_t),
              "RenderConstants must match the HLSL 16-byte cbuffer packing");

RenderConstants make_constants(const im_dehaze_params &dehaze,
                               const im_basic_params &basic,
                               const im_filter_params &filter,
                               const ImageStats &stats,
                               uint32_t lut_edge,
                               uint32_t pixels,
                               uint32_t row_stride,
                               bool identity_copy) {
    RenderConstants result{};
    result.dehaze0[0] = dehaze.strength;
    result.dehaze0[1] = dehaze.naturalness;
    result.dehaze0[2] = dehaze.fog_retention;
    result.dehaze0[3] = dehaze.local_contrast;
    result.dehaze1[0] = dehaze.color_recovery;
    result.dehaze1[1] = dehaze.color_protection;
    result.dehaze1[2] = dehaze.highlight_protection;
    result.dehaze1[3] = dehaze.shadow_protection;
    result.dehaze2[0] = dehaze.brightness_protection;
    result.basic0[0] = basic.exposure;
    result.basic0[1] = basic.contrast;
    result.basic0[2] = basic.highlights;
    result.basic0[3] = basic.shadows;
    result.basic1[0] = basic.whites;
    result.basic1[1] = basic.blacks;
    result.basic1[2] = basic.vibrance;
    result.basic1[3] = basic.saturation;
    result.filter0[0] = filter.curve_mix;
    result.filter0[1] = filter.lut_mix;
    result.filter0[2] = filter.exposure_ev;
    result.filter0[3] = filter.contrast;
    result.filter1[0] = filter.saturation;
    result.filter1[1] = filter.warmth;
    result.filter1[2] = filter.tint;
    result.filter1[3] = filter.fade;
    result.stats0[0] = stats.mean_luma;
    result.stats0[1] = stats.max_luma;
    result.stats0[2] = stats.air_r;
    result.stats0[3] = stats.air_g;
    result.stats1[0] = stats.air_b;
    result.stats1[1] = stats.haze_level;
    result.stats1[2] = stats.brightness_gain;
    result.dispatch[0] = lut_edge;
    result.dispatch[1] = pixels;
    result.dispatch[2] = identity_copy ? 1u : 0u;
    result.dispatch[3] = row_stride;
    return result;
}

bool neutral_basic(const im_basic_params &basic) {
    return basic.exposure == 0.0f && basic.contrast == 0.0f &&
           basic.highlights == 0.0f && basic.shadows == 0.0f &&
           basic.whites == 0.0f && basic.blacks == 0.0f &&
           basic.vibrance == 0.0f && basic.saturation == 0.0f;
}

bool identity_render(const im_dehaze_params &dehaze,
                     const im_basic_params &basic,
                     const im_filter_params &filter,
                     uint32_t lut_edge) {
    return dehaze.strength <= 1e-6f && neutral_basic(basic) &&
           filter.curve_mix == 0.0f &&
           (lut_edge < 2 || filter.lut_mix == 0.0f) &&
           filter.exposure_ev == 0.0f && filter.contrast == 1.0f &&
           filter.saturation == 1.0f && filter.warmth == 0.0f &&
           filter.tint == 0.0f && filter.fade == 0.0f;
}

} // namespace

class D3D12Backend final : public Backend {
public:
    explicit D3D12Backend(std::string &error) {
        if (!initialize_device(error) || !initialize_pipeline(error) || !initialize_commands(error)) {
            return;
        }
        ready_ = true;
    }

    ~D3D12Backend() override {
        if (fence_event_) CloseHandle(fence_event_);
    }

    bool ready() const { return ready_; }
    const char *name() const override { return "D3D12"; }

    bool set_images(const std::array<ImageLevel, 3> &levels, std::string &error) override {
        std::array<ComPtr<ID3D12Resource>, 3> next_resources{};
        std::array<ComPtr<ID3D12Resource>, 3> staging{};
        for (size_t i = 0; i < levels.size(); ++i) {
            const uint64_t expected = static_cast<uint64_t>(levels[i].width) * levels[i].height * 3;
            if (!levels[i].width || !levels[i].height || expected != levels[i].pixels.size()) {
                error = "Preview pyramid contains an invalid image level";
                return false;
            }
            const size_t bytes = levels[i].pixels.size() * sizeof(uint16_t);
            if (!create_gpu_buffer(bytes, D3D12_RESOURCE_FLAG_NONE,
                                   D3D12_RESOURCE_STATE_COPY_DEST, next_resources[i], error,
                                   "Could not allocate D3D12 preview buffer") ||
                !create_staging_buffer(levels[i].pixels.data(), bytes, staging[i], error,
                                       "Could not prepare D3D12 preview upload")) {
                return false;
            }
        }

        if (!begin_commands(error)) return false;
        for (size_t i = 0; i < levels.size(); ++i) {
            command_list_->CopyBufferRegion(next_resources[i].Get(), 0, staging[i].Get(), 0,
                                             aligned_buffer_size(levels[i].pixels.size() * sizeof(uint16_t)));
            transition(next_resources[i].Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                       D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        }
        if (!submit_and_wait(error, "Could not upload D3D12 preview images")) return false;

        sources_ = std::move(next_resources);
        for (size_t i = 0; i < levels.size(); ++i) {
            widths_[i] = levels[i].width;
            heights_[i] = levels[i].height;
        }
        return true;
    }

    bool set_filter(const im_filter_params &params,
                    const std::vector<uint16_t> &curve,
                    const std::vector<uint16_t> &lut,
                    uint32_t lut_edge,
                    std::string &error) override {
        if (curve.size() != 3 * 256) {
            error = "D3D12 filter curve must contain three 256-value channels";
            return false;
        }
        if ((lut_edge < 2 && !lut.empty()) || (lut_edge >= 2 &&
            static_cast<uint64_t>(lut_edge) * lut_edge * lut_edge * 3 != lut.size())) {
            error = "D3D12 filter LUT size does not match its edge length";
            return false;
        }

        const uint16_t dummy_lut[3] = {0, 0, 0};
        const uint16_t *lut_data = lut.empty() ? dummy_lut : lut.data();
        const size_t lut_values = lut.empty() ? 3 : lut.size();
        ComPtr<ID3D12Resource> next_curve;
        ComPtr<ID3D12Resource> next_lut;
        ComPtr<ID3D12Resource> curve_staging;
        ComPtr<ID3D12Resource> lut_staging;
        if (!create_gpu_buffer(curve.size() * sizeof(uint16_t), D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST, next_curve, error,
                               "Could not allocate D3D12 curve buffer") ||
            !create_gpu_buffer(lut_values * sizeof(uint16_t), D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST, next_lut, error,
                               "Could not allocate D3D12 LUT buffer") ||
            !create_staging_buffer(curve.data(), curve.size() * sizeof(uint16_t), curve_staging,
                                   error, "Could not prepare D3D12 curve upload") ||
            !create_staging_buffer(lut_data, lut_values * sizeof(uint16_t), lut_staging,
                                   error, "Could not prepare D3D12 LUT upload")) {
            return false;
        }

        if (!begin_commands(error)) return false;
        command_list_->CopyBufferRegion(next_curve.Get(), 0, curve_staging.Get(), 0,
                                         aligned_buffer_size(curve.size() * sizeof(uint16_t)));
        command_list_->CopyBufferRegion(next_lut.Get(), 0, lut_staging.Get(), 0,
                                         aligned_buffer_size(lut_values * sizeof(uint16_t)));
        transition(next_curve.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        transition(next_lut.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (!submit_and_wait(error, "Could not upload D3D12 filter resources")) return false;

        curve_ = std::move(next_curve);
        lut_ = std::move(next_lut);
        filter_ = params;
        lut_edge_ = lut_edge;
        return true;
    }

    bool render_cached(unsigned level, const im_dehaze_params &dehaze,
                       const im_basic_params &basic, const ImageStats &stats,
                       uint16_t *destination, size_t destination_values,
                       std::string &error) override {
        if (level >= sources_.size() || !sources_[level]) {
            error = "Requested preview level has not been uploaded";
            return false;
        }
        return dispatch(sources_[level].Get(), widths_[level], heights_[level], stats,
                        dehaze, basic, destination, destination_values, error);
    }

    bool render_full(const ImageLevel &source, const ImageStats &stats,
                     const im_dehaze_params &dehaze, const im_basic_params &basic,
                     uint16_t *destination, size_t destination_values,
                     std::string &error) override {
        if (!source.width || !source.height || source.pixels.empty()) {
            error = "Full-resolution D3D12 input is empty";
            return false;
        }
        ComPtr<ID3D12Resource> gpu_source;
        ComPtr<ID3D12Resource> staging;
        const size_t bytes = source.pixels.size() * sizeof(uint16_t);
        if (!create_gpu_buffer(bytes, D3D12_RESOURCE_FLAG_NONE,
                               D3D12_RESOURCE_STATE_COPY_DEST, gpu_source, error,
                               "Could not allocate D3D12 full-resolution input") ||
            !create_staging_buffer(source.pixels.data(), bytes, staging, error,
                                   "Could not prepare D3D12 full-resolution upload")) {
            return false;
        }
        if (!begin_commands(error)) return false;
        command_list_->CopyBufferRegion(gpu_source.Get(), 0, staging.Get(), 0,
                                         aligned_buffer_size(bytes));
        transition(gpu_source.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
        if (!submit_and_wait(error, "Could not upload full-resolution input to D3D12")) return false;
        return dispatch(gpu_source.Get(), source.width, source.height, stats,
                        dehaze, basic, destination, destination_values, error);
    }

private:
    bool initialize_device(std::string &error) {
        ComPtr<IDXGIFactory4> factory;
        HRESULT result = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
        if (FAILED(result)) return record_failure(error, "Could not create a DXGI factory", result);

        for (UINT index = 0;; ++index) {
            ComPtr<IDXGIAdapter1> candidate;
            result = factory->EnumAdapters1(index, &candidate);
            if (result == DXGI_ERROR_NOT_FOUND) break;
            if (FAILED(result)) return record_failure(error, "Could not enumerate DXGI adapters", result);
            DXGI_ADAPTER_DESC1 description{};
            if (FAILED(candidate->GetDesc1(&description)) || (description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) {
                continue;
            }
            ComPtr<ID3D12Device> candidate_device;
            result = D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_11_0,
                                       IID_PPV_ARGS(&candidate_device));
            if (SUCCEEDED(result)) {
                adapter_ = std::move(candidate);
                device_ = std::move(candidate_device);
                break;
            }
        }
        if (!device_) {
            error = "No hardware D3D12-capable GPU is available";
            return false;
        }
        return true;
    }

    bool initialize_pipeline(std::string &error) {
        D3D12_ROOT_PARAMETER parameters[5]{};
        parameters[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
        parameters[0].Constants.ShaderRegister = 0;
        parameters[0].Constants.RegisterSpace = 0;
        parameters[0].Constants.Num32BitValues = 40;
        parameters[0].ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

        const UINT shader_registers[4] = {0, 0, 1, 2};
        const D3D12_ROOT_PARAMETER_TYPE types[4] = {
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            D3D12_ROOT_PARAMETER_TYPE_UAV,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
            D3D12_ROOT_PARAMETER_TYPE_SRV,
        };
        for (UINT i = 0; i < 4; ++i) {
            D3D12_ROOT_PARAMETER &parameter = parameters[i + 1];
            parameter.ParameterType = types[i];
            parameter.Descriptor.ShaderRegister = shader_registers[i];
            parameter.Descriptor.RegisterSpace = 0;
            parameter.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;
        }
        D3D12_ROOT_SIGNATURE_DESC root_description{};
        root_description.NumParameters = static_cast<UINT>(std::size(parameters));
        root_description.pParameters = parameters;
        root_description.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;
        ComPtr<ID3DBlob> serialized;
        ComPtr<ID3DBlob> serialization_error;
        HRESULT result = D3D12SerializeRootSignature(&root_description,
                                                     D3D_ROOT_SIGNATURE_VERSION_1,
                                                     &serialized, &serialization_error);
        if (FAILED(result)) {
            std::string detail = hresult_text("Could not serialize the D3D12 root signature", result);
            if (serialization_error && serialization_error->GetBufferPointer()) {
                detail += ": ";
                detail.append(static_cast<const char *>(serialization_error->GetBufferPointer()),
                              serialization_error->GetBufferSize());
            }
            error = std::move(detail);
            return false;
        }
        result = device_->CreateRootSignature(0, serialized->GetBufferPointer(),
                                              serialized->GetBufferSize(),
                                              IID_PPV_ARGS(&root_signature_));
        if (FAILED(result)) return record_failure(error, "Could not create the D3D12 root signature", result);

        ComPtr<ID3DBlob> shader;
        ComPtr<ID3DBlob> compile_error;
        result = D3DCompile(kHlslShaderSource, std::strlen(kHlslShaderSource), "renderer.hlsl",
                            nullptr, nullptr, "render_kernel", "cs_5_0",
                            D3DCOMPILE_ENABLE_STRICTNESS | D3DCOMPILE_OPTIMIZATION_LEVEL3,
                            0, &shader, &compile_error);
        if (FAILED(result)) {
            std::string detail = hresult_text("Could not compile the D3D12 HLSL shader", result);
            if (compile_error && compile_error->GetBufferPointer()) {
                detail += ": ";
                detail.append(static_cast<const char *>(compile_error->GetBufferPointer()),
                              compile_error->GetBufferSize());
            }
            error = std::move(detail);
            return false;
        }

        D3D12_COMPUTE_PIPELINE_STATE_DESC pipeline_description{};
        pipeline_description.pRootSignature = root_signature_.Get();
        pipeline_description.CS.pShaderBytecode = shader->GetBufferPointer();
        pipeline_description.CS.BytecodeLength = shader->GetBufferSize();
        result = device_->CreateComputePipelineState(&pipeline_description,
                                                      IID_PPV_ARGS(&pipeline_));
        if (FAILED(result)) return record_failure(error, "Could not create the D3D12 compute pipeline", result);
        return true;
    }

    bool initialize_commands(std::string &error) {
        D3D12_COMMAND_QUEUE_DESC queue_description{};
        queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
        queue_description.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
        queue_description.Flags = D3D12_COMMAND_QUEUE_FLAG_NONE;
        queue_description.NodeMask = 0;
        HRESULT result = device_->CreateCommandQueue(&queue_description, IID_PPV_ARGS(&queue_));
        if (FAILED(result)) return record_failure(error, "Could not create a D3D12 command queue", result);
        result = device_->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                 IID_PPV_ARGS(&command_allocator_));
        if (FAILED(result)) return record_failure(error, "Could not create a D3D12 command allocator", result);
        result = device_->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                           command_allocator_.Get(), pipeline_.Get(),
                                           IID_PPV_ARGS(&command_list_));
        if (FAILED(result)) return record_failure(error, "Could not create a D3D12 command list", result);
        result = command_list_->Close();
        if (FAILED(result)) return record_failure(error, "Could not close the initial D3D12 command list", result);
        result = device_->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
        if (FAILED(result)) return record_failure(error, "Could not create a D3D12 fence", result);
        fence_event_ = CreateEventA(nullptr, FALSE, FALSE, nullptr);
        if (!fence_event_) {
            error = "Could not create a D3D12 fence event (Win32 error " +
                    std::to_string(GetLastError()) + ")";
            return false;
        }
        return true;
    }

    bool create_gpu_buffer(size_t bytes, D3D12_RESOURCE_FLAGS flags,
                           D3D12_RESOURCE_STATES initial_state,
                           ComPtr<ID3D12Resource> &resource, std::string &error,
                           const char *operation) {
        const UINT64 width = aligned_buffer_size(bytes);
        if (!width) {
            error = std::string(operation) + ": buffer size overflow";
            return false;
        }
        const D3D12_HEAP_PROPERTIES heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_RESOURCE_DESC description = buffer_description(width, flags);
        const HRESULT result = device_->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &description, initial_state, nullptr,
            IID_PPV_ARGS(&resource));
        if (FAILED(result)) return record_failure(error, operation, result);
        return true;
    }

    bool create_staging_buffer(const uint16_t *source, size_t bytes,
                               ComPtr<ID3D12Resource> &resource, std::string &error,
                               const char *operation) {
        const UINT64 width = aligned_buffer_size(bytes);
        if (!width || !source) {
            error = std::string(operation) + ": invalid source or buffer size";
            return false;
        }
        const D3D12_HEAP_PROPERTIES heap = heap_properties(D3D12_HEAP_TYPE_UPLOAD);
        const D3D12_RESOURCE_DESC description = buffer_description(width, D3D12_RESOURCE_FLAG_NONE);
        HRESULT result = device_->CreateCommittedResource(
            &heap, D3D12_HEAP_FLAG_NONE, &description, D3D12_RESOURCE_STATE_GENERIC_READ,
            nullptr, IID_PPV_ARGS(&resource));
        if (FAILED(result)) return record_failure(error, operation, result);
        D3D12_RANGE no_read{0, 0};
        void *mapped = nullptr;
        result = resource->Map(0, &no_read, &mapped);
        if (FAILED(result)) return record_failure(error, operation, result);
        std::memcpy(mapped, source, bytes);
        if (width > bytes) std::memset(static_cast<uint8_t *>(mapped) + bytes, 0, width - bytes);
        resource->Unmap(0, nullptr);
        return true;
    }

    bool begin_commands(std::string &error) {
        HRESULT result = command_allocator_->Reset();
        if (FAILED(result)) return record_failure(error, "Could not reset the D3D12 command allocator", result);
        result = command_list_->Reset(command_allocator_.Get(), pipeline_.Get());
        if (FAILED(result)) return record_failure(error, "Could not reset the D3D12 command list", result);
        return true;
    }

    void transition(ID3D12Resource *resource,
                    D3D12_RESOURCE_STATES before,
                    D3D12_RESOURCE_STATES after) {
        D3D12_RESOURCE_BARRIER barrier{};
        barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        barrier.Flags = D3D12_RESOURCE_BARRIER_FLAG_NONE;
        barrier.Transition.pResource = resource;
        barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        barrier.Transition.StateBefore = before;
        barrier.Transition.StateAfter = after;
        command_list_->ResourceBarrier(1, &barrier);
    }

    bool submit_and_wait(std::string &error, const char *operation) {
        HRESULT result = command_list_->Close();
        if (FAILED(result)) return record_failure(error, operation, result);
        ID3D12CommandList *lists[] = {command_list_.Get()};
        queue_->ExecuteCommandLists(1, lists);
        const UINT64 target = ++fence_value_;
        result = queue_->Signal(fence_.Get(), target);
        if (FAILED(result)) return record_failure(error, operation, result);
        if (fence_->GetCompletedValue() < target) {
            result = fence_->SetEventOnCompletion(target, fence_event_);
            if (FAILED(result)) return record_failure(error, operation, result);
            const DWORD wait_result = WaitForSingleObject(fence_event_, INFINITE);
            if (wait_result != WAIT_OBJECT_0) {
                error = std::string(operation) + ": waiting for the GPU failed (Win32 error " +
                        std::to_string(GetLastError()) + ")";
                return false;
            }
        }
        result = device_->GetDeviceRemovedReason();
        if (FAILED(result)) return record_failure(error, operation, result);
        return true;
    }

    bool ensure_output_capacity(size_t output_bytes, std::string &error) {
        if (output_bytes <= output_capacity_) return true;
        ComPtr<ID3D12Resource> next_output;
        ComPtr<ID3D12Resource> next_readback;
        const UINT64 width = aligned_buffer_size(output_bytes);
        if (!width) {
            error = "D3D12 output buffer size overflow";
            return false;
        }
        const D3D12_HEAP_PROPERTIES default_heap = heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        const D3D12_RESOURCE_DESC output_description = buffer_description(width, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        HRESULT result = device_->CreateCommittedResource(
            &default_heap, D3D12_HEAP_FLAG_NONE, &output_description,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, nullptr, IID_PPV_ARGS(&next_output));
        if (FAILED(result)) return record_failure(error, "Could not allocate D3D12 output buffer", result);

        const D3D12_HEAP_PROPERTIES readback_heap = heap_properties(D3D12_HEAP_TYPE_READBACK);
        const D3D12_RESOURCE_DESC readback_description = buffer_description(width, D3D12_RESOURCE_FLAG_NONE);
        result = device_->CreateCommittedResource(
            &readback_heap, D3D12_HEAP_FLAG_NONE, &readback_description,
            D3D12_RESOURCE_STATE_COPY_DEST, nullptr, IID_PPV_ARGS(&next_readback));
        if (FAILED(result)) return record_failure(error, "Could not allocate D3D12 readback buffer", result);
        output_ = std::move(next_output);
        readback_ = std::move(next_readback);
        output_capacity_ = static_cast<size_t>(width);
        return true;
    }

    bool dispatch(ID3D12Resource *source, uint32_t width, uint32_t height,
                  const ImageStats &stats, const im_dehaze_params &dehaze,
                  const im_basic_params &basic, uint16_t *destination,
                  size_t destination_values, std::string &error) {
        const uint64_t pixels64 = static_cast<uint64_t>(width) * height;
        const uint64_t values64 = pixels64 * 3;
        if (!source || !width || !height || pixels64 > std::numeric_limits<uint32_t>::max() ||
            values64 != destination_values || !destination || !curve_ || !lut_) {
            error = "D3D12 render buffers or output dimensions are invalid";
            return false;
        }
        if (values64 > std::numeric_limits<size_t>::max() / sizeof(uint32_t)) {
            error = "D3D12 output size overflow";
            return false;
        }
        const size_t output_bytes = static_cast<size_t>(values64) * sizeof(uint32_t);
        if (!ensure_output_capacity(output_bytes, error) || !begin_commands(error)) return false;

        const bool exact_copy = identity_render(dehaze, basic, filter_, lut_edge_);
        constexpr uint64_t max_groups_per_axis = 65535;
        const uint64_t needed_groups = (pixels64 + 255) / 256;
        const UINT groups_x = static_cast<UINT>(std::min(needed_groups, max_groups_per_axis));
        if (!groups_x) {
            error = "D3D12 dispatch has no thread groups";
            return false;
        }
        const uint64_t groups_y64 = (needed_groups + groups_x - 1) / groups_x;
        if (!groups_y64 || groups_y64 > max_groups_per_axis) {
            error = "D3D12 dispatch exceeds the supported two-dimensional grid size";
            return false;
        }
        const UINT groups_y = static_cast<UINT>(groups_y64);
        const uint32_t row_stride = groups_x * 256u;
        const RenderConstants constants = make_constants(dehaze, basic, filter_, stats,
                                                         lut_edge_, static_cast<uint32_t>(pixels64),
                                                         row_stride, exact_copy);
        command_list_->SetComputeRootSignature(root_signature_.Get());
        command_list_->SetPipelineState(pipeline_.Get());
        command_list_->SetComputeRoot32BitConstants(0, 40, &constants, 0);
        command_list_->SetComputeRootShaderResourceView(1, source->GetGPUVirtualAddress());
        command_list_->SetComputeRootUnorderedAccessView(2, output_->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(3, curve_->GetGPUVirtualAddress());
        command_list_->SetComputeRootShaderResourceView(4, lut_->GetGPUVirtualAddress());
        command_list_->Dispatch(groups_x, groups_y, 1);
        transition(output_.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_SOURCE);
        command_list_->CopyBufferRegion(readback_.Get(), 0, output_.Get(), 0, output_bytes);
        transition(output_.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        if (!submit_and_wait(error, "D3D12 render failed")) return false;

        void *mapped_buffer = nullptr;
        D3D12_RANGE read_range{0, output_bytes};
        HRESULT result = readback_->Map(0, &read_range, &mapped_buffer);
        if (FAILED(result)) return record_failure(error, "Could not map D3D12 render output", result);
        const uint32_t *mapped = static_cast<const uint32_t *>(mapped_buffer);
        for (size_t i = 0; i < destination_values; ++i) {
            destination[i] = static_cast<uint16_t>(mapped[i]);
        }
        D3D12_RANGE no_write{0, 0};
        readback_->Unmap(0, &no_write);
        return true;
    }

    bool ready_ = false;
    ComPtr<IDXGIAdapter1> adapter_;
    ComPtr<ID3D12Device> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12CommandAllocator> command_allocator_;
    ComPtr<ID3D12GraphicsCommandList> command_list_;
    ComPtr<ID3D12Fence> fence_;
    HANDLE fence_event_ = nullptr;
    UINT64 fence_value_ = 0;
    ComPtr<ID3D12RootSignature> root_signature_;
    ComPtr<ID3D12PipelineState> pipeline_;
    std::array<ComPtr<ID3D12Resource>, 3> sources_{};
    std::array<uint32_t, 3> widths_{};
    std::array<uint32_t, 3> heights_{};
    ComPtr<ID3D12Resource> curve_;
    ComPtr<ID3D12Resource> lut_;
    ComPtr<ID3D12Resource> output_;
    ComPtr<ID3D12Resource> readback_;
    size_t output_capacity_ = 0;
    im_filter_params filter_{};
    uint32_t lut_edge_ = 0;
};

std::unique_ptr<Backend> create_d3d12_backend(std::string &error) {
    auto backend = std::make_unique<D3D12Backend>(error);
    if (!backend->ready()) return nullptr;
    return backend;
}

} // namespace imprint
