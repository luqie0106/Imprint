#include "native_preview.h"
#include "imprint_renderer.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <new>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace {
#if defined(_WIN32)
using LibraryHandle = HMODULE;
#else
using LibraryHandle = void *;
#endif

using CreateFn = im_status (*)(im_backend_kind, im_renderer **);
using DestroyFn = void (*)(im_renderer *);
using LastErrorFn = const char *(*)(const im_renderer *);
using BackendNameFn = const char *(*)(const im_renderer *);
using ExpandFn = im_status (*)(uint32_t, uint32_t, const uint8_t *, size_t,
                               uint16_t *, size_t);
using CompactFn = im_status (*)(uint32_t, uint32_t, const uint16_t *, size_t,
                                uint8_t *, size_t);
using RenderFullFn = im_status (*)(im_renderer *, uint32_t, uint32_t,
                                   const uint16_t *, size_t,
                                   const im_dehaze_params *, const im_basic_params *);
using GetOutputSizeFn = im_status (*)(const im_renderer *, uint32_t *, uint32_t *, size_t *);
using CopyOutputFn = im_status (*)(const im_renderer *, uint16_t *, size_t);

struct RendererApi {
    CreateFn create = nullptr;
    DestroyFn destroy = nullptr;
    LastErrorFn last_error = nullptr;
    BackendNameFn backend_name = nullptr;
    ExpandFn expand = nullptr;
    CompactFn compact = nullptr;
    RenderFullFn render_full = nullptr;
    GetOutputSizeFn get_output_size = nullptr;
    CopyOutputFn copy_output = nullptr;
};

struct Runtime {
    LibraryHandle library = nullptr;
    im_renderer *renderer = nullptr;
    RendererApi api{};
    std::string path;

    void clear() noexcept {
        if (renderer && api.destroy) api.destroy(renderer);
        renderer = nullptr;
        api = {};
        path.clear();
        if (library) close_library(library);
        library = nullptr;
    }

    ~Runtime() { clear(); }

    static void close_library(LibraryHandle handle) noexcept {
#if defined(_WIN32)
        FreeLibrary(handle);
#else
        dlclose(handle);
#endif
    }
};

std::mutex g_runtime_mutex;
Runtime g_runtime;

LibraryHandle open_library(const char *path, std::string *detail) {
#if defined(_WIN32)
    const int required = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS,
                                              path, -1, nullptr, 0);
    if (required <= 0) {
        if (detail) *detail = "library path is not valid UTF-8";
        return nullptr;
    }
    std::wstring wide_path(static_cast<size_t>(required), L'\0');
    if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, path, -1,
                            wide_path.data(), required) <= 0) {
        if (detail) *detail = "library path could not be converted to UTF-16";
        return nullptr;
    }
    LibraryHandle handle = LoadLibraryW(wide_path.c_str());
    if (!handle && detail) *detail = "LoadLibraryW failed with Windows error " +
                                      std::to_string(GetLastError());
    return handle;
#else
    dlerror();
    LibraryHandle handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!handle && detail) {
        const char *message = dlerror();
        *detail = message ? message : "dlopen failed without a diagnostic";
    }
    return handle;
#endif
}

void *lookup_symbol(LibraryHandle handle, const char *name, std::string *detail) {
#if defined(_WIN32)
    FARPROC symbol = GetProcAddress(handle, name);
    if (!symbol && detail) *detail = "missing renderer symbol: " + std::string(name);
    return reinterpret_cast<void *>(symbol);
#else
    dlerror();
    void *symbol = dlsym(handle, name);
    const char *message = dlerror();
    if ((!symbol || message) && detail) {
        *detail = "missing renderer symbol " + std::string(name);
        if (message) *detail += ": " + std::string(message);
    }
    return symbol;
#endif
}

template <typename Function>
bool resolve(LibraryHandle handle, const char *name, Function *function,
             std::string *detail) {
    void *symbol = lookup_symbol(handle, name, detail);
    if (!symbol) return false;
    *function = reinterpret_cast<Function>(symbol);
    return true;
}

const char *status_name(im_status status) noexcept {
    switch (status) {
        case IM_STATUS_OK: return "success";
        case IM_STATUS_INVALID_ARGUMENT: return "invalid argument";
        case IM_STATUS_BACKEND_UNAVAILABLE: return "GPU backend unavailable";
        case IM_STATUS_NOT_READY: return "renderer output not ready";
        case IM_STATUS_BUFFER_TOO_SMALL: return "renderer buffer too small";
        case IM_STATUS_RUNTIME_ERROR: return "renderer runtime error";
    }
    return "unknown renderer status";
}

std::string renderer_error(const Runtime &runtime, const char *fallback) {
    if (runtime.renderer && runtime.api.last_error) {
        const char *message = runtime.api.last_error(runtime.renderer);
        if (message && message[0] != '\0') return message;
    }
    return fallback;
}

std::string active_backend(const Runtime &runtime) {
    if (runtime.renderer && runtime.api.backend_name) {
        const char *name = runtime.api.backend_name(runtime.renderer);
        if (name && name[0] != '\0') return name;
    }
    return "unavailable";
}

void write_text(char *destination, uint32_t capacity, const std::string &value) noexcept {
    if (!destination || capacity == 0) return;
    const size_t copy_length = std::min(value.size(), static_cast<size_t>(capacity - 1));
    if (copy_length != 0) std::memcpy(destination, value.data(), copy_length);
    destination[copy_length] = '\0';
}

bool ensure_runtime(const char *path, std::string *error) {
    if (g_runtime.library && g_runtime.renderer && g_runtime.path == path) return true;

    std::string load_detail;
    LibraryHandle candidate_library = open_library(path, &load_detail);
    if (!candidate_library) {
        *error = "could not load native renderer library: " + load_detail;
        return false;
    }

    RendererApi candidate_api;
    if (!resolve(candidate_library, "im_renderer_create", &candidate_api.create, &load_detail) ||
        !resolve(candidate_library, "im_renderer_destroy", &candidate_api.destroy, &load_detail) ||
        !resolve(candidate_library, "im_renderer_last_error", &candidate_api.last_error, &load_detail) ||
        !resolve(candidate_library, "im_renderer_backend_name", &candidate_api.backend_name, &load_detail) ||
        !resolve(candidate_library, "im_native_expand_rgb8_to_rgb16", &candidate_api.expand, &load_detail) ||
        !resolve(candidate_library, "im_native_compact_rgb16_to_rgb8", &candidate_api.compact, &load_detail) ||
        !resolve(candidate_library, "im_renderer_render_full", &candidate_api.render_full, &load_detail) ||
        !resolve(candidate_library, "im_renderer_get_output_size", &candidate_api.get_output_size, &load_detail) ||
        !resolve(candidate_library, "im_renderer_copy_output", &candidate_api.copy_output, &load_detail)) {
        Runtime::close_library(candidate_library);
        *error = "incompatible native renderer library: " + load_detail;
        return false;
    }

    im_renderer *candidate_renderer = nullptr;
    const im_status create_status = candidate_api.create(IM_BACKEND_AUTO, &candidate_renderer);
    if (create_status != IM_STATUS_OK || !candidate_renderer) {
        if (candidate_renderer) candidate_api.destroy(candidate_renderer);
        Runtime::close_library(candidate_library);
        *error = std::string("could not create native GPU renderer (") +
                 status_name(create_status) + ")";
        return false;
    }
    const char *candidate_backend = candidate_api.backend_name(candidate_renderer);
    if (!candidate_backend || candidate_backend[0] == '\0') {
        candidate_api.destroy(candidate_renderer);
        Runtime::close_library(candidate_library);
        *error = "renderer did not report the selected backend";
        return false;
    }

    g_runtime.clear();
    g_runtime.library = candidate_library;
    g_runtime.renderer = candidate_renderer;
    g_runtime.api = candidate_api;
    g_runtime.path = path;
    return true;
}

int32_t fail(char *backend, uint32_t backend_cap, char *error, uint32_t errcap,
             const std::string &message, const std::string &backend_name = "unavailable") noexcept {
    write_text(backend, backend_cap, backend_name);
    write_text(error, errcap, message);
    return 0;
}
} // namespace

extern "C" int32_t imprint_geo_native_preview(const char *library_path,
                                               const uint8_t *rgb,
                                               uint32_t width,
                                               uint32_t height,
                                               uint8_t *out_rgb,
                                               char *backend,
                                               uint32_t backend_cap,
                                               char *error,
                                               uint32_t errcap) {
    write_text(backend, backend_cap, "unavailable");
    write_text(error, errcap, "");

    if (!library_path || library_path[0] == '\0') {
        return fail(backend, backend_cap, error, errcap,
                    "native renderer library path is required");
    }
    if (!rgb || !out_rgb || width == 0 || height == 0) {
        return fail(backend, backend_cap, error, errcap,
                    "RGB input/output and nonzero dimensions are required");
    }

    if (width > 65535 || height > 65535) {
        return fail(backend, backend_cap, error, errcap,
                    "image dimensions exceed the existing renderer full-resolution limit");
    }
    const uint64_t sample_count_64 = static_cast<uint64_t>(width) * height * 3;
    if (sample_count_64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max()) ||
        sample_count_64 > (1ull << 29)) {
        return fail(backend, backend_cap, error, errcap,
                    "image dimensions exceed the existing renderer full-resolution limit");
    }
    const size_t sample_count = static_cast<size_t>(sample_count_64);
    const size_t byte_count = sample_count;

    std::lock_guard<std::mutex> lock(g_runtime_mutex);
    try {
        std::string runtime_error;
        if (!ensure_runtime(library_path, &runtime_error)) {
            return fail(backend, backend_cap, error, errcap, runtime_error);
        }
        const std::string backend_name = active_backend(g_runtime);

        std::vector<uint16_t> expanded(sample_count);
        std::vector<uint16_t> rendered(sample_count);
        std::vector<uint8_t> compacted(byte_count);

        im_status status = g_runtime.api.expand(width, height, rgb, byte_count,
                                                expanded.data(), expanded.size());
        if (status != IM_STATUS_OK) {
            return fail(backend, backend_cap, error, errcap,
                        std::string("RGB8 expansion failed: ") + status_name(status), backend_name);
        }

        const im_dehaze_params neutral_dehaze{0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                              0.0f, 0.0f, 0.0f, 0.0f};
        const im_basic_params neutral_basic{0.0f, 0.0f, 0.0f, 0.0f,
                                            0.0f, 0.0f, 0.0f, 0.0f};
        status = g_runtime.api.render_full(g_runtime.renderer, width, height,
                                           expanded.data(), expanded.size(),
                                           &neutral_dehaze, &neutral_basic);
        if (status != IM_STATUS_OK) {
            return fail(backend, backend_cap, error, errcap,
                        "full-resolution native renderer failed: " +
                            renderer_error(g_runtime, status_name(status)), backend_name);
        }

        uint32_t output_width = 0;
        uint32_t output_height = 0;
        size_t output_samples = 0;
        status = g_runtime.api.get_output_size(g_runtime.renderer, &output_width,
                                               &output_height, &output_samples);
        if (status != IM_STATUS_OK) {
            return fail(backend, backend_cap, error, errcap,
                        "could not inspect native renderer output: " +
                            renderer_error(g_runtime, status_name(status)), backend_name);
        }
        if (output_width != width || output_height != height || output_samples != sample_count) {
            return fail(backend, backend_cap, error, errcap,
                        "native renderer returned output with incompatible dimensions or sample count",
                        backend_name);
        }

        status = g_runtime.api.copy_output(g_runtime.renderer, rendered.data(), rendered.size());
        if (status != IM_STATUS_OK) {
            return fail(backend, backend_cap, error, errcap,
                        "native renderer output copy failed: " +
                            renderer_error(g_runtime, status_name(status)), backend_name);
        }

        status = g_runtime.api.compact(width, height, rendered.data(), rendered.size(),
                                       compacted.data(), compacted.size());
        if (status != IM_STATUS_OK) {
            return fail(backend, backend_cap, error, errcap,
                        std::string("RGB16 compaction failed: ") + status_name(status), backend_name);
        }

        if (std::memcmp(rgb, compacted.data(), byte_count) != 0) {
            return fail(backend, backend_cap, error, errcap,
                        "native renderer is incompatible with byte-exact RGB8 identity; use the C++ passthrough fallback",
                        backend_name);
        }

        write_text(backend, backend_cap, backend_name);
        write_text(error, errcap, "");
        std::memmove(out_rgb, compacted.data(), byte_count);
        return 1;
    } catch (const std::bad_alloc &) {
        return fail(backend, backend_cap, error, errcap,
                    "insufficient memory for full-resolution native preview",
                    active_backend(g_runtime));
    } catch (const std::exception &exception) {
        return fail(backend, backend_cap, error, errcap,
                    std::string("native preview adapter failed: ") + exception.what(),
                    active_backend(g_runtime));
    } catch (...) {
        return fail(backend, backend_cap, error, errcap,
                    "unknown native preview adapter failure", active_backend(g_runtime));
    }
}
