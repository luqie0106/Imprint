#include "backend.hpp"

#if defined(IMPRINT_BACKEND_METAL)
namespace imprint { std::unique_ptr<Backend> create_metal_backend(std::string &error); }
#endif
#if defined(IMPRINT_BACKEND_CUDA)
namespace imprint { std::unique_ptr<Backend> create_cuda_backend(std::string &error); }
#endif

namespace imprint {

std::unique_ptr<Backend> create_backend(im_backend_kind kind, std::string &error) {
    if (kind == IM_BACKEND_AUTO) {
#if defined(IMPRINT_BACKEND_METAL)
        auto backend = create_metal_backend(error);
        if (backend) return backend;
#endif
#if defined(IMPRINT_BACKEND_CUDA)
        auto backend = create_cuda_backend(error);
        if (backend) return backend;
#endif
        if (error.empty()) error = "No native GPU backend was built for this platform";
        return nullptr;
    }

    if (kind == IM_BACKEND_METAL) {
#if defined(IMPRINT_BACKEND_METAL)
        return create_metal_backend(error);
#else
        error = "Metal backend is not available in this build";
        return nullptr;
#endif
    }

    if (kind == IM_BACKEND_CUDA) {
#if defined(IMPRINT_BACKEND_CUDA)
        return create_cuda_backend(error);
#else
        error = "CUDA backend is not available in this build";
        return nullptr;
#endif
    }

    error = "Unknown backend kind";
    return nullptr;
}

} // namespace imprint
