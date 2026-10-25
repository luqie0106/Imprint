#include "acr_post_tone_basic.hpp"

namespace imprint::acr {

void normalize_post_tone_basic(std::uint32_t process_version,
                               const PostToneBasicSlots& input,
                               const PostToneBasicSlots& source_defaults,
                               PostToneBasicSlots& output) noexcept {
    PostToneBasicSlots candidate{};
    const std::uint32_t version_plus_one = process_version + std::uint32_t{1};
    const bool use_source_defaults = version_plus_one < UINT32_C(0x05070001);

    for (std::size_t i = 0; i < candidate.size(); ++i) {
        switch (i) {
            case 0x09:
            case 0x0a:
            case 0x0f:
            case 0x70:
            case 0x71:
            case 0x72:
            case 0x73:
            case 0x74:
            case 0x75:
            case 0x76:
            case 0x81:
            case 0x82:
                candidate[i] = input[i];
                break;
            case 0x0e:
                candidate[i] = input[i] < 0 ? 0 : input[i];
                break;
            case 0x64:
                candidate[i] = 100;
                break;
            case 0x19:
            case 0x1a:
            case 0x1b:
            case 0x1c:
            case 0x1d:
            case 0x1e:
            case 0x1f:
            case 0x20:
            case 0x21:
            case 0x22:
                candidate[i] = use_source_defaults ? source_defaults[i] : 0;
                break;
            default:
                candidate[i] = 0;
                break;
        }
    }

    output = candidate;
}

}  // namespace imprint::acr
