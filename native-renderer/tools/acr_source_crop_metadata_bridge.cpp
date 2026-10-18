#include "../src/acr_source_crop_metadata.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace {

constexpr std::size_t kWordCount = 15;

std::int32_t as_i32(std::uint32_t word) noexcept {
    std::int32_t value = 0;
    std::memcpy(&value, &word, sizeof(value));
    return value;
}

std::uint32_t as_u32(std::int32_t value) noexcept {
    std::uint32_t word = 0;
    std::memcpy(&word, &value, sizeof(word));
    return word;
}

imprint::acr::IFDCropState decode(const std::uint32_t *words) noexcept {
    imprint::acr::IFDCropState state{};
    state.new_subfile_type = words[0];
    state.image_width = words[1];
    state.image_height = words[2];
    state.active_area = {
        as_i32(words[3]), as_i32(words[4]), as_i32(words[5]), as_i32(words[6])};
    state.crop_origin_x = {words[7], words[8]};
    state.crop_origin_y = {words[9], words[10]};
    state.crop_size_x = {words[11], words[12]};
    state.crop_size_y = {words[13], words[14]};
    return state;
}

std::array<std::uint32_t, kWordCount> encode(
    const imprint::acr::IFDCropState &state) noexcept {
    return {
        state.new_subfile_type,
        state.image_width,
        state.image_height,
        as_u32(state.active_area.top),
        as_u32(state.active_area.left),
        as_u32(state.active_area.bottom),
        as_u32(state.active_area.right),
        state.crop_origin_x.numerator,
        state.crop_origin_x.denominator,
        state.crop_origin_y.numerator,
        state.crop_origin_y.denominator,
        state.crop_size_x.numerator,
        state.crop_size_x.denominator,
        state.crop_size_y.numerator,
        state.crop_size_y.denominator,
    };
}

}  // namespace

extern "C" int acr_test_finalize_source_ifd_crop_metadata(
    const std::uint32_t *input_words,
    std::uint32_t *output_words) noexcept {
    if (input_words == nullptr || output_words == nullptr) {
        return static_cast<int>(imprint::acr::IFDSourceCropStatus::invalid_input);
    }

    // Decode the full input before writing, so exact or partial overlap is safe.
    const imprint::acr::IFDCropState input = decode(input_words);
    imprint::acr::IFDCropState output{};
    const imprint::acr::IFDSourceCropStatus status =
        imprint::acr::finalize_source_ifd_crop(input, output);
    if (status != imprint::acr::IFDSourceCropStatus::ok) {
        return static_cast<int>(status);
    }

    const auto words = encode(output);
    std::memcpy(output_words, words.data(), kWordCount * sizeof(words[0]));
    return static_cast<int>(status);
}
