// Research-only full-resolution CFA renderer using the pinned redlamp-libraw
// build. This reads a NEF, optionally overlays an external little-endian
// uint16 CFA in memory, then runs LibRaw's ordinary dcraw processing path.
// It never writes to the NEF or invokes Adobe software.

#include <libraw/libraw.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <fcntl.h>
#include <unistd.h>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct Timings {
  double open_seconds = 0.0;
  double unpack_seconds = 0.0;
  double cfa_overlay_seconds = 0.0;
  double process_seconds = 0.0;
  double make_image_seconds = 0.0;
  double save_ppm_seconds = 0.0;
};

static double elapsed_seconds(Clock::time_point start, Clock::time_point end) {
  return std::chrono::duration<double>(end - start).count();
}

static std::string json_quote(const std::string &value) {
  std::ostringstream out;
  out << '"';
  for (unsigned char ch : value) {
    switch (ch) {
    case '"': out << "\\\""; break;
    case '\\': out << "\\\\"; break;
    case '\b': out << "\\b"; break;
    case '\f': out << "\\f"; break;
    case '\n': out << "\\n"; break;
    case '\r': out << "\\r"; break;
    case '\t': out << "\\t"; break;
    default:
      if (ch < 0x20) {
        out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
            << static_cast<unsigned>(ch) << std::dec << std::setfill(' ');
      } else {
        out << static_cast<char>(ch);
      }
    }
  }
  out << '"';
  return out.str();
}

static fs::path normalized_path(const fs::path &path) {
  std::error_code ec;
  fs::path resolved = fs::weakly_canonical(path, ec);
  if (!ec)
    return resolved;
  ec.clear();
  resolved = fs::absolute(path, ec);
  return ec ? path.lexically_normal() : resolved.lexically_normal();
}

static bool write_all(int fd, const void *buffer, size_t length) {
  const auto *bytes = static_cast<const unsigned char *>(buffer);
  while (length > 0) {
    ssize_t written = ::write(fd, bytes, length);
    if (written < 0) {
      if (errno == EINTR)
        continue;
      return false;
    }
    if (written == 0)
      return false;
    bytes += static_cast<size_t>(written);
    length -= static_cast<size_t>(written);
  }
  return true;
}

static bool write_text_exclusive(const fs::path &path, const std::string &text,
                                 std::string &error) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    error = "cannot create " + path.string() + ": " + std::strerror(errno);
    return false;
  }
  bool ok = write_all(fd, text.data(), text.size());
  int saved_errno = errno;
  if (::close(fd) != 0 && ok) {
    ok = false;
    saved_errno = errno;
  }
  if (!ok) {
    ::unlink(path.c_str());
    error = "cannot finish writing " + path.string() + ": " +
            std::strerror(saved_errno);
  }
  return ok;
}

static bool write_ppm16_exclusive(const fs::path &path,
                                 const libraw_processed_image_t &image,
                                 std::string &error) {
  int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    error = "cannot create " + path.string() + ": " + std::strerror(errno);
    return false;
  }

  std::ostringstream header;
  header << "P6\n" << image.width << ' ' << image.height << "\n65535\n";
  const std::string header_bytes = header.str();
  bool ok = write_all(fd, header_bytes.data(), header_bytes.size());
  std::vector<unsigned char> row(static_cast<size_t>(image.width) * 3U * 2U);
  const auto *source = static_cast<const unsigned char *>(image.data);

  for (size_t y = 0; ok && y < image.height; ++y) {
    const size_t source_row = y * static_cast<size_t>(image.width) * 3U * 2U;
    for (size_t sample = 0; sample < static_cast<size_t>(image.width) * 3U;
         ++sample) {
      uint16_t value = 0;
      std::memcpy(&value, source + source_row + sample * 2U, sizeof(value));
      row[sample * 2U] = static_cast<unsigned char>(value >> 8);
      row[sample * 2U + 1U] = static_cast<unsigned char>(value & 0xffU);
    }
    ok = write_all(fd, row.data(), row.size());
  }

  int saved_errno = errno;
  if (::close(fd) != 0 && ok) {
    ok = false;
    saved_errno = errno;
  }
  if (!ok) {
    ::unlink(path.c_str());
    error = "cannot finish writing " + path.string() + ": " +
            std::strerror(saved_errno);
  }
  return ok;
}

static void print_usage(const char *program) {
  std::cerr << "usage: " << program
            << " NEF OUTPUT.ppm OUTPUT.json [--cfa CFA.u16]\n"
            << "  CFA input is little-endian uint16, exactly raw_width * raw_height samples.\n"
            << "  This experiment is scoped to the Nikon HE* decoder path.\n"
            << "  Outputs must be new paths; existing files are never overwritten.\n";
}

static std::string libraw_error(int code) {
  const char *message = LibRaw::strerror(code);
  return message ? message : "unknown LibRaw error";
}

int main(int argc, char **argv) {
  if (argc == 2 && std::string(argv[1]) == "--help") {
    print_usage(argv[0]);
    return 0;
  }
  if (argc != 4 && argc != 6) {
    print_usage(argv[0]);
    return 64;
  }

  const fs::path nef_path(argv[1]);
  const fs::path ppm_path(argv[2]);
  const fs::path json_path(argv[3]);
  fs::path cfa_path;
  if (argc == 6) {
    if (std::string(argv[4]) != "--cfa") {
      print_usage(argv[0]);
      return 64;
    }
    cfa_path = fs::path(argv[5]);
  }

  std::error_code fs_error;
  if (!fs::is_regular_file(nef_path, fs_error) || fs_error) {
    std::cerr << "input NEF is not a readable regular file\n";
    return 2;
  }
  if (!cfa_path.empty() &&
      (!fs::is_regular_file(cfa_path, fs_error) || fs_error)) {
    std::cerr << "external CFA is not a readable regular file\n";
    return 2;
  }

  const fs::path nef_normalized = normalized_path(nef_path);
  const fs::path ppm_normalized = normalized_path(ppm_path);
  const fs::path json_normalized = normalized_path(json_path);
  if (ppm_normalized == json_normalized || ppm_normalized == nef_normalized ||
      json_normalized == nef_normalized ||
      (!cfa_path.empty() &&
       (normalized_path(cfa_path) == ppm_normalized ||
        normalized_path(cfa_path) == json_normalized))) {
    std::cerr << "input and output paths must be distinct\n";
    return 2;
  }
  if (fs::exists(ppm_path, fs_error) || fs_error) {
    std::cerr << "refusing to overwrite output: " << ppm_path << '\n';
    return 2;
  }
  fs_error.clear();
  if (fs::exists(json_path, fs_error) || fs_error) {
    std::cerr << "refusing to overwrite output: " << json_path << '\n';
    return 2;
  }

  LibRaw raw;
  Timings timings;
  auto stage_start = Clock::now();
  int result = raw.open_file(nef_path.c_str());
  auto stage_end = Clock::now();
  timings.open_seconds = elapsed_seconds(stage_start, stage_end);
  if (result != LIBRAW_SUCCESS) {
    std::cerr << "LibRaw.open_file failed: " << libraw_error(result) << '\n';
    return 3;
  }

  stage_start = Clock::now();
  result = raw.unpack();
  stage_end = Clock::now();
  timings.unpack_seconds = elapsed_seconds(stage_start, stage_end);
  if (result != LIBRAW_SUCCESS) {
    std::cerr << "LibRaw.unpack failed: " << libraw_error(result) << '\n';
    return 3;
  }
  const char *unpack_name = raw.unpack_function_name();
  const std::string unpack_function = unpack_name ? unpack_name : "unknown";
  if (unpack_function != "nikon_he_load_raw()") {
    std::cerr << "this experiment is scoped to Nikon HE*; LibRaw selected "
              << unpack_function << '\n';
    return 3;
  }

  const size_t raw_width = raw.imgdata.sizes.raw_width;
  const size_t raw_height = raw.imgdata.sizes.raw_height;
  const size_t raw_pitch_bytes = raw.imgdata.sizes.raw_pitch;
  if (raw_width == 0 || raw_height == 0 ||
      raw_width > std::numeric_limits<size_t>::max() / raw_height) {
    std::cerr << "LibRaw reported invalid raw dimensions\n";
    return 3;
  }
  const size_t raw_samples = raw_width * raw_height;
  if (raw_samples > std::numeric_limits<size_t>::max() / sizeof(uint16_t)) {
    std::cerr << "raw CFA size overflows this platform\n";
    return 3;
  }
  if (raw.imgdata.rawdata.raw_image == nullptr) {
    std::cerr << "LibRaw did not expose a single-plane CFA buffer\n";
    return 3;
  }
  if (raw_pitch_bytes < raw_width * sizeof(uint16_t) ||
      raw_pitch_bytes % sizeof(uint16_t) != 0) {
    std::cerr << "LibRaw raw_pitch cannot hold aligned uint16 CFA rows\n";
    return 3;
  }
  const size_t raw_pitch_samples = raw_pitch_bytes / sizeof(uint16_t);
  if (raw_height != 0 && raw_pitch_samples >
                             std::numeric_limits<size_t>::max() / raw_height) {
    std::cerr << "LibRaw raw_pitch dimensions overflow this platform\n";
    return 3;
  }

  if (!cfa_path.empty()) {
    stage_start = Clock::now();
    const uintmax_t file_bytes = fs::file_size(cfa_path, fs_error);
    const size_t expected_bytes = raw_samples * sizeof(uint16_t);
    if (fs_error || file_bytes != expected_bytes) {
      std::cerr << "external CFA must be exactly raw_width * raw_height * 2 bytes; expected "
                << expected_bytes << ", got "
                << (fs_error ? 0 : static_cast<unsigned long long>(file_bytes))
                << '\n';
      return 4;
    }

    std::ifstream input(cfa_path, std::ios::binary);
    if (!input) {
      std::cerr << "cannot open external CFA\n";
      return 4;
    }
    std::vector<unsigned char> bytes(expected_bytes);
    input.read(reinterpret_cast<char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
    if (!input || static_cast<size_t>(input.gcount()) != expected_bytes) {
      std::cerr << "short read from external CFA\n";
      return 4;
    }
    for (size_t y = 0; y < raw_height; ++y) {
      uint16_t *destination =
          raw.imgdata.rawdata.raw_image + y * raw_pitch_samples;
      const size_t source_row = y * raw_width;
      for (size_t x = 0; x < raw_width; ++x) {
        const size_t source_sample = source_row + x;
        destination[x] =
            static_cast<uint16_t>(bytes[source_sample * 2U]) |
            static_cast<uint16_t>(static_cast<uint16_t>(
                                      bytes[source_sample * 2U + 1U])
                                  << 8);
      }
    }
    stage_end = Clock::now();
    timings.cfa_overlay_seconds = elapsed_seconds(stage_start, stage_end);
  }

  // Capture source metadata before processing mutates LibRaw's working state.
  const auto &idata = raw.imgdata.idata;
  const auto &sizes = raw.imgdata.sizes;
  const auto &color = raw.imgdata.color;
  const std::string camera_make = idata.make;
  const std::string camera_model = idata.model;
  const std::string camera_unique_model = color.UniqueCameraModel;
  const unsigned raw_black = color.black;
  const unsigned raw_white = color.maximum;
  unsigned cblack[4] = {color.cblack[0], color.cblack[1], color.cblack[2],
                        color.cblack[3]};
  float camera_wb[4];
  float pre_wb[4];
  float rgb_camera_matrix[3][4];
  std::memcpy(camera_wb, color.cam_mul, sizeof(camera_wb));
  std::memcpy(pre_wb, color.pre_mul, sizeof(pre_wb));
  std::memcpy(rgb_camera_matrix, color.rgb_cam, sizeof(rgb_camera_matrix));

  // Keep every requested rendering control explicit. The 1.0 multiplier is
  // exactly 0 EV; automatic brightness remains enabled as requested.
  auto &params = raw.imgdata.params;
  params.use_camera_wb = 1;
  params.use_auto_wb = 0;
  params.use_camera_matrix = 1;
  params.output_color = 1; // LibRaw output_colorspace 1 = sRGB
  params.output_bps = 16;
  params.output_tiff = 0;
  params.user_qual = 3; // AHD
  params.no_auto_bright = 0;
  params.exp_correc = 1;
  params.exp_shift = 1.0f;
  params.exp_preser = 0.0f;
  params.user_flip = -1; // honor source orientation

  stage_start = Clock::now();
  result = raw.dcraw_process();
  stage_end = Clock::now();
  timings.process_seconds = elapsed_seconds(stage_start, stage_end);
  if (result != LIBRAW_SUCCESS) {
    std::cerr << "LibRaw.dcraw_process failed: " << libraw_error(result) << '\n';
    return 5;
  }

  int image_error = LIBRAW_SUCCESS;
  stage_start = Clock::now();
  std::unique_ptr<libraw_processed_image_t, void (*)(libraw_processed_image_t *)>
      image(raw.dcraw_make_mem_image(&image_error), &LibRaw::dcraw_clear_mem);
  stage_end = Clock::now();
  timings.make_image_seconds = elapsed_seconds(stage_start, stage_end);
  if (!image || image_error != LIBRAW_SUCCESS) {
    std::cerr << "LibRaw.dcraw_make_mem_image failed: "
              << libraw_error(image_error) << '\n';
    return 5;
  }

  const size_t expected_image_bytes = static_cast<size_t>(image->width) *
                                      static_cast<size_t>(image->height) * 3U * 2U;
  if (image->type != LIBRAW_IMAGE_BITMAP || image->colors != 3 ||
      image->bits != 16 || image->data_size != expected_image_bytes ||
      image->width == 0 || image->height == 0) {
    std::cerr << "LibRaw memory image is not a complete RGB16 bitmap\n";
    return 5;
  }

  stage_start = Clock::now();
  std::string write_error;
  if (!write_ppm16_exclusive(ppm_path, *image, write_error)) {
    std::cerr << write_error << '\n';
    return 6;
  }
  stage_end = Clock::now();
  timings.save_ppm_seconds = elapsed_seconds(stage_start, stage_end);

  const double total_seconds = timings.open_seconds + timings.unpack_seconds +
                               timings.cfa_overlay_seconds + timings.process_seconds +
                               timings.make_image_seconds + timings.save_ppm_seconds;
  std::ostringstream json;
  json << std::setprecision(9)
       << "{\n"
       << "  \"decoder\": " << json_quote(std::string("LibRaw ") + LibRaw::version()) << ",\n"
       << "  \"libraw_unpack_function\": " << json_quote(unpack_function) << ",\n"
       << "  \"cfa_source\": "
       << json_quote(cfa_path.empty() ? "LibRaw unpacked NEF CFA" : "external little-endian uint16 CFA")
       << ",\n"
       << "  \"external_cfa_applied\": " << (cfa_path.empty() ? "false" : "true") << ",\n"
       << "  \"camera\": {\"make\": " << json_quote(camera_make)
       << ", \"model\": " << json_quote(camera_model)
       << ", \"unique_model\": " << json_quote(camera_unique_model) << "},\n"
       << "  \"raw_dimensions\": {\"width\": " << sizes.raw_width
       << ", \"height\": " << sizes.raw_height
       << ", \"pitch_bytes\": " << sizes.raw_pitch << "},\n"
       << "  \"raw_levels\": {\"black\": " << raw_black
       << ", \"cblack_first_four\": [" << cblack[0] << ", " << cblack[1]
       << ", " << cblack[2] << ", " << cblack[3] << "]"
       << ", \"white\": " << raw_white << "},\n"
       << "  \"white_balance\": {\"requested\": \"camera\", \"cam_mul\": ["
       << camera_wb[0] << ", " << camera_wb[1] << ", " << camera_wb[2]
       << ", " << camera_wb[3] << "], \"pre_mul\": [" << pre_wb[0] << ", "
       << pre_wb[1] << ", " << pre_wb[2] << ", " << pre_wb[3] << "]},\n"
       << "  \"rgb_camera_matrix\": [[" << rgb_camera_matrix[0][0] << ", "
       << rgb_camera_matrix[0][1] << ", " << rgb_camera_matrix[0][2] << ", "
       << rgb_camera_matrix[0][3] << "], [" << rgb_camera_matrix[1][0] << ", "
       << rgb_camera_matrix[1][1] << ", " << rgb_camera_matrix[1][2] << ", "
       << rgb_camera_matrix[1][3] << "], [" << rgb_camera_matrix[2][0] << ", "
       << rgb_camera_matrix[2][1] << ", " << rgb_camera_matrix[2][2] << ", "
       << rgb_camera_matrix[2][3] << "]],\n"
       << "  \"render\": {\"white_balance\": \"camera\", \"output_color\": \"sRGB (1)\", "
       << "\"output_bps\": 16, \"quality\": \"AHD (user_qual=3)\", "
       << "\"auto_brightness\": true, \"exposure_correction\": true, "
       << "\"exposure_multiplier\": 1.0, \"exposure_ev\": 0.0, "
       << "\"exposure_preservation\": 0.0, \"use_camera_matrix\": true, "
       << "\"orientation\": \"source orientation (user_flip=-1)\"},\n"
       << "  \"output\": {\"format\": \"PPM P6\", \"sample_endian\": \"big\", "
       << "\"width\": " << image->width << ", \"height\": " << image->height
       << ", \"colors\": " << image->colors << ", \"bits_per_sample\": "
       << image->bits << ", \"data_bytes\": " << image->data_size << "},\n"
       << "  \"timings_seconds\": {\"open_file\": " << timings.open_seconds
       << ", \"unpack\": " << timings.unpack_seconds
       << ", \"cfa_overlay\": " << timings.cfa_overlay_seconds
       << ", \"dcraw_process\": " << timings.process_seconds
       << ", \"dcraw_make_mem_image\": " << timings.make_image_seconds
       << ", \"save_ppm\": " << timings.save_ppm_seconds
       << ", \"total\": " << total_seconds << "}\n}\n";
  if (!write_text_exclusive(json_path, json.str(), write_error)) {
    ::unlink(ppm_path.c_str()); // Remove only the PPM just created by this run.
    std::cerr << write_error << '\n';
    return 6;
  }

  std::cout << "rendered " << image->width << 'x' << image->height
            << " RGB16 PPM with LibRaw " << LibRaw::version() << '\n';
  std::cout << "PPM: " << ppm_path << '\n' << "JSON: " << json_path << '\n';
  return 0;
}
