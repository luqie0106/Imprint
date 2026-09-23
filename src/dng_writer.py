"""Minimal standards-based writer for Linear/Demosaiced DNG files.

The output is a TIFF/EP container with DNG tags, LinearRaw photometric data,
packed integer strips, and a reduced JPEG preview IFD. It never writes
directly to the final filename.

This product includes DNG technology under license by Adobe.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from fractions import Fraction
import io
import math
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile
import threading
from typing import Any, Iterable
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SLONG, SRATIONAL, FLOAT, DOUBLE = 1, 2, 3, 4, 5, 7, 9, 10, 11, 12
TYPE_SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1, SLONG: 4, SRATIONAL: 8, FLOAT: 4, DOUBLE: 8}
_OUTPUT_LOCK = threading.Lock()


@dataclass
class _Tag:
    code: int
    kind: int
    count: int
    data: bytes


def _ascii(value: Any) -> bytes:
    return str(value).encode("utf-8", "replace") + b"\0"


def _shorts(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack("<" + "H" * len(values), *values)


def _longs(values: Iterable[int]) -> bytes:
    values = tuple(values)
    return struct.pack("<" + "I" * len(values), *values)


def _rationals(values: Iterable[tuple[int, int]], signed: bool = False) -> bytes:
    values = tuple(values)
    fmt = "<" + ("ii" if signed else "II") * len(values)
    flat = [part for value in values for part in value]
    return struct.pack(fmt, *flat)


def _tag(code: int, kind: int, data: bytes, count: int | None = None) -> _Tag:
    return _Tag(code, kind, count if count is not None else len(data) // TYPE_SIZES[kind], data)


def _metadata_text(value: Any) -> str | None:
    """Return non-empty metadata text that can be represented in TIFF ASCII."""
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            value = bytes(value).rstrip(b"\0").decode("utf-8")
        except UnicodeDecodeError:
            return None
    text = str(value).strip()
    if not text:
        return None
    # TIFF/Exif ASCII is a 7-bit field.  Skipping an unencodable value is safer
    # than placing UTF-8 bytes in a field readers expect to be ASCII.
    try:
        text.encode("ascii")
    except UnicodeEncodeError:
        return None
    return text


def _ascii_metadata(value: Any) -> bytes | None:
    text = _metadata_text(value)
    return None if text is None else text.encode("ascii") + b"\0"


def _datetime_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y:%m:%d %H:%M:%S")
    text = _metadata_text(value)
    if text is None:
        return None
    if re.fullmatch(r"\d{4}:\d{2}:\d{2} \d{2}:\d{2}:\d{2}", text):
        return text
    # Be liberal with ISO strings supplied by callers, while always writing
    # the EXIF-required colon-separated representation.
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed.strftime("%Y:%m:%d %H:%M:%S")


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            denominator = float(value[1])
            if denominator == 0:
                return None
            value = float(value[0]) / denominator
        except (TypeError, ValueError, OverflowError):
            return None
    elif isinstance(value, str) and "/" in value:
        numerator, separator, denominator = value.partition("/")
        if separator:
            try:
                value = float(numerator.strip()) / float(denominator.strip())
            except (TypeError, ValueError, OverflowError, ZeroDivisionError):
                return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _rational_pair(value: Any, *, signed: bool = False) -> tuple[int, int] | None:
    number = _number(value)
    if number is None:
        return None
    if not signed and number < 0:
        return None
    try:
        fraction = Fraction(number).limit_denominator(10_000_000)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    numerator, denominator = fraction.numerator, fraction.denominator
    if denominator <= 0:
        return None
    if signed:
        if numerator < -(2**31) or numerator > 2**31 - 1 or denominator > 2**31 - 1:
            return None
    elif numerator > 2**32 - 1 or denominator > 2**32 - 1:
        return None
    return numerator, denominator


def _short_value(value: Any, *, minimum: int = 0) -> int | None:
    number = _number(value)
    if number is None or not number.is_integer():
        return None
    result = int(number)
    return result if minimum <= result <= 65535 else None


def _coordinate(value: Any, maximum: float) -> list[tuple[int, int]] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    result: list[tuple[int, int]] = []
    parts: list[float] = []
    for item in value:
        pair = _rational_pair(item)
        number = _number(item)
        if pair is None or number is None or number < 0:
            return None
        result.append(pair)
        parts.append(number)
    if parts[0] > maximum or parts[1] >= 60 or parts[2] >= 60:
        return None
    return result


def _gps_tags(info: dict[str, Any]) -> list[_Tag]:
    """Build a GPS IFD only when a complete, standard coordinate is present."""
    latitude = _coordinate(info.get("GPSLatitude"), 90)
    longitude = _coordinate(info.get("GPSLongitude"), 180)
    latitude_ref = (_metadata_text(info.get("GPSLatitudeRef")) or "").upper()
    longitude_ref = (_metadata_text(info.get("GPSLongitudeRef")) or "").upper()
    if latitude is None or longitude is None or latitude_ref not in {"N", "S"} or longitude_ref not in {"E", "W"}:
        return []

    tags = [
        _tag(0, BYTE, bytes((2, 3, 0, 0)), 4),
        _tag(1, ASCII, _ascii(latitude_ref)),
        _tag(2, RATIONAL, _rationals(latitude)),
        _tag(3, ASCII, _ascii(longitude_ref)),
        _tag(4, RATIONAL, _rationals(longitude)),
    ]
    altitude = _number(info.get("GPSAltitude"))
    if altitude is not None:
        altitude_pair = _rational_pair(abs(altitude))
        if altitude_pair is not None:
            altitude_ref = _short_value(info.get("GPSAltitudeRef"), minimum=0)
            altitude_flag = 1 if altitude < 0 else (altitude_ref if altitude_ref in {0, 1} else 0)
            tags.append(_tag(5, BYTE, bytes((altitude_flag,)), 1))
            tags.append(_tag(6, RATIONAL, _rationals([altitude_pair])))

    timestamp = info.get("GPSTimeStamp")
    if isinstance(timestamp, (tuple, list)) and len(timestamp) == 3:
        time_pairs = [_rational_pair(part) for part in timestamp]
        if all(pair is not None for pair in time_pairs):
            tags.append(_tag(7, RATIONAL, _rationals(time_pairs)))

    date_stamp = _metadata_text(info.get("GPSDateStamp"))
    if date_stamp and re.fullmatch(r"\d{4}:\d{2}:\d{2}", date_stamp):
        tags.append(_tag(29, ASCII, _ascii(date_stamp)))
    return tags


def _exif_tags(info: dict[str, Any], width: int, height: int) -> list[_Tag]:
    """Build the ExifIFD using only standard fields with legal TIFF values."""
    tags = [
        _tag(36864, UNDEFINED, b"0232", 4),  # ExifVersion 2.32
        _tag(40961, SHORT, _shorts([1])),  # output pixels are linear sRGB
        _tag(40962, LONG, _longs([width])),
        _tag(40963, LONG, _longs([height])),
    ]
    original = _datetime_text(info.get("DateTimeOriginal"))
    digitized = _datetime_text(info.get("DateTimeDigitized")) or original
    if original is not None:
        tags.append(_tag(36867, ASCII, _ascii(original)))
    if digitized is not None:
        tags.append(_tag(36868, ASCII, _ascii(digitized)))

    exposure = _rational_pair(info.get("ExposureTime"))
    if exposure is not None:
        tags.append(_tag(33434, RATIONAL, _rationals([exposure])))
    f_number = _rational_pair(info.get("FNumber"))
    if f_number is not None:
        tags.append(_tag(33437, RATIONAL, _rationals([f_number])))
    iso = _short_value(info.get("ISO", info.get("ISOSpeedRatings")), minimum=1)
    if iso is not None:
        tags.append(_tag(34855, SHORT, _shorts([iso])))
    bias = _rational_pair(info.get("ExposureBiasValue"), signed=True)
    if bias is not None:
        tags.append(_tag(37380, SRATIONAL, _rationals([bias], signed=True)))
    for code, key in (
        (34850, "ExposureProgram"),
        (37383, "MeteringMode"),
        (37385, "Flash"),
        (41986, "ExposureMode"),
        (41987, "WhiteBalance"),
        (41989, "FocalLengthIn35mmFilm"),
        (41990, "SceneCaptureType"),
        (41991, "GainControl"),
        (41992, "Contrast"),
        (41993, "Saturation"),
        (41994, "Sharpness"),
    ):
        value = _short_value(info.get(key), minimum=0)
        if value is not None:
            tags.append(_tag(code, SHORT, _shorts([value])))
    focal = _rational_pair(info.get("FocalLength"))
    if focal is not None:
        tags.append(_tag(37386, RATIONAL, _rationals([focal])))

    for code, key in (
        (36880, "OffsetTime"), (36881, "OffsetTimeOriginal"),
        (36882, "OffsetTimeDigitized"), (37520, "SubsecTime"),
        (37521, "SubsecTimeOriginal"), (37522, "SubsecTimeDigitized"),
        (42016, "ImageUniqueID"), (42032, "CameraOwnerName"),
        (42035, "LensMake"), (42036, "LensModel"),
        (42033, "BodySerialNumber"), (42037, "LensSerialNumber"),
    ):
        encoded = _ascii_metadata(info.get(key))
        if encoded is not None:
            tags.append(_tag(code, ASCII, encoded))

    lens_specification = info.get("LensSpecification")
    if isinstance(lens_specification, (tuple, list)) and len(lens_specification) == 4:
        lens_pairs = [_rational_pair(value) for value in lens_specification]
        if all(pair is not None for pair in lens_pairs):
            tags.append(_tag(42034, RATIONAL, _rationals(lens_pairs)))
    return tags


def _acr_lens_xmp(source: Path, info: dict[str, Any]) -> bytes | None:
    """Build embedded Camera Raw and Imprint lens-correction metadata.

    When Lensfun correction has already been baked into the Linear RGB pixels,
    the packet explicitly disables an additional ACR profile pass and records
    what Imprint applied.  The legacy automatic-profile request remains only
    for callers that have not supplied a correction result.
    """
    lens_model = _metadata_text(info.get("LensModel"))
    camera_model = _metadata_text(info.get("Model"))
    picture_control = _metadata_text(info.get("NikonPictureControlName"))
    source_camera_profile = _metadata_text(info.get("SourceCameraProfileName"))
    embedded_profile = _metadata_text(info.get("DNGEmbeddedProfileName"))
    if (
        lens_model is None
        and camera_model is None
        and picture_control is None
        and source_camera_profile is None
        and embedded_profile is None
    ):
        return None
    lens_serial = _metadata_text(info.get("LensSerialNumber"))
    camera_serial = _metadata_text(info.get("BodySerialNumber"))
    lens_make = _metadata_text(info.get("LensMake"))
    lens_info = info.get("LensSpecification")

    auxiliary: list[str] = []
    if lens_model is not None:
        auxiliary.append(f'aux:Lens="{escape(lens_model, {chr(34): "&quot;"})}"')
    if lens_make is not None:
        auxiliary.append(f'aux:LensMake="{escape(lens_make, {chr(34): "&quot;"})}"')
    if isinstance(lens_info, (tuple, list)) and len(lens_info) == 4:
        lens_pairs = [_rational_pair(value) for value in lens_info]
        if all(pair is not None for pair in lens_pairs):
            encoded_lens_info = " ".join(
                f"{pair[0]}/{pair[1]}" for pair in lens_pairs if pair is not None
            )
            auxiliary.append(f'aux:LensInfo="{encoded_lens_info}"')
    if lens_serial is not None:
        auxiliary.append(f'aux:LensSerialNumber="{escape(lens_serial, {chr(34): "&quot;"})}"')
    if camera_serial is not None:
        auxiliary.append(f'aux:SerialNumber="{escape(camera_serial, {chr(34): "&quot;"})}"')
    correction_applied = bool(info.get("LensCorrectionApplied"))
    correction_attributes: list[str] = []
    if correction_applied:
        correction_attributes.append('imprint:LensCorrectionApplied="True"')
        for key, attribute in (
            ("LensCorrectionEngine", "LensCorrectionEngine"),
            ("LensCorrectionCamera", "LensCorrectionCamera"),
            ("LensCorrectionLens", "LensCorrectionLens"),
            ("LensCorrectionOperations", "LensCorrectionOperations"),
        ):
            value = _metadata_text(info.get(key))
            if value is not None:
                correction_attributes.append(
                    f'imprint:{attribute}="{escape(value, {chr(34): "&quot;"})}"'
                )
        profile_attributes = (
            '   crs:LensProfileEnable="0"\n'
            '   crs:LensProfileSetup="Custom"\n'
            '   crs:LensProfileDistortionScale="0"\n'
            '   crs:LensProfileVignettingScale="0"\n'
        )
    else:
        profile_attributes = (
            '   crs:LensProfileEnable="1"\n'
            '   crs:LensProfileSetup="Auto"\n'
            # This legacy packet requests peripheral-illumination correction
            # only; it does not claim that geometry was already transformed.
            '   crs:LensProfileDistortionScale="0"\n'
            '   crs:LensProfileVignettingScale="100"\n'
        )
    if picture_control is not None:
        correction_attributes.append(
            'imprint:NikonPictureControlName="'
            f'{escape(picture_control, {chr(34): "&quot;"})}"'
        )
    if source_camera_profile is not None:
        correction_attributes.append(
            'imprint:SourceCameraProfileName="'
            f'{escape(source_camera_profile, {chr(34): "&quot;"})}"'
        )
    if embedded_profile is not None:
        correction_attributes.append(
            'crs:CameraProfile="'
            f'{escape(embedded_profile, {chr(34): "&quot;"})}"'
        )
    raw_name = escape(source.name, {chr(34): "&quot;"})
    packet = (
        '<?xpacket begin="\ufeff" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/" x:xmptk="Imprint">\n'
        ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
        '  <rdf:Description rdf:about=""\n'
        '   xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/"\n'
        '   xmlns:aux="http://ns.adobe.com/exif/1.0/aux/"\n'
        '   xmlns:imprint="https://imprint.local/ns/lens-correction/1.0/"\n'
        '   crs:HasSettings="True"\n'
        f'{profile_attributes}'
        f'   crs:RawFileName="{raw_name}"\n'
        f'   {" ".join(auxiliary + correction_attributes)}/>\n'
        ' </rdf:RDF>\n'
        '</x:xmpmeta>\n'
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def _native_color_profile(info: dict[str, Any]) -> tuple[list[tuple[int, int]], list[tuple[int, int]], bytes] | None:
    """Validate and encode the complete camera-native DNG color profile.

    The three fields form one coherent profile: ColorMatrix1 maps XYZ to the
    camera's reference space, AsShotNeutral describes the captured white
    balance, and UniqueCameraModel identifies that space.  Falling back to
    linear sRGB is safe only when none of the fields were supplied.
    """
    keys = ("DNGColorMatrix1", "DNGAsShotNeutral", "DNGUniqueCameraModel")
    supplied = [key in info for key in keys]
    if not any(supplied):
        return None
    if not all(supplied):
        raise ValueError("camera-native DNG color metadata must include ColorMatrix1, AsShotNeutral, and UniqueCameraModel")

    matrix_value = info["DNGColorMatrix1"]
    if not isinstance(matrix_value, (tuple, list)) or len(matrix_value) != 9:
        raise ValueError("DNGColorMatrix1 must contain nine finite numbers")
    matrix_numbers = [_number(value) for value in matrix_value]
    matrix_pairs = [_rational_pair(value, signed=True) for value in matrix_value]
    if any(value is None for value in matrix_numbers) or any(pair is None for pair in matrix_pairs):
        raise ValueError("DNGColorMatrix1 must contain nine finite, representable numbers")
    values = [value for value in matrix_numbers if value is not None]
    scale = max(abs(value) for value in values)
    if scale == 0:
        raise ValueError("DNGColorMatrix1 must be non-degenerate")
    a, b, c, d, e, f, g, h, i = values
    determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g)
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12 * scale**3:
        raise ValueError("DNGColorMatrix1 must be non-degenerate")

    neutral_value = info["DNGAsShotNeutral"]
    if not isinstance(neutral_value, (tuple, list)) or len(neutral_value) != 3:
        raise ValueError("DNGAsShotNeutral must contain three positive finite numbers")
    neutral_numbers = [_number(value) for value in neutral_value]
    neutral_pairs = [_rational_pair(value) for value in neutral_value]
    if (
        any(value is None or value <= 0 for value in neutral_numbers)
        or any(pair is None for pair in neutral_pairs)
    ):
        raise ValueError("DNGAsShotNeutral must contain three positive, representable numbers")

    unique_model = info["DNGUniqueCameraModel"]
    if not isinstance(unique_model, str) or not unique_model.strip():
        raise ValueError("DNGUniqueCameraModel must be a non-empty string")
    model_bytes = _ascii_metadata(unique_model)
    if model_bytes is None:
        raise ValueError("DNGUniqueCameraModel must contain TIFF ASCII characters")

    return (
        [pair for pair in matrix_pairs if pair is not None],
        [pair for pair in neutral_pairs if pair is not None],
        model_bytes,
    )


_EMBEDDED_PROFILE_TAGS = {
    50721, 50722, 50723, 50724, 50727, 50778, 50779,
    50931, 50932, 50936, 50937, 50938, 50939, 50940,
    50941, 50942, 50964, 50965, 50981, 50982, 51107,
    51108, 51109,
}


def _embedded_profile_tags(reference: Path, expected_name: str) -> list[_Tag]:
    """Copy only a complete, copy-permitted camera profile from DNG IFD0."""
    file_size = reference.stat().st_size
    with reference.open("rb") as handle:
        header = handle.read(8)
        if len(header) != 8 or header[:4] != b"II*\x00":
            raise ValueError("profile reference must be a little-endian classic DNG")
        ifd_offset = struct.unpack_from("<I", header, 4)[0]
        if ifd_offset < 8 or ifd_offset + 2 > file_size:
            raise ValueError("profile reference has an invalid root IFD")
        handle.seek(ifd_offset)
        count = struct.unpack("<H", handle.read(2))[0]
        if count > 256 or ifd_offset + 2 + 12 * count + 4 > file_size:
            raise ValueError("profile reference root IFD is too large")
        entries = handle.read(12 * count)
        tags: dict[int, _Tag] = {}
        for index in range(count):
            entry = entries[index * 12:(index + 1) * 12]
            code, kind, item_count, value_offset = struct.unpack("<HHII", entry)
            if code not in _EMBEDDED_PROFILE_TAGS:
                continue
            if code in tags or kind not in TYPE_SIZES:
                raise ValueError("profile reference contains duplicate or unsupported tags")
            byte_count = TYPE_SIZES[kind] * item_count
            if byte_count < 1 or byte_count > 16 * 1024 * 1024:
                raise ValueError("profile reference tag has an invalid size")
            if byte_count <= 4:
                data = entry[8:8 + byte_count]
            else:
                if value_offset + byte_count > file_size:
                    raise ValueError("profile reference tag points outside the file")
                handle.seek(value_offset)
                data = handle.read(byte_count)
                if len(data) != byte_count:
                    raise ValueError("profile reference tag is truncated")
            tags[code] = _Tag(code, kind, item_count, data)
    if not {50936, 50941, 50981, 50982}.issubset(tags):
        raise ValueError("profile reference lacks embedded profile data")
    name_tag = tags[50936]
    policy_tag = tags[50941]
    if name_tag.kind != ASCII or name_tag.data.rstrip(b"\0").decode("ascii", "replace") != expected_name:
        raise ValueError("profile reference name does not match the source")
    if policy_tag.kind != LONG or policy_tag.count != 1 or struct.unpack("<I", policy_tag.data)[0] != 0:
        raise ValueError("profile reference does not permit copying")
    dims_tag = tags[50981]
    table_tag = tags[50982]
    if dims_tag.kind != LONG or dims_tag.count != 3:
        raise ValueError("profile reference has invalid look-table dimensions")
    hue, saturation, value = struct.unpack("<III", dims_tag.data)
    if (hue < 1 or saturation < 2 or value < 1 or hue * saturation * value > 1_000_000
            or table_tag.kind != FLOAT or table_tag.count != hue * saturation * value * 3):
        raise ValueError("profile reference has an incomplete look table")
    return list(tags.values())


def _build_ifd(tags: list[_Tag], offset: int, next_ifd: int = 0) -> tuple[bytes, dict[int, int]]:
    tags = sorted(tags, key=lambda item: item.code)
    table_size = 2 + len(tags) * 12 + 4
    extra_offset = offset + table_size
    entries = bytearray(struct.pack("<H", len(tags)))
    extras = bytearray()
    data_offsets: dict[int, int] = {}
    for item in tags:
        entries += struct.pack("<HHI", item.code, item.kind, item.count)
        if len(item.data) <= 4:
            entries += item.data.ljust(4, b"\0")
        else:
            if extra_offset % 2:
                extras += b"\0"
                extra_offset += 1
            entries += struct.pack("<I", extra_offset)
            data_offsets[item.code] = extra_offset
            extras += item.data
            extra_offset += len(item.data)
    entries += struct.pack("<I", next_ifd)
    return bytes(entries + extras), data_offsets


def _quantize_samples(image_rgb16: np.ndarray, bits_per_sample: int) -> np.ndarray:
    """Scale full-range uint16 pixels to the requested integer sample range."""
    maximum = (1 << bits_per_sample) - 1
    if bits_per_sample == 16:
        return image_rgb16
    scaled = (
        image_rgb16.astype(np.uint32) * maximum + 32767
    ) // 65535
    return scaled.astype(np.uint16)


def _packed_strip_bytes(strip: np.ndarray, bits_per_sample: int) -> bytes:
    """Return a TIFF strip with MSB-first packed 10/12/14-bit samples.

    DNG packs all samples in a strip continuously; padding is added only once
    at the end of the strip, not at every scanline boundary. Work is limited
    to one strip so full-resolution photos do not need a second image-sized
    bit buffer.
    """
    if bits_per_sample == 16:
        return strip.astype("<u2", copy=False).tobytes(order="C")
    if bits_per_sample == 8:
        return strip.astype(np.uint8).tobytes(order="C")
    samples = strip.reshape(-1).astype(">u2", copy=False)
    bits = np.unpackbits(samples.view(np.uint8)).reshape(samples.size, 16)
    return np.packbits(bits[:, 16 - bits_per_sample:].reshape(-1)).tobytes()


def _jpeg_preview(image: np.ndarray, max_edge: int = 1024) -> tuple[bytes, int, int]:
    preview = Image.fromarray((image >> 8).astype(np.uint8), mode="RGB")
    preview.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
    stream = io.BytesIO()
    preview.save(stream, "JPEG", quality=88, optimize=True)
    return stream.getvalue(), preview.width, preview.height


def _unique_output_path(output_dir: Path, source: Path) -> Path:
    candidate = output_dir / f"{source.stem}_dehaze.dng"
    index = 2
    while candidate.exists():
        candidate = output_dir / f"{source.stem}_dehaze_{index}.dng"
        index += 1
    return candidate


def write_linear_dng(
    image_rgb16: np.ndarray,
    source_path: str | Path,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    bits_per_sample: int = 16,
) -> Path:
    """Atomically create a lossless RGB Linear DNG and return its path."""
    if image_rgb16.dtype != np.uint16 or image_rgb16.ndim != 3 or image_rgb16.shape[2] != 3:
        raise ValueError("Linear DNG input must be a uint16 RGB array")
    if bits_per_sample not in {8, 10, 12, 14, 16}:
        raise ValueError("Linear DNG bit depth must be 8, 10, 12, 14, or 16")
    source = Path(source_path)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = _unique_output_path(destination_dir, source)
    height, width = image_rgb16.shape[:2]
    row_bytes = (width * 3 * bits_per_sample + 7) // 8
    rows_per_strip = max(1, min(height, max(1, 1024 * 1024 // max(1, row_bytes))))
    strip_rows = [
        (row, min(height, row + rows_per_strip))
        for row in range(0, height, rows_per_strip)
    ]
    strip_byte_counts = [
        ((end - start) * width * 3 * bits_per_sample + 7) // 8
        for start, end in strip_rows
    ]
    preview, preview_width, preview_height = _jpeg_preview(image_rgb16)
    info = metadata or {}
    native_color_profile = _native_color_profile(info)
    make = _metadata_text(info.get("Make")) or "Imprint"
    model = _metadata_text(info.get("Model")) or "RGB Linear DNG"
    captured = (
        _datetime_text(info.get("DateTime"))
        or _datetime_text(info.get("DateTimeOriginal"))
        or datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    )
    exif_tags = _exif_tags(info, width, height)
    gps_tags = _gps_tags(info)
    artist = _ascii_metadata(info.get("Artist"))
    copyright_text = _ascii_metadata(info.get("Copyright"))
    acr_xmp = _acr_lens_xmp(source, info)
    lens_info = info.get("LensSpecification")
    correction_description = None
    if info.get("LensCorrectionApplied"):
        operations = _metadata_text(info.get("LensCorrectionOperations")) or "lens correction"
        correction_description = _ascii_metadata(
            f"Imprint baked Lensfun correction into pixels: {operations}"
        )

    # Layout: TIFF header, primary IFD, full-res strips, preview IFD, JPEG preview.
    color_matrix = (
        _rationals(native_color_profile[0], signed=True)
        if native_color_profile is not None
        else _rationals(
            [(3133856, 1000000), (-1616867, 1000000), (-490615, 1000000),
             (-978768, 1000000), (1916142, 1000000), (33454, 1000000),
             (71945, 1000000), (-228991, 1000000), (1405243, 1000000)],
            True,
        )
    )
    as_shot_neutral = (
        _rationals(native_color_profile[1])
        if native_color_profile is not None
        else _rationals([(1, 1), (1, 1), (1, 1)])
    )
    unique_camera_model = (
        native_color_profile[2]
        if native_color_profile is not None
        else _ascii(f"{make} {model}")
    )
    primary_tags = [
        _tag(254, LONG, _longs([0])), _tag(256, LONG, _longs([width])),
        _tag(257, LONG, _longs([height])),
        _tag(258, SHORT, _shorts([bits_per_sample] * 3)),
        # Integer LinearRaw with arbitrary 8..16 bit samples is stored
        # uncompressed. DNG Deflate is not permitted for 10/12/14-bit integer
        # raw data and Camera Raw rejects that otherwise TIFF-readable pairing.
        _tag(259, SHORT, _shorts([1])), _tag(262, SHORT, _shorts([34892])),
        _tag(271, ASCII, _ascii(make)), _tag(272, ASCII, _ascii(model)),
        _tag(273, LONG, _longs([0] * len(strip_rows))), _tag(274, SHORT, _shorts([1])),
        _tag(277, SHORT, _shorts([3])), _tag(278, LONG, _longs([rows_per_strip])),
        _tag(279, LONG, _longs(strip_byte_counts)),
        _tag(284, SHORT, _shorts([1])), _tag(305, ASCII, _ascii("Imprint Natural Dehaze")),
        _tag(306, ASCII, _ascii(captured)),
        # ExifIFD is written below the image strips; the pointer is filled in
        # after all variable-size IFD data has been laid out.
        _tag(34665, LONG, _longs([0])),
        _tag(339, SHORT, _shorts([1, 1, 1])),
        _tag(50706, BYTE, bytes([1, 4, 0, 0]), 4), _tag(50707, BYTE, bytes([1, 4, 0, 0]), 4),
        _tag(50708, ASCII, unique_camera_model),
        _tag(50717, LONG, _longs([(1 << bits_per_sample) - 1] * 3)),
        _tag(50719, LONG, _longs([0, 0])), _tag(50720, LONG, _longs([width, height])),
        # Keep the established linear-sRGB profile when no camera-native
        # profile is supplied; otherwise write the caller's complete profile.
        _tag(50721, SRATIONAL, color_matrix),
        _tag(50728, RATIONAL, as_shot_neutral),
        _tag(50730, SRATIONAL, _rationals([(0, 10000)], True)),
        _tag(50734, RATIONAL, _rationals([(1, 1)])), _tag(50778, SHORT, _shorts([21])),
        _tag(50780, RATIONAL, _rationals([(1, 1)])), _tag(50829, LONG, _longs([0, 0, height, width])),
    ]
    if correction_description is not None:
        primary_tags.append(_tag(270, ASCII, correction_description))
    if isinstance(lens_info, (tuple, list)) and len(lens_info) == 4:
        lens_pairs = [_rational_pair(value) for value in lens_info]
        if all(pair is not None for pair in lens_pairs):
            primary_tags.append(_tag(50736, RATIONAL, _rationals(lens_pairs)))
    if artist is not None:
        primary_tags.append(_tag(315, ASCII, artist))
    if copyright_text is not None:
        primary_tags.append(_tag(33432, ASCII, copyright_text))
    if acr_xmp is not None:
        primary_tags.append(_tag(700, BYTE, acr_xmp, len(acr_xmp)))
    if gps_tags:
        primary_tags.append(_tag(34853, LONG, _longs([0])))

    primary_placeholder, _ = _build_ifd(primary_tags, 8)
    strip_start = 8 + len(primary_placeholder)
    strip_offsets = []
    cursor = strip_start
    for byte_count in strip_byte_counts:
        strip_offsets.append(cursor)
        cursor += byte_count

    exif_ifd_offset = cursor
    exif_ifd, _ = _build_ifd(exif_tags, exif_ifd_offset)
    cursor += len(exif_ifd)
    gps_ifd_offset = cursor if gps_tags else 0
    gps_ifd = b""
    if gps_tags:
        gps_ifd, _ = _build_ifd(gps_tags, gps_ifd_offset)
        cursor += len(gps_ifd)

    preview_ifd_offset = cursor
    preview_tags = [
        _tag(254, LONG, _longs([1])), _tag(256, LONG, _longs([preview_width])),
        _tag(257, LONG, _longs([preview_height])), _tag(258, SHORT, _shorts([8, 8, 8])),
        _tag(259, SHORT, _shorts([7])), _tag(262, SHORT, _shorts([6])),
        _tag(273, LONG, _longs([0])), _tag(277, SHORT, _shorts([3])),
        _tag(278, LONG, _longs([preview_height])), _tag(279, LONG, _longs([len(preview)])),
        _tag(284, SHORT, _shorts([1])),
    ]
    preview_placeholder, _ = _build_ifd(preview_tags, preview_ifd_offset)
    preview_data_offset = preview_ifd_offset + len(preview_placeholder)
    preview_tags = [item if item.code != 273 else _tag(273, LONG, _longs([preview_data_offset])) for item in preview_tags]
    preview_ifd, _ = _build_ifd(preview_tags, preview_ifd_offset)
    primary_tags = [item if item.code != 273 else _tag(273, LONG, _longs(strip_offsets)) for item in primary_tags]
    primary_tags = [item if item.code != 34665 else _tag(34665, LONG, _longs([exif_ifd_offset])) for item in primary_tags]
    if gps_tags:
        primary_tags = [item if item.code != 34853 else _tag(34853, LONG, _longs([gps_ifd_offset])) for item in primary_tags]
    primary_ifd, _ = _build_ifd(primary_tags, 8, preview_ifd_offset)

    fd, temporary_name = tempfile.mkstemp(prefix=f".{final_path.stem}-", suffix=".tmp", dir=destination_dir)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(b"II*\x00" + struct.pack("<I", 8))
            handle.write(primary_ifd)
            for (start, end), expected_size in zip(strip_rows, strip_byte_counts):
                quantized_strip = _quantize_samples(
                    image_rgb16[start:end],
                    bits_per_sample,
                )
                packed_strip = _packed_strip_bytes(quantized_strip, bits_per_sample)
                if len(packed_strip) != expected_size:
                    raise OSError("DNG strip packing produced an invalid byte count")
                handle.write(packed_strip)
            handle.write(exif_ifd)
            if gps_ifd:
                handle.write(gps_ifd)
            handle.write(preview_ifd)
            handle.write(preview)
            handle.flush()
            os.fsync(handle.fileno())
        with temporary.open("rb") as check:
            signature = check.read(4)
        if temporary.stat().st_size < 1024 or signature != b"II*\x00":
            raise OSError("DNG validation failed before commit")
        with _OUTPUT_LOCK:
            # Re-check inside the process-wide commit lock so simultaneous jobs can
            # never replace one another's output.
            final_path = _unique_output_path(destination_dir, source)
            os.replace(temporary, final_path)
        return final_path
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


_DNG_CODEC_STRIP_BYTES = 16 * 1024 * 1024
_UINT32_MAX = (1 << 32) - 1


def _dng_strip_layout(width: int, height: int, samples_per_pixel: int) -> tuple[int, list[tuple[int, int]]]:
    """Return strips whose decoded uint16 data is bounded to 16 MiB each."""
    row_bytes = width * samples_per_pixel * 2
    rows_per_strip = max(1, min(height, max(1, _DNG_CODEC_STRIP_BYTES // max(1, row_bytes))))
    rows = [
        (start, min(height, start + rows_per_strip))
        for start in range(0, height, rows_per_strip)
    ]
    return rows_per_strip, rows


def _imagecodecs_encoder(name: str):
    """Return an optional imagecodecs encoder without making it a hard import."""
    try:
        import imagecodecs
    except ImportError:
        return None
    return getattr(imagecodecs, name, None)


def _codec_executable(name: str) -> str | None:
    executable = shutil.which(name)
    if executable is not None:
        return executable
    for candidate in (Path("/opt/homebrew/bin") / name, Path("/usr/local/bin") / name):
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


def _pnm16_bytes(samples: np.ndarray) -> bytes:
    """Serialize a 16-bit grayscale or RGB strip as big-endian PNM."""
    if samples.dtype != np.uint16 or samples.ndim not in {2, 3}:
        raise ValueError("codec input must be a uint16 grayscale or RGB strip")
    if samples.ndim == 3 and samples.shape[2] != 3:
        raise ValueError("codec RGB strip must have three channels")
    height, width = samples.shape[:2]
    magic = b"P5" if samples.ndim == 2 else b"P6"
    header = magic + f"\n{width} {height}\n65535\n".encode("ascii")
    return header + np.ascontiguousarray(samples.astype(">u2", copy=False)).tobytes(order="C")


def _encode_lossless_jpeg(strip: np.ndarray) -> bytes:
    encoder = _imagecodecs_encoder("ljpeg_encode")
    encoded = None
    if encoder is not None:
        try:
            encoded = bytes(encoder(strip, bitspersample=16))
        except ImportError:
            # imagecodecs exposes delayed-import stubs for optional codecs.
            encoded = None
    if encoded is None:
        cjpeg = _codec_executable("cjpeg")
        if cjpeg is None:
            raise RuntimeError(
                "lossless CFA JPEG encoding requires imagecodecs.ljpeg_encode or cjpeg"
            )
        with tempfile.TemporaryDirectory(prefix="imprint-ljpeg-") as work_dir:
            input_path = Path(work_dir) / "strip.pgm"
            output_path = Path(work_dir) / "strip.jpg"
            input_path.write_bytes(_pnm16_bytes(strip))
            subprocess.run(
                [
                    cjpeg, "-precision", "16", "-lossless", "1", "-grayscale",
                    "-outfile", str(output_path), str(input_path),
                ],
                check=True,
                capture_output=True,
            )
            encoded = output_path.read_bytes()
    if len(encoded) < 4 or not encoded.startswith(b"\xff\xd8") or not encoded.endswith(b"\xff\xd9"):
        raise OSError("lossless JPEG encoder returned an invalid stream")
    return encoded


def _encode_jpeg_xl(strip: np.ndarray) -> bytes:
    """Encode enhanced RGB using the DNG 1.7 parameters seen in an ACR sample.

    Adobe Camera Raw records JXLDistance=0.01 and JXLEffort=7 on its
    LinearRaw RGB SubIFD. imagecodecs defaults to linear sRGB; the CLI hint
    below keeps its interpretation of these linear sample values consistent.
    """
    encoder = _imagecodecs_encoder("jpegxl_encode")
    encoded = None
    if encoder is not None:
        try:
            encoded = bytes(
                encoder(
                    strip,
                    distance=0.01,
                    lossless=False,
                    effort=7,
                    usecontainer=False,
                )
            )
        except ImportError:
            # imagecodecs exposes delayed-import stubs for optional codecs.
            encoded = None
    if encoded is None:
        cjxl = _codec_executable("cjxl")
        if cjxl is None:
            raise RuntimeError(
                "RGB JPEG XL encoding requires imagecodecs.jpegxl_encode or cjxl"
            )
        with tempfile.TemporaryDirectory(prefix="imprint-jxl-") as work_dir:
            input_path = Path(work_dir) / "strip.ppm"
            output_path = Path(work_dir) / "strip.jxl"
            input_path.write_bytes(_pnm16_bytes(strip))
            subprocess.run(
                [
                    cjxl, str(input_path), str(output_path), "-d", "0.01", "-e", "7",
                    "--container=0", "-x", "color_space=RGB_D65_SRG_Rel_Lin", "--quiet",
                ],
                check=True,
                capture_output=True,
            )
            encoded = output_path.read_bytes()
    if len(encoded) < 2 or not (encoded.startswith(b"\xff\x0a") or encoded[4:8] == b"JXL "):
        raise OSError("JPEG XL encoder returned an invalid stream")
    return encoded


def _compress_strips(
    image: np.ndarray,
    strips: list[tuple[int, int]],
    encoder,
    spool,
) -> list[int]:
    """Write encoded or raw image strips to a seekable spool and return bytecounts."""
    byte_counts = []
    for start, end in strips:
        encoded = encoder(image[start:end])
        byte_count = len(encoded)
        if byte_count < 1 or byte_count > _UINT32_MAX:
            raise OSError("DNG compressed strip size is outside classic TIFF limits")
        spool.write(encoded)
        byte_counts.append(byte_count)
    return byte_counts


def _aligned_ifd(tags: list[_Tag], offset: int, next_ifd: int = 0) -> bytes:
    """Build an IFD and pad its end so the following directory is word-aligned."""
    ifd, _ = _build_ifd(tags, offset, next_ifd)
    return ifd + (b"\0" if len(ifd) % 2 else b"")


def _bounded_preview_rgb8(image: np.ndarray, max_edge: int) -> np.ndarray:
    """Make a bounded RGB8 preview from RGB8 sRGB or legacy uint16 samples.

    When the source is larger than the requested edge, center-sample directly
    into the bounded result. This avoids allocating a second full-resolution
    RGB image just to discard most of it. The normal UI path supplies an
    already resized uint8 display-referred sRGB image.
    """
    height, width = image.shape[:2]
    scale = min(1.0, max_edge / max(height, width))
    target_width = max(1, int(round(width * scale)))
    target_height = max(1, int(round(height * scale)))
    if image.dtype == np.uint8 and (target_width, target_height) == (width, height):
        return image if image.flags.c_contiguous else np.ascontiguousarray(image)
    output = np.empty((target_height, target_width, 3), dtype=np.uint8)

    if (target_width, target_height) == (width, height):
        row_indices = np.arange(height)
        column_indices = np.arange(width)
    else:
        row_indices = np.minimum(
            height - 1,
            ((np.arange(target_height, dtype=np.float64) + 0.5) * height / target_height).astype(np.int64),
        )
        column_indices = np.minimum(
            width - 1,
            ((np.arange(target_width, dtype=np.float64) + 0.5) * width / target_width).astype(np.int64),
        )

    # Bound temporaries even when a caller gives us a very tall source image.
    rows_per_chunk = max(1, min(target_height, 128))
    for start in range(0, target_height, rows_per_chunk):
        end = min(target_height, start + rows_per_chunk)
        selected = image[row_indices[start:end, None], column_indices[None, :], :]
        if image.dtype == np.uint16:
            np.right_shift(selected, 8, out=selected)
        output[start:end] = selected.astype(np.uint8, copy=False)
    return output


def _enhanced_preview_images(
    image: np.ndarray,
    *,
    preview_max_edge: int = 4096,
    thumbnail_max_edge: int = 256,
) -> tuple[bytes, int, int, bytes, int, int]:
    """Return bounded baseline-JPEG preview and raw RGB8 IFD0 thumbnail."""
    rgb8 = _bounded_preview_rgb8(image, preview_max_edge)
    full_preview = Image.fromarray(rgb8, mode="RGB")
    full_stream = io.BytesIO()
    full_preview.save(
        full_stream,
        "JPEG",
        quality=88,
        optimize=True,
        progressive=False,
        subsampling=2,
    )

    thumbnail = full_preview.resize(
        _preview_size(full_preview.width, full_preview.height, thumbnail_max_edge),
        Image.Resampling.LANCZOS,
    )
    return (
        full_stream.getvalue(),
        full_preview.width,
        full_preview.height,
        thumbnail.tobytes(),
        thumbnail.width,
        thumbnail.height,
    )


def _preview_size(width: int, height: int, max_edge: int) -> tuple[int, int]:
    scale = min(1.0, max_edge / max(width, height))
    return max(1, int(round(width * scale))), max(1, int(round(height * scale)))


def write_enhanced_dng(
    enhanced_rgb16: np.ndarray,
    raw_cfa16: np.ndarray,
    cfa_pattern: np.ndarray | Iterable[Iterable[int]],
    source_path: str | Path,
    output_dir: str | Path,
    metadata: dict[str, Any] | None = None,
    *,
    orientation: int = 1,
    preview_rgb16: np.ndarray | None = None,
) -> Path:
    """Atomically write a DNG containing original CFA and enhanced RGB data.

    IFD0 is a reduced RGB thumbnail. Its two SubIFDs hold the untouched sensor
    CFA samples and the full-resolution enhanced LinearRaw RGB samples; its
    next IFD is a display-referred sRGB JPEG preview. ``preview_rgb16`` accepts
    uint8 display-referred sRGB or the legacy uint16 representation. The two
    full-resolution arrays must already share sensor orientation.
    """
    if (
        not isinstance(enhanced_rgb16, np.ndarray)
        or enhanced_rgb16.dtype != np.uint16
        or enhanced_rgb16.ndim != 3
        or enhanced_rgb16.shape[2] != 3
    ):
        raise ValueError("Enhanced DNG RGB input must be a uint16 HxWx3 array")
    if (
        not isinstance(raw_cfa16, np.ndarray)
        or raw_cfa16.dtype != np.uint16
        or raw_cfa16.ndim != 2
    ):
        raise ValueError("Enhanced DNG CFA input must be a uint16 HxW array")
    height, width = enhanced_rgb16.shape[:2]
    if height < 1 or width < 1 or raw_cfa16.shape != (height, width):
        raise ValueError("Enhanced RGB and CFA inputs must have matching non-empty dimensions")
    if orientation not in {1, 3, 6, 8}:
        raise ValueError("DNG orientation must be one of 1, 3, 6, or 8")
    if preview_rgb16 is not None and (
        not isinstance(preview_rgb16, np.ndarray)
        or preview_rgb16.dtype not in {np.dtype(np.uint8), np.dtype(np.uint16)}
        or preview_rgb16.ndim != 3
        or preview_rgb16.shape[2] != 3
        or preview_rgb16.shape[0] < 1
        or preview_rgb16.shape[1] < 1
    ):
        raise ValueError("DNG preview input must be a non-empty uint8 or uint16 HxWx3 array")

    pattern = np.asarray(cfa_pattern)
    if (
        pattern.shape != (2, 2)
        or pattern.dtype == np.bool_
        or not np.issubdtype(pattern.dtype, np.integer)
        or np.any(pattern < 0)
        or np.any(pattern > 2)
        or tuple(np.bincount(pattern.reshape(-1).astype(np.int64), minlength=3)) != (1, 2, 1)
    ):
        raise ValueError("CFA pattern must be a 2x2 Bayer pattern with one red, two green, and one blue index")
    pattern_bytes = bytes(int(value) for value in pattern.reshape(-1))

    info = dict(metadata or {})
    # A source profile name becomes an active ACR selection only when the
    # matching DNG profile data has actually been embedded below.
    info.pop("DNGEmbeddedProfileName", None)
    profile_reference = info.get("DNGProfileReferencePath")
    embedded_profile_tags: list[_Tag] = []
    if profile_reference is not None:
        expected_name = _metadata_text(info.get("SourceCameraProfileName"))
        if expected_name is None:
            raise ValueError("embedded camera profile requires a source profile name")
        embedded_profile_tags = _embedded_profile_tags(Path(profile_reference), expected_name)
        info["DNGEmbeddedProfileName"] = expected_name
    native_color_profile = _native_color_profile(info)
    if native_color_profile is None:
        raise ValueError("Enhanced CFA DNG requires complete camera-native color metadata")

    black_value = info.get("DNGBlackLevel")
    if not isinstance(black_value, (tuple, list)) or len(black_value) != 4:
        raise ValueError("DNGBlackLevel must contain four CFA black levels")
    black_levels = [_short_value(value, minimum=0) for value in black_value]
    white_level = _short_value(info.get("DNGWhiteLevel"), minimum=1)
    if any(value is None for value in black_levels) or white_level is None:
        raise ValueError("DNG CFA black and white levels must be integers in the uint16 range")
    black_level_values = [int(value) for value in black_levels if value is not None]
    if any(value >= white_level for value in black_level_values):
        raise ValueError("DNGWhiteLevel must be greater than every CFA black level")
    crop_origin = info.get("DNGDefaultCropOrigin", (0, 0))
    crop_size = info.get("DNGDefaultCropSize", (width, height))
    if (
        not isinstance(crop_origin, (tuple, list)) or len(crop_origin) != 2
        or not isinstance(crop_size, (tuple, list)) or len(crop_size) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in (*crop_origin, *crop_size))
        or crop_origin[0] < 0 or crop_origin[1] < 0
        or crop_size[0] < 1 or crop_size[1] < 1
        or crop_origin[0] + crop_size[0] > width
        or crop_origin[1] + crop_size[1] > height
    ):
        raise ValueError("DNG crop must fit within the original CFA image")

    source = Path(source_path)
    destination_dir = Path(output_dir).expanduser().resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    final_path = _unique_output_path(destination_dir, source)
    make = _metadata_text(info.get("Make")) or "Imprint"
    model = _metadata_text(info.get("Model")) or "Camera RAW"
    captured = (
        _datetime_text(info.get("DateTime"))
        or _datetime_text(info.get("DateTimeOriginal"))
        or datetime.now().strftime("%Y:%m:%d %H:%M:%S")
    )
    (
        preview_jpeg,
        preview_width,
        preview_height,
        thumbnail_rgb,
        thumbnail_width,
        thumbnail_height,
    ) = _enhanced_preview_images(
        enhanced_rgb16 if preview_rgb16 is None else preview_rgb16
    )
    exif_tags = _exif_tags(info, width, height)
    gps_tags = _gps_tags(info)
    acr_xmp = _acr_lens_xmp(source, info)
    artist = _ascii_metadata(info.get("Artist"))
    copyright_text = _ascii_metadata(info.get("Copyright"))

    compressed_spool = tempfile.TemporaryFile(mode="w+b")
    try:
        raw_rows_per_strip, raw_strips = _dng_strip_layout(width, height, 1)
        rgb_rows_per_strip, rgb_strips = _dng_strip_layout(width, height, 3)
        raw_byte_counts = _compress_strips(
            raw_cfa16, raw_strips, lambda strip: _packed_strip_bytes(strip, 16), compressed_spool
        )
        rgb_byte_counts = _compress_strips(
            enhanced_rgb16, rgb_strips, _encode_jpeg_xl, compressed_spool
        )
        compressed_size = sum(raw_byte_counts) + sum(rgb_byte_counts)
        if compressed_spool.tell() != compressed_size:
            raise OSError("DNG compression spool has an invalid size")
        color_matrix = _rationals(native_color_profile[0], signed=True)
        as_shot_neutral = _rationals(native_color_profile[1])
        unique_camera_model = native_color_profile[2]

        # IFD0 is a small, uncompressed RGB thumbnail for Finder and other
        # readers that only inspect the first TIFF directory. The high-resolution
        # JPEG preview is linked through IFD0's next-IFD pointer; the raw and
        # enhanced images remain its two DNG SubIFDs.
        root_tags = [
            _tag(254, LONG, _longs([1])),
            _tag(256, LONG, _longs([thumbnail_width])),
            _tag(257, LONG, _longs([thumbnail_height])),
            _tag(258, SHORT, _shorts([8, 8, 8])),
            _tag(259, SHORT, _shorts([1])),
            _tag(262, SHORT, _shorts([2])),
            _tag(271, ASCII, _ascii(make)),
            _tag(272, ASCII, _ascii(model)),
            _tag(273, LONG, _longs([0])),
            _tag(274, SHORT, _shorts([orientation])),
            _tag(277, SHORT, _shorts([3])),
            _tag(278, LONG, _longs([thumbnail_height])),
            _tag(279, LONG, _longs([len(thumbnail_rgb)])),
            _tag(284, SHORT, _shorts([1])),
            _tag(305, ASCII, _ascii("Imprint Natural Dehaze")),
            _tag(306, ASCII, _ascii(captured)),
            _tag(330, LONG, _longs([0, 0])),
            _tag(34665, LONG, _longs([0])),
            _tag(339, SHORT, _shorts([1, 1, 1])),
            _tag(50706, BYTE, bytes((1, 7, 0, 0)), 4),
            _tag(50707, BYTE, bytes((1, 7, 0, 0)), 4),
            _tag(50708, ASCII, unique_camera_model),
            _tag(50721, SRATIONAL, color_matrix),
            _tag(50728, RATIONAL, as_shot_neutral),
            _tag(50778, SHORT, _shorts([21])),  # LibRaw matrix uses D65
            _tag(50879, SHORT, _shorts([0])),  # scene-referred colorimetric reference
            _tag(50970, LONG, _longs([2])),  # PreviewColorSpace: sRGB
        ]
        if acr_xmp is not None:
            root_tags.append(_tag(700, BYTE, acr_xmp, len(acr_xmp)))
        if artist is not None:
            root_tags.append(_tag(315, ASCII, artist))
        if copyright_text is not None:
            root_tags.append(_tag(33432, ASCII, copyright_text))
        if gps_tags:
            root_tags.append(_tag(34853, LONG, _longs([0])))
        if embedded_profile_tags:
            root_by_code = {item.code: item for item in root_tags}
            root_by_code.update({item.code: item for item in embedded_profile_tags})
            root_by_code[50934] = _tag(50934, ASCII, _ascii(info["DNGEmbeddedProfileName"]))
            root_tags = list(root_by_code.values())

        raw_tags = [
            _tag(254, LONG, _longs([0])),
            _tag(256, LONG, _longs([width])),
            _tag(257, LONG, _longs([height])),
            _tag(258, SHORT, _shorts([16])),
            _tag(259, SHORT, _shorts([1])),
            _tag(262, SHORT, _shorts([32803])),
            _tag(273, LONG, _longs([0] * len(raw_strips))),
            _tag(274, SHORT, _shorts([orientation])),
            _tag(277, SHORT, _shorts([1])),
            _tag(278, LONG, _longs([raw_rows_per_strip])),
            _tag(279, LONG, _longs(raw_byte_counts)),
            _tag(284, SHORT, _shorts([1])),
            _tag(33421, SHORT, _shorts([2, 2])),
            _tag(33422, BYTE, pattern_bytes, 4),
            _tag(50710, BYTE, bytes((0, 1, 2)), 3),
            _tag(50711, SHORT, _shorts([1])),  # rectangular CFA layout
            _tag(50713, SHORT, _shorts([2, 2])),
            _tag(50714, RATIONAL, _rationals([(value, 1) for value in black_level_values])),
            _tag(50717, LONG, _longs([white_level])),
            _tag(50719, LONG, _longs(crop_origin)),
            _tag(50720, LONG, _longs(crop_size)),
            _tag(50829, LONG, _longs([0, 0, height, width])),
        ]
        enhanced_tags = [
            _tag(254, LONG, _longs([16])),
            _tag(256, LONG, _longs([width])),
            _tag(257, LONG, _longs([height])),
            _tag(258, SHORT, _shorts([16, 16, 16])),
            _tag(259, SHORT, _shorts([52546])),
            _tag(262, SHORT, _shorts([34892])),
            _tag(273, LONG, _longs([0] * len(rgb_strips))),
            _tag(274, SHORT, _shorts([orientation])),
            _tag(277, SHORT, _shorts([3])),
            _tag(278, LONG, _longs([rgb_rows_per_strip])),
            _tag(279, LONG, _longs(rgb_byte_counts)),
            _tag(284, SHORT, _shorts([1])),
            _tag(339, SHORT, _shorts([1, 1, 1])),
        _tag(50714, RATIONAL, _rationals([(0, 1), (0, 1), (0, 1)])),
        _tag(50717, LONG, _longs([65535, 65535, 65535])),
        _tag(51182, ASCII, _ascii("Imprint Dehaze")),
    ]

        preview_tags = [
            _tag(254, LONG, _longs([1])),  # reduced-resolution PreviewIFD
            _tag(256, LONG, _longs([preview_width])),
            _tag(257, LONG, _longs([preview_height])),
            _tag(258, SHORT, _shorts([8, 8, 8])),
            _tag(259, SHORT, _shorts([7])),  # baseline JPEG
            _tag(262, SHORT, _shorts([6])),  # YCbCr JPEG components
            _tag(273, LONG, _longs([0])),
            _tag(274, SHORT, _shorts([orientation])),
            _tag(277, SHORT, _shorts([3])),
            _tag(278, LONG, _longs([preview_height])),
            _tag(279, LONG, _longs([len(preview_jpeg)])),
            _tag(284, SHORT, _shorts([1])),
            _tag(530, SHORT, _shorts([2, 2])),  # JPEG's 4:2:0 chroma sampling
            _tag(531, SHORT, _shorts([1])),  # centered chroma positioning
            _tag(50970, LONG, _longs([2])),  # PreviewColorSpace: sRGB
        ]

        # Measure each directory with zero-valued pointer placeholders. Replacing
        # placeholders with final offsets does not change their serialized sizes.
        root_placeholder = _aligned_ifd(root_tags, 8)
        raw_ifd_offset = 8 + len(root_placeholder)
        raw_placeholder = _aligned_ifd(raw_tags, raw_ifd_offset)
        enhanced_ifd_offset = raw_ifd_offset + len(raw_placeholder)
        enhanced_placeholder = _aligned_ifd(enhanced_tags, enhanced_ifd_offset)
        exif_ifd_offset = enhanced_ifd_offset + len(enhanced_placeholder)
        exif_ifd = _aligned_ifd(exif_tags, exif_ifd_offset)
        gps_ifd_offset = exif_ifd_offset + len(exif_ifd) if gps_tags else 0
        gps_ifd = _aligned_ifd(gps_tags, gps_ifd_offset) if gps_tags else b""
        preview_ifd_offset = exif_ifd_offset + len(exif_ifd) + len(gps_ifd)
        preview_ifd_placeholder = _aligned_ifd(preview_tags, preview_ifd_offset)
        image_data_offset = preview_ifd_offset + len(preview_ifd_placeholder)

        raw_offsets: list[int] = []
        cursor = image_data_offset
        for byte_count in raw_byte_counts:
            raw_offsets.append(cursor)
            cursor += byte_count
        rgb_offsets: list[int] = []
        for byte_count in rgb_byte_counts:
            rgb_offsets.append(cursor)
            cursor += byte_count
        thumbnail_offset = cursor
        preview_jpeg_offset = thumbnail_offset + len(thumbnail_rgb)
        output_end = preview_jpeg_offset + len(preview_jpeg)
        if output_end > _UINT32_MAX:
            raise OSError("DNG output exceeds classic TIFF 32-bit offset limits")

        root_tags = [
            item if item.code != 273 else _tag(273, LONG, _longs([thumbnail_offset]))
            if item.code not in {330, 34665, 34853}
            else item
            for item in root_tags
        ]
        root_tags = [
            _tag(330, LONG, _longs([raw_ifd_offset, enhanced_ifd_offset]))
            if item.code == 330
            else _tag(34665, LONG, _longs([exif_ifd_offset]))
            if item.code == 34665
            else _tag(34853, LONG, _longs([gps_ifd_offset]))
            if item.code == 34853
            else item
            for item in root_tags
        ]
        raw_tags = [
            _tag(273, LONG, _longs(raw_offsets)) if item.code == 273 else item
            for item in raw_tags
        ]
        enhanced_tags = [
            _tag(273, LONG, _longs(rgb_offsets)) if item.code == 273 else item
            for item in enhanced_tags
        ]
        preview_tags = [
            _tag(273, LONG, _longs([preview_jpeg_offset])) if item.code == 273 else item
            for item in preview_tags
        ]

        root_ifd = _aligned_ifd(root_tags, 8, preview_ifd_offset)
        raw_ifd = _aligned_ifd(raw_tags, raw_ifd_offset)
        enhanced_ifd = _aligned_ifd(enhanced_tags, enhanced_ifd_offset)
        preview_ifd = _aligned_ifd(preview_tags, preview_ifd_offset)
        if (len(root_ifd), len(raw_ifd), len(enhanced_ifd), len(preview_ifd)) != (
            len(root_placeholder), len(raw_placeholder), len(enhanced_placeholder), len(preview_ifd_placeholder)
        ):
            raise OSError("DNG IFD layout changed while resolving offsets")

        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{final_path.stem}-", suffix=".tmp", dir=destination_dir
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(b"II*\x00" + struct.pack("<I", 8))
                handle.write(root_ifd)
                handle.write(raw_ifd)
                handle.write(enhanced_ifd)
                handle.write(exif_ifd)
                if gps_ifd:
                    handle.write(gps_ifd)
                handle.write(preview_ifd)
                compressed_spool.seek(0)
                shutil.copyfileobj(compressed_spool, handle, length=1024 * 1024)
                handle.write(thumbnail_rgb)
                handle.write(preview_jpeg)
                handle.flush()
                os.fsync(handle.fileno())

            with temporary.open("rb") as check:
                signature = check.read(4)
            if temporary.stat().st_size != output_end or signature != b"II*\x00":
                raise OSError("DNG validation failed before commit")

            # A hard link is an atomic create-if-absent operation on the same
            # filesystem. It keeps the complete temporary file visible at once
            # without allowing another process to overwrite an existing DNG.
            with _OUTPUT_LOCK:
                index = 1
                while True:
                    candidate = destination_dir / (
                        f"{source.stem}_dehaze.dng"
                        if index == 1
                        else f"{source.stem}_dehaze_{index}.dng"
                    )
                    try:
                        os.link(temporary, candidate)
                    except FileExistsError:
                        index += 1
                        continue
                    final_path = candidate
                    break
            temporary.unlink()
            return final_path
        except Exception:
            temporary.unlink(missing_ok=True)
            raise

    finally:
        compressed_spool.close()
