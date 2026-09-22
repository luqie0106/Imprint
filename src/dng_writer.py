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
import struct
import tempfile
import threading
from typing import Any, Iterable
from xml.sax.saxutils import escape

import numpy as np
from PIL import Image


BYTE, ASCII, SHORT, LONG, RATIONAL, UNDEFINED, SLONG, SRATIONAL = 1, 2, 3, 4, 5, 7, 9, 10
TYPE_SIZES = {BYTE: 1, ASCII: 1, SHORT: 2, LONG: 4, RATIONAL: 8, UNDEFINED: 1, SLONG: 4, SRATIONAL: 8}
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
    if lens_model is None and camera_model is None:
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
        _tag(50708, ASCII, _ascii(f"{make} {model}")),
        _tag(50717, LONG, _longs([(1 << bits_per_sample) - 1] * 3)),
        _tag(50719, LONG, _longs([0, 0])), _tag(50720, LONG, _longs([width, height])),
        # XYZ D50 to linear-sRGB. The pixels are exported in this declared working
        # space, rather than pretending that processed RGB remains camera-native.
        _tag(50721, SRATIONAL, _rationals([(3133856, 1000000), (-1616867, 1000000), (-490615, 1000000), (-978768, 1000000), (1916142, 1000000), (33454, 1000000), (71945, 1000000), (-228991, 1000000), (1405243, 1000000)], True)),
        _tag(50728, RATIONAL, _rationals([(1, 1), (1, 1), (1, 1)])),
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
