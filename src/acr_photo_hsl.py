"""Read a selected photo's narrow HSL metadata projection, never its render state.

The field contract comes from the saved 2026-11-08 Look metadata evidence.
Missing current controls are local projection zeros, not a claim about Adobe's
complete photo loader. Nested defaults retain the recovered sentinel/reset
order. No installed resource lookup, native merge, Prepare, or pixel operation
is performed here; the selected consumer state remains explicitly unbound.
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import xml.etree.ElementTree as ET


_CRS = "{http://ns.adobe.com/camera-raw-settings/1.0/}"
_RDF = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}"
_MAX_PACKET_BYTES = 4 * 1024 * 1024
_SENTINEL = -999999
_COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
_HSL_KEYS = tuple(
    f"{kind}Adjustment{color}"
    for kind in ("Hue", "Saturation", "Luminance")
    for color in _COLORS
)
_SPECIAL_UUIDS = {
    "226E6263ED94405B973B41F5A6E67BE9",
    "5763124DCEAF42459912C1BD2AAC8FE5",
}
_INT_PREFIX = re.compile(r"[ \t\n\r\f\v]*([+-]?)([0-9]+)")
_FLOAT_PREFIX = re.compile(
    r"[ \t\n\r\f\v]*[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)"
    r"(?:[eE][+-]?[0-9]+)?"
)
_HEX_PREFIX = re.compile(r"[ \t\n\r\f\v]*[+-]?0[xX]")
_UNSAFE_XML = re.compile(r"<!\s*(?:DOCTYPE|ENTITY)\b", re.IGNORECASE | re.ASCII)
_XML_ENCODING = re.compile(
    r"\A\ufeff?\s*<\?xml\s+[^?]*\bencoding\s*=\s*(['\"])(.*?)\1",
    re.IGNORECASE | re.ASCII,
)


class PhotoHslError(ValueError):
    """A stable, path- and packet-free error suitable for an API response."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(f"Cannot inspect photo HSL metadata ({code}).")


def _check_packet_source(packet_source: str) -> None:
    if packet_source not in ("embedded", "sidecar"):
        raise PhotoHslError("invalid_packet_source")


def _check_container_text(element: ET.Element) -> None:
    if (element.text or "").strip() or any((child.tail or "").strip() for child in element):
        raise PhotoHslError("invalid_structure")


class _Scope:
    """Index only direct CRS properties, including ignored ones for duplicates."""

    def __init__(self, element: ET.Element):
        _check_container_text(element)
        self.fields: dict[str, str | ET.Element] = {}
        for key, value in element.attrib.items():
            if key.startswith(_CRS):
                self.fields[key[len(_CRS):]] = value
        for child in element:
            if not isinstance(child.tag, str) or not child.tag.startswith(_CRS):
                continue
            key = child.tag[len(_CRS):]
            if key in self.fields:
                raise PhotoHslError("duplicate_field")
            self.fields[key] = child

    def scalar(self, key: str) -> str | None:
        value = self.fields.get(key)
        if isinstance(value, ET.Element):
            if value.attrib or len(value):
                raise PhotoHslError("complex_scalar")
            return value.text or ""
        return value


def _struct(value: str | ET.Element) -> _Scope:
    if not isinstance(value, ET.Element):
        raise PhotoHslError("invalid_structure")
    _check_container_text(value)
    descriptions = [child for child in value if child.tag == _RDF + "Description"]
    if descriptions:
        # A wrapped struct cannot also supply outer attributes/properties.
        if len(descriptions) != 1 or len(value) != 1 or value.attrib:
            raise PhotoHslError("invalid_structure")
        value = descriptions[0]
        allowed_rdf = {_RDF + "about": ""}
    else:
        allowed_rdf = {_RDF + "parseType": "Resource"}
    for key, text in value.attrib.items():
        if key.startswith(_RDF) and (key not in allowed_rdf or text != allowed_rdf[key]):
            raise PhotoHslError("invalid_structure")
    return _Scope(value)


def _hsl_value(text: str) -> int:
    match = _INT_PREFIX.match(text)
    if match is None:
        raise PhotoHslError("invalid_hsl")
    # Avoid Python's large-int digit limit; do not emulate overflowing scanf.
    digits = match.group(2).lstrip("0") or "0"
    if len(digits) > 10:
        raise PhotoHslError("invalid_hsl")
    value = int((match.group(1) or "") + digits)
    if not -(2**31) <= value < 2**31 or not -100 <= value <= 100:
        raise PhotoHslError("invalid_hsl")
    return value


def _hsl24(scope: _Scope, missing: int) -> list[int]:
    return [missing if (text := scope.scalar(key)) is None else _hsl_value(text) for key in _HSL_KEYS]


def _boolean(text: str | None, missing: bool) -> bool:
    if text is None:
        return missing
    if text.isascii() and text.lower() in ("true", "false"):
        return text.lower() == "true"
    raise PhotoHslError("invalid_boolean")


def _amount(text: str | None) -> float:
    if text is None:
        return 1.0
    match = _FLOAT_PREFIX.match(text)
    if match is None or _HEX_PREFIX.match(text):
        raise PhotoHslError("invalid_amount")
    # An incomplete exponent is not in this intentionally narrow decimal subset.
    if text[match.end():match.end() + 1] in ("e", "E") and "e" not in match.group().lower():
        raise PhotoHslError("invalid_amount")
    try:
        value = float(match.group())
    except (ValueError, OverflowError):
        raise PhotoHslError("invalid_amount") from None
    if not math.isfinite(value):
        raise PhotoHslError("invalid_amount")
    return value


def _look(scope: _Scope) -> dict | None:
    value = scope.fields.get("Look")
    if value is None:
        return None
    look = _struct(value)
    name, uuid = look.scalar("Name"), look.scalar("UUID")
    if not name or not name.strip() or uuid is None or re.fullmatch(r"[0-9a-fA-F]{32}", uuid) is None:
        raise PhotoHslError("invalid_look")
    if uuid == "0" * 32:
        raise PhotoHslError("invalid_look")
    if uuid.upper() in _SPECIAL_UUIDS or _boolean(look.scalar("Stubbed"), False):
        raise PhotoHslError("unsupported_look")
    parameters = look.fields.get("Parameters")
    if parameters is None:
        raise PhotoHslError("invalid_look")
    parameters_scope = _struct(parameters)
    defaults = _hsl24(parameters_scope, _SENTINEL)
    gray = _boolean(parameters_scope.scalar("ConvertToGrayscale"), False)
    enabled = _boolean(parameters_scope.scalar("EnableColorAdjustments"), True)
    if not enabled:
        defaults = [0] * 24
    if gray:
        defaults = [_SENTINEL] * 24
    return {
        "name": name,
        "uuid": uuid,
        "amount": _amount(look.scalar("Amount")),
        "defaults24": defaults,
        "camera_profile": parameters_scope.scalar("CameraProfile"),
        "convert_to_gray": gray,
        "enable_color_adjustments": enabled,
    }


def inspect_hsl_packet(packet: bytes, packet_source: str = "embedded") -> dict:
    """Parse one UTF-8 photo packet without I/O or complete-render claims."""
    _check_packet_source(packet_source)
    if not isinstance(packet, bytes):
        raise PhotoHslError("invalid_packet")
    if len(packet) > _MAX_PACKET_BYTES:
        raise PhotoHslError("packet_too_large")
    if not packet:
        raise PhotoHslError("packet_missing")
    try:
        text = packet.decode("utf-8")
    except UnicodeDecodeError:
        raise PhotoHslError("invalid_utf8") from None
    if _UNSAFE_XML.search(text):
        raise PhotoHslError("unsafe_xml")
    encoding = _XML_ENCODING.match(text)
    if encoding and encoding.group(2).lower() not in ("utf-8", "utf8"):
        raise PhotoHslError("invalid_utf8")
    try:
        parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True, insert_pis=True))
        root = ET.fromstring(text, parser=parser)
    except (ET.ParseError, ValueError):
        raise PhotoHslError("malformed_xml") from None
    rdf = [element for element in root.iter() if element.tag == _RDF + "RDF"]
    if len(rdf) != 1:
        raise PhotoHslError("invalid_structure")
    _check_container_text(rdf[0])
    if rdf[0] is not root and all(child is not rdf[0] for child in root):
        raise PhotoHslError("invalid_structure")
    if any(isinstance(child.tag, str) and child.tag != _RDF + "Description" for child in rdf[0]):
        raise PhotoHslError("invalid_structure")
    descriptions = [child for child in rdf[0] if child.tag == _RDF + "Description"]
    if len(descriptions) != 1:
        raise PhotoHslError("invalid_structure")
    description = descriptions[0]
    # This exact embedded subject label permits inspection of the selected
    # packet only; it does not establish its provenance or camera binding.
    allowed_subjects = ("", "DJI Meta Data") if packet_source == "embedded" else ("",)
    if any(key.startswith(_RDF) and (key != _RDF + "about" or value not in allowed_subjects)
           for key, value in description.attrib.items()):
        raise PhotoHslError("invalid_structure")
    scope = _Scope(description)
    if "PresetType" in scope.fields:
        raise PhotoHslError("not_photo_packet")
    current = _hsl24(scope, 0)
    current_gray = _boolean(scope.scalar("ConvertToGrayscale"), False)
    current_enabled = _boolean(scope.scalar("EnableColorAdjustments"), True)
    if not current_enabled:
        current = [0] * 24
    return {
        "version": "acr-photo-hsl-v1",
        "packet_source": packet_source,
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "current24": current,
        "current_camera_profile": scope.scalar("CameraProfile"),
        "current_convert_to_gray": current_gray,
        "current_enable_color_adjustments": current_enabled,
        "look": _look(scope),
        "consumer_bound": False,
        "limitation": "consumer_state_unbound",
    }


def _file_stat(path: Path, error: str) -> os.stat_result:
    try:
        result = path.stat()
    except OSError:
        raise PhotoHslError(error) from None
    if not stat.S_ISREG(result.st_mode):
        raise PhotoHslError(error)
    return result


def _identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _check_unchanged(path: Path, before: os.stat_result, error: str) -> None:
    if _identity(_file_stat(path, error)) != _identity(before):
        raise PhotoHslError(error)


def _sidecar_path(photo: Path) -> Path:
    try:
        candidates = [
            path for path in photo.parent.iterdir()
            if path.stem.casefold() == photo.stem.casefold() and path.suffix.casefold() == ".xmp"
        ]
    except OSError:
        raise PhotoHslError("sidecar_unavailable") from None
    if not candidates:
        raise PhotoHslError("sidecar_missing")
    if len(candidates) != 1:
        raise PhotoHslError("sidecar_ambiguous")
    return candidates[0]


def _read_sidecar(sidecar: Path, before: os.stat_result) -> bytes:
    if before.st_size > _MAX_PACKET_BYTES:
        raise PhotoHslError("packet_too_large")
    try:
        with sidecar.open("rb") as stream:
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                raise PhotoHslError("sidecar_changed")
            packet = stream.read(_MAX_PACKET_BYTES + 1)
            if _identity(os.fstat(stream.fileno())) != _identity(before):
                raise PhotoHslError("sidecar_changed")
    except OSError:
        raise PhotoHslError("sidecar_read_failed") from None
    return packet


def _read_embedded(photo: Path) -> bytes:
    # Keep the pure XML parser independent of image decoding dependencies.
    try:
        import image_io
        executable = image_io._find_exiftool()
    except (ImportError, OSError):
        raise PhotoHslError("exiftool_unavailable") from None
    if not executable:
        raise PhotoHslError("exiftool_unavailable")
    try:
        result = subprocess.run(
            [executable, "-b", "-XMP", str(photo)],
            capture_output=True, check=True, timeout=8,
        )
    except (OSError, subprocess.SubprocessError):
        raise PhotoHslError("xmp_extraction_failed") from None
    return result.stdout


def inspect_photo_hsl(path: Path, packet_source: str) -> dict:
    """Read only the explicitly selected embedded packet or unique sibling XMP.

    Files are checked before/after reading; sidecar reads additionally compare
    the open descriptor. There is no source fallback, cache, or catalog lookup.
    """
    _check_packet_source(packet_source)
    photo = Path(path)
    before = _file_stat(photo, "source_unavailable")
    if packet_source == "embedded":
        packet = _read_embedded(photo)
    else:
        sidecar = _sidecar_path(photo)
        sidecar_before = _file_stat(sidecar, "sidecar_unavailable")
        packet = _read_sidecar(sidecar, sidecar_before)
        _check_unchanged(sidecar, sidecar_before, "sidecar_changed")
        if _sidecar_path(photo) != sidecar:
            raise PhotoHslError("sidecar_changed")
    _check_unchanged(photo, before, "source_changed")
    return inspect_hsl_packet(packet, packet_source)
