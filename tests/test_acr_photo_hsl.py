"""Synthetic packet and read-only I/O tests; no RAW or installed Look inputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace
from xml.sax.saxutils import escape, quoteattr

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import acr_photo_hsl as hsl


UUID = "0123456789abcdef0123456789ABCDEF"
COLORS = ("Red", "Orange", "Yellow", "Green", "Aqua", "Blue", "Purple", "Magenta")
KEYS = tuple(
    f"{kind}Adjustment{color}"
    for kind in ("Hue", "Saturation", "Luminance")
    for color in COLORS
)
SENTINEL = -999999
MAX_PACKET_BYTES = 4 * 1024 * 1024
NS = (
    'xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#" '
    'xmlns:crs="http://ns.adobe.com/camera-raw-settings/1.0/" '
    'xmlns:other="urn:other"'
)


def _attrs(values: dict[str, str]) -> str:
    return " ".join(f"crs:{key}={quoteattr(value)}" for key, value in values.items())


def _elements(values: dict[str, str]) -> str:
    return "".join(f"<crs:{key}>{escape(value)}</crs:{key}>" for key, value in values.items())


def _packet(attributes: str = "", children: str = "", about: str = 'rdf:about=""') -> bytes:
    return (
        f'<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        f'<rdf:RDF {NS}><rdf:Description {about} {attributes}>'
        f"{children}</rdf:Description></rdf:RDF></x:xmpmeta>"
    ).encode("utf-8")


def _look(
    parameters: dict[str, str] | None = None,
    identity: dict[str, str] | None = None,
    *,
    wrapped: bool = False,
    element_scalars: bool = False,
    extra_look: str = "",
    extra_parameters: str = "",
) -> str:
    identity = {"Name": "Selected Look", "UUID": UUID, **(identity or {})}
    parameters = parameters or {}
    if element_scalars:
        look_attributes, look_children = "", _elements(identity)
        parameter_attributes, parameter_children = "", _elements(parameters)
    else:
        look_attributes, look_children = _attrs(identity), ""
        parameter_attributes, parameter_children = _attrs(parameters), ""
    if wrapped:
        param = (
            f"<crs:Parameters><rdf:Description {parameter_attributes}>"
            f"{parameter_children}{extra_parameters}</rdf:Description></crs:Parameters>"
        )
        return (
            f"<crs:Look><rdf:Description {look_attributes}>{look_children}"
            f"{param}{extra_look}</rdf:Description></crs:Look>"
        )
    param = (
        f'<crs:Parameters rdf:parseType="Resource" {parameter_attributes}>'
        f"{parameter_children}{extra_parameters}</crs:Parameters>"
    )
    return (
        f'<crs:Look rdf:parseType="Resource" {look_attributes}>'
        f"{look_children}{param}{extra_look}</crs:Look>"
    )


def _rejected(packet: bytes, code: str | None = None) -> hsl.PhotoHslError:
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet(packet)
    error = caught.value
    assert isinstance(error, ValueError)
    assert re.fullmatch(r"[a-z][a-z0-9_]*", error.code)
    assert str(error) == f"Cannot inspect photo HSL metadata ({error.code})."
    if code is not None:
        assert error.code == code
    return error


@pytest.mark.parametrize("packet_source", ["embedded", "sidecar"])
def test_empty_photo_projection_has_explicit_unbound_defaults(packet_source: str) -> None:
    packet = _packet()
    result = hsl.inspect_hsl_packet(packet, packet_source)
    assert result == {
        "version": "acr-photo-hsl-v1",
        "packet_source": packet_source,
        "packet_sha256": hashlib.sha256(packet).hexdigest(),
        "current24": [0] * 24,
        "current_camera_profile": None,
        "current_convert_to_gray": False,
        "current_enable_color_adjustments": True,
        "look": None,
        "consumer_bound": False,
        "limitation": "consumer_state_unbound",
    }
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("element_scalars", [False, True])
def test_top_and_nested_hsl_and_profile_scopes_are_independent(
    wrapped: bool, element_scalars: bool,
) -> None:
    current = {key: str(index - 12) for index, key in enumerate(KEYS)}
    defaults = {key: str(40 - index) for index, key in enumerate(KEYS)}
    children = _look(
        {**defaults, "CameraProfile": " Group: Camera BW suffix "},
        {"Name": "Not the camera profile", "Amount": " 1.234tail"},
        wrapped=wrapped,
        element_scalars=element_scalars,
    )
    current["CameraProfile"] = " Top-level color profile "
    if element_scalars:
        packet = _packet(children=_elements(current) + children)
    else:
        packet = _packet(_attrs(current), children)
    result = hsl.inspect_hsl_packet(packet)
    assert result["current24"] == list(range(-12, 12))
    assert result["current_camera_profile"] == " Top-level color profile "
    assert result["look"] == {
        "name": "Not the camera profile",
        "uuid": UUID,
        "amount": 1.234,
        "defaults24": list(range(40, 16, -1)),
        "camera_profile": " Group: Camera BW suffix ",
        "convert_to_gray": False,
        "enable_color_adjustments": True,
    }
    assert result["consumer_bound"] is False


def test_wrong_layers_and_namespaces_never_supply_controls_or_profile() -> None:
    wrong_scope = (
        '<crs:Parameters crs:HueAdjustmentRed="99" crs:CameraProfile="Wrong top parameters"/>'
        '<other:HueAdjustmentOrange>98</other:HueAdjustmentOrange>'
        '<other:Look crs:Name="Wrong namespace"/>'
        '<other:Container crs:HueAdjustmentYellow="97">'
        '<crs:HueAdjustmentGreen>96</crs:HueAdjustmentGreen></other:Container>'
    )
    packet = _packet(
        'crs:CameraProfile="Top profile"',
        wrong_scope + _look(
            extra_look='<crs:HueAdjustmentRed>95</crs:HueAdjustmentRed>'
            '<crs:CameraProfile>Wrong look identity scope</crs:CameraProfile>',
            extra_parameters='<other:HueAdjustmentRed>94</other:HueAdjustmentRed>'
            '<other:Container><crs:HueAdjustmentBlue>93</crs:HueAdjustmentBlue></other:Container>',
        ),
    )
    result = hsl.inspect_hsl_packet(packet)
    assert result["current24"] == [0] * 24
    assert result["look"]["defaults24"] == [SENTINEL] * 24
    assert result["look"]["camera_profile"] is None


def test_current_is_not_replaced_by_nested_defaults() -> None:
    result = hsl.inspect_hsl_packet(_packet(
        'crs:HueAdjustmentRed="7"',
        _look({"HueAdjustmentRed": "-9", "SaturationAdjustmentOrange": "0"}),
    ))
    assert result["current24"] == [7] + [0] * 23
    expected = [SENTINEL] * 24
    expected[0], expected[9] = -9, 0
    assert result["look"]["defaults24"] == expected


@pytest.mark.parametrize(
    ("enabled", "gray", "expected"),
    [(True, False, None), (False, False, 0), (True, True, SENTINEL), (False, True, SENTINEL)],
)
def test_nested_missing_zero_category_reset_and_gray_order(enabled, gray, expected) -> None:
    packet = _packet(children=_look({
        "SaturationAdjustmentOrange": "0",
        "LuminanceAdjustmentOrange": "-7",
        "EnableColorAdjustments": str(enabled),
        "ConvertToGrayscale": str(gray),
    }))
    look = hsl.inspect_hsl_packet(packet)["look"]
    expected_values = [SENTINEL] * 24
    expected_values[9], expected_values[17] = 0, -7
    if expected is not None:
        expected_values = [expected] * 24
    assert look["defaults24"] == expected_values
    assert look["enable_color_adjustments"] is enabled
    assert look["convert_to_gray"] is gray


@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("gray", [True, False])
def test_current_category_reset_but_gray_preserves_observed_values(enabled: bool, gray: bool) -> None:
    packet = _packet(_attrs({
        "HueAdjustmentRed": "31",
        "ConvertToGrayscale": str(gray),
        "EnableColorAdjustments": str(enabled),
        "CameraProfile": "Group: Camera ACROS + R",
    }))
    result = hsl.inspect_hsl_packet(packet)
    assert result["current24"] == ([31] + [0] * 23 if enabled else [0] * 24)
    assert result["current_convert_to_gray"] is gray
    assert result["current_enable_color_adjustments"] is enabled
    assert result["current_camera_profile"] == "Group: Camera ACROS + R"


@pytest.mark.parametrize(
    ("text", "expected"),
    [("  -14tail", -14), ("17.9", 17), ("0x10", 0), ("1e2", 1),
     ("+100suffix", 100), ("-100", -100), ("-0", 0), ("000007", 7),
     ("\t+42suffix", 42), ("0" * 5000 + "9", 9)],
)
def test_hsl_ascii_decimal_prefix_contract(text: str, expected: int) -> None:
    result = hsl.inspect_hsl_packet(_packet(
        children=_elements({"HueAdjustmentRed": text}) + _look({"HueAdjustmentRed": text}),
    ))
    assert result["current24"][0] == result["look"]["defaults24"][0] == expected


@pytest.mark.parametrize("text", [
    "", "junk", "+", "-", "101", "-101", "2147483647", "-2147483648",
    "2147483648", "-2147483649", "9" * 5000, "１２", "١٢", "\u00a012", "--2",
])
@pytest.mark.parametrize("nested", [False, True])
def test_present_bad_hsl_is_rejected_not_treated_as_missing_or_clamped(text: str, nested: bool) -> None:
    values = {"HueAdjustmentRed": text}
    packet = _packet(children=_look(values)) if nested else _packet(_attrs(values))
    _rejected(packet, "invalid_hsl")


@pytest.mark.parametrize("key", ["ConvertToGrayscale", "EnableColorAdjustments"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("text", ["1", "0", " true", "False ", "TRUEsuffix", "", "Ｔｒｕｅ"])
def test_bool_rejects_non_exact_ascii_forms(key: str, nested: bool, text: str) -> None:
    values = {key: text}
    packet = _packet(children=_look(values)) if nested else _packet(_attrs(values))
    _rejected(packet, "invalid_boolean")


def test_bool_case_is_ascii_insensitive_and_profiles_preserve_empty_strings() -> None:
    result = hsl.inspect_hsl_packet(_packet(
        _attrs({"ConvertToGrayscale": "tRuE", "EnableColorAdjustments": "fAlSe", "CameraProfile": ""}),
        _look({"ConvertToGrayscale": "FaLsE", "EnableColorAdjustments": "TrUe", "CameraProfile": ""}),
    ))
    assert result["current_convert_to_gray"] is True
    assert result["current_enable_color_adjustments"] is False
    assert result["current_camera_profile"] == result["look"]["camera_profile"] == ""


@pytest.mark.parametrize(
    ("text", "expected"),
    [("0", 0.0), ("-0", -0.0), ("1.234tail", 1.234), ("  +.75e+1suffix", 7.5),
     ("1.e-2suffix", 0.01), ("-5.25", -5.25), ("3", 3.0),
     ("1e-2extra", 0.01),
     ("1.7976931348623157e308", sys.float_info.max)],
)
def test_amount_decimal_prefix_is_finite_unquantized_and_unclamped(text: str, expected: float) -> None:
    look = hsl.inspect_hsl_packet(_packet(children=_look(identity={"Amount": text})))["look"]
    assert look["amount"] == expected
    assert isinstance(look["amount"], float)
    json.dumps(look, allow_nan=False)


@pytest.mark.parametrize("text", [
    "", "junk", "+", "-", ".", "nan", "NaNtail", "+inf", "-Infinity", "1e9999",
    "-1e9999", "0x1p0", "  -0X1.0p1", "1e", "1e+tail", "１２", "\u00a01",
])
def test_amount_rejects_nonfinite_hex_or_invalid_decimal_subset(text: str) -> None:
    _rejected(_packet(children=_look(identity={"Amount": text})), "invalid_amount")


def test_supports_amount_and_wrong_scope_amount_do_not_supply_amount() -> None:
    result = hsl.inspect_hsl_packet(_packet(
        'crs:Amount="0.5" crs:SupportsAmount="True"',
        _look({"Amount": "0.25"}, {"SupportsAmount": "False"}),
    ))
    assert result["look"]["amount"] == 1.0


@pytest.mark.parametrize("identity", [
    {"Name": ""}, {"Name": "   "}, {"UUID": ""}, {"UUID": "0" * 32},
    {"UUID": "f" * 31}, {"UUID": "g" * 32}, {"UUID": "f" * 33}, {"UUID": " " + "f" * 32},
])
def test_look_requires_nonempty_name_and_explicit_nonzero_32hex_uuid(identity) -> None:
    _rejected(_packet(children=_look(identity=identity)), "invalid_look")


@pytest.mark.parametrize("identity", [{"Name": "Selected"}, {"UUID": UUID}])
def test_missing_identity_has_no_fallback(identity) -> None:
    _rejected(_packet(children=(
        f'<crs:Look {_attrs(identity)}><crs:Parameters/></crs:Look>'
    )), "invalid_look")


def test_look_requires_parameters() -> None:
    _rejected(_packet(children=f'<crs:Look {_attrs({"Name": "A", "UUID": UUID})}/>'), "invalid_look")


@pytest.mark.parametrize("stub", ["True", "TRUE", "tRuE"])
def test_stubbed_look_never_resolves_resources(stub) -> None:
    _rejected(_packet(children=_look(identity={"Stubbed": stub})), "unsupported_look")


@pytest.mark.parametrize("stub", ["1", " false", "False ", ""])
def test_stubbed_bool_is_also_strict(stub) -> None:
    _rejected(_packet(children=_look(identity={"Stubbed": stub})), "invalid_boolean")


def test_explicit_false_stub_is_allowed() -> None:
    assert hsl.inspect_hsl_packet(_packet(children=_look(identity={"Stubbed": "fAlSe"})))["look"]


@pytest.mark.parametrize("uuid", [
    "226E6263ED94405B973B41F5A6E67BE9", "5763124DCEAF42459912C1BD2AAC8FE5",
    "226e6263ed94405b973b41f5a6e67be9", "5763124dceaf42459912c1bd2aac8fe5",
])
def test_special_uuid_runtime_gates_are_not_assumed(uuid: str) -> None:
    _rejected(_packet(children=_look(identity={"UUID": uuid})), "unsupported_look")


@pytest.mark.parametrize("preset", ["Look", "Normal", "", "Photo"])
@pytest.mark.parametrize("element", [False, True])
def test_any_top_level_preset_type_is_not_a_photo(preset: str, element: bool) -> None:
    values = {"PresetType": preset}
    packet = _packet(children=_elements(values)) if element else _packet(_attrs(values))
    _rejected(packet, "not_photo_packet")


@pytest.mark.parametrize("children", [
    _look() + _look(),
    '<crs:HueAdjustmentRed>1</crs:HueAdjustmentRed>' * 2,
    '<crs:CameraProfile>A</crs:CameraProfile>' * 2,
    '<crs:Ignored>same</crs:Ignored>' * 2,
])
def test_duplicate_direct_properties_are_rejected_even_if_equal(children: str) -> None:
    _rejected(_packet(children=children), "duplicate_field")


@pytest.mark.parametrize("key,value", [
    ("HueAdjustmentRed", "1"), ("CameraProfile", "A"),
    ("ConvertToGrayscale", "False"), ("EnableColorAdjustments", "True"), ("Ignored", "same"),
])
@pytest.mark.parametrize("nested", [False, True])
def test_attribute_plus_element_is_ambiguous_even_if_equal(key, value, nested) -> None:
    if nested:
        packet = _packet(children=_look({key: value}, extra_parameters=_elements({key: value})))
    else:
        packet = _packet(_attrs({key: value}), _elements({key: value}))
    _rejected(packet, "duplicate_field")


@pytest.mark.parametrize("extra", [
    '<crs:Name>Selected Look</crs:Name>', '<crs:UUID>' + UUID + '</crs:UUID>',
    '<crs:Parameters/>',
])
def test_duplicate_look_identity_and_parameters(extra: str) -> None:
    _rejected(_packet(children=_look(extra_look=extra)), "duplicate_field")


@pytest.mark.parametrize("key", [
    "HueAdjustmentRed", "CameraProfile", "ConvertToGrayscale", "EnableColorAdjustments",
])
@pytest.mark.parametrize("scalar", [
    '<rdf:Alt><rdf:li>0</rdf:li></rdf:Alt>', '<rdf:Description/>', '<!-- hidden -->0',
    '<?hidden value?>0',
])
def test_complex_top_scalars_are_rejected(key: str, scalar: str) -> None:
    _rejected(_packet(children=f"<crs:{key}>{scalar}</crs:{key}>"), "complex_scalar")


@pytest.mark.parametrize("key", [
    "HueAdjustmentRed", "CameraProfile", "ConvertToGrayscale", "EnableColorAdjustments",
])
def test_complex_parameter_scalars_are_not_flattened(key: str) -> None:
    _rejected(_packet(children=_look(extra_parameters=(
        f"<crs:{key}><rdf:Alt><rdf:li>0</rdf:li></rdf:Alt></crs:{key}>"
    ))), "complex_scalar")


@pytest.mark.parametrize("key", ["Name", "UUID", "Amount", "Stubbed"])
def test_complex_look_identity_scalar_is_rejected(key: str) -> None:
    packet = _packet(children=_look(element_scalars=True))
    packet = packet.replace(
        f"<crs:{key}>".encode(), f'<crs:{key} other:attribute="not-plain">'.encode(),
    ) if key in ("Name", "UUID") else _packet(children=_look(extra_look=(
        f"<crs:{key}><rdf:Description/></crs:{key}>"
    )))
    _rejected(packet, "complex_scalar")


@pytest.mark.parametrize("children", [
    '<crs:Look>not a struct</crs:Look>',
    '<crs:Look crs:Name="A"><rdf:Description/></crs:Look>',
    '<crs:Look><rdf:Description/><rdf:Description/></crs:Look>',
    '<crs:Look><rdf:Description/><crs:Name>A</crs:Name></crs:Look>',
    '<crs:Look rdf:parseType="Resource"><rdf:Description/></crs:Look>',
    '<crs:Look rdf:resource="remote"/>',
    '<crs:Look><rdf:Description rdf:about="remote"/></crs:Look>',
])
def test_struct_wrappers_do_not_mix_outer_and_inner_scopes(children: str) -> None:
    _rejected(_packet(children=children), "invalid_structure")


@pytest.mark.parametrize("parameters", [
    '<crs:Parameters>not a struct</crs:Parameters>',
    '<crs:Parameters crs:HueAdjustmentRed="1"><rdf:Description/></crs:Parameters>',
    '<crs:Parameters><rdf:Description/><rdf:Description/></crs:Parameters>',
    '<crs:Parameters><rdf:Description/><crs:HueAdjustmentRed>1</crs:HueAdjustmentRed></crs:Parameters>',
])
def test_parameter_struct_scope_must_also_be_unambiguous(parameters: str) -> None:
    packet = _packet(children=(f'<crs:Look {_attrs({"Name": "A", "UUID": UUID})}>{parameters}</crs:Look>'))
    _rejected(packet, "invalid_structure")


@pytest.mark.parametrize("packet,code", [
    (b"", "packet_missing"), (b"not xml", "malformed_xml"),
    (b"<x>", "malformed_xml"), (b"<x>\xff</x>", "invalid_utf8"),
    ('<x/>'.encode('utf-16'), "invalid_utf8"),
    (b'<?xml version="1.0" encoding="ISO-8859-1"?><x/>', "invalid_utf8"),
    (b'<?xml version="1.0" encoding="UTF-16"?><x/>', "invalid_utf8"),
    (b'<!DOCTYPE x [<!ENTITY a "text">]><x/>', "unsafe_xml"),
    (b'<!doctype x><x/>', "unsafe_xml"), (b'<!EnTiTy a "text"><x/>', "unsafe_xml"),
    (b'<!-- <!DOCTYPE hidden -->' + _packet(), "unsafe_xml"),
    (_packet('crs:HueAdjustmentRed="1" crs:HueAdjustmentRed="1"'), "malformed_xml"),
    (_packet(children='<crs:HueAdjustmentRed other:attribute="x">1</crs:HueAdjustmentRed>'), "complex_scalar"),
    (_packet(about='rdf:about="nonempty"'), "invalid_structure"),
    (_packet(about='rdf:nodeID="unbound-subject"'), "invalid_structure"),
    (b"<x/>", "invalid_structure"),
    (f'<rdf:RDF {NS}/>'.encode(), "invalid_structure"),
    (f'<rdf:RDF {NS}><rdf:Description/><rdf:Description/></rdf:RDF>'.encode(), "invalid_structure"),
    (f'<rdf:RDF {NS}>mixed text<rdf:Description/></rdf:RDF>'.encode(), "invalid_structure"),
    (f'<rdf:RDF {NS}><rdf:Description/><rdf:Bag/></rdf:RDF>'.encode(), "invalid_structure"),
    (f'<root {NS}><other:Container><rdf:RDF><rdf:Description crs:HueAdjustmentRed="77"/></rdf:RDF></other:Container></root>'.encode(), "invalid_structure"),
    (f'<root {NS}><rdf:RDF><rdf:Description/></rdf:RDF><rdf:RDF/></root>'.encode(), "invalid_structure"),
    (_packet(children='<rdf:RDF/>'), "invalid_structure"),
])
def test_xml_encoding_security_and_rdf_shape_fail_closed(packet: bytes, code: str) -> None:
    _rejected(packet, code)


def test_packet_limit_inclusive_and_hash_uses_original_bytes() -> None:
    base = b'<?xml version="1.0" encoding="UTF-8"?>' + _packet()
    packet = base + b" " * (MAX_PACKET_BYTES - len(base))
    result = hsl.inspect_hsl_packet(packet)
    assert result["packet_sha256"] == hashlib.sha256(packet).hexdigest()
    _rejected(packet + b" ", "packet_too_large")


def test_utf8_bom_xpacket_and_missing_empty_about_are_accepted() -> None:
    packet = b'\xef\xbb\xbf<?xpacket begin=""?>' + _packet(about="") + b'<?xpacket end="w"?>'
    assert hsl.inspect_hsl_packet(packet)["current24"] == [0] * 24


def test_exact_dji_subject_is_accepted_only_as_unbound_embedded_observation() -> None:
    packet = _packet(
        'crs:HueAdjustmentRed="7"',
        _look({"HueAdjustmentRed": "-9"}, wrapped=True),
        about='rdf:about="DJI Meta Data"',
    )
    result = hsl.inspect_hsl_packet(packet, "embedded")
    assert result["current24"] == [7] + [0] * 23
    assert result["look"]["defaults24"] == [-9] + [SENTINEL] * 23
    assert result["packet_sha256"] == hashlib.sha256(packet).hexdigest()
    assert result["packet_source"] == "embedded"
    assert result["consumer_bound"] is False
    assert result["limitation"] == "consumer_state_unbound"
    assert "DJI Meta Data" not in json.dumps(result)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet(packet, "sidecar")
    assert caught.value.code == "invalid_structure"


@pytest.mark.parametrize("source", ["embedded", "sidecar"])
@pytest.mark.parametrize("about", ["", 'rdf:about=""'])
def test_missing_and_empty_subjects_remain_accepted_for_both_sources(source, about) -> None:
    assert hsl.inspect_hsl_packet(_packet(about=about), source)["look"] is None


@pytest.mark.parametrize("source", ["embedded", "sidecar"])
@pytest.mark.parametrize("subject", [
    "https://example.invalid/photo", "file:///external/photo.dng", "urn:photo:selected",
    "dji Meta Data", "DJI META DATA", "DJI Meta data", " DJI Meta Data",
    "DJI Meta Data ", "DJI Meta Data/", "DJI Meta Data suffix", "DJI  Meta Data",
])
def test_subject_exception_does_not_accept_uris_case_or_spacing_variants(source, subject) -> None:
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet(_packet(about=f"rdf:about={quoteattr(subject)}"), source)
    assert caught.value.code == "invalid_structure"


@pytest.mark.parametrize("source", ["embedded", "sidecar"])
@pytest.mark.parametrize("scope", ["Look", "Parameters"])
def test_dji_subject_exception_never_applies_to_nested_structs(source, scope) -> None:
    children = _look(wrapped=True).replace(
        f"<crs:{scope}><rdf:Description",
        f'<crs:{scope}><rdf:Description rdf:about="DJI Meta Data"',
        1,
    )
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet(_packet(children=children), source)
    assert caught.value.code == "invalid_structure"


def test_dji_subject_does_not_relax_single_description_requirement() -> None:
    packet = (
        f'<rdf:RDF {NS}><rdf:Description rdf:about="DJI Meta Data"/>'
        '<rdf:Description rdf:about=""/></rdf:RDF>'
    ).encode()
    _rejected(packet, "invalid_structure")


@pytest.mark.parametrize("source", ["", "auto", "EMBEDDED", "sidecar ", None, 1])
def test_invalid_packet_source_is_rejected_before_io(source, tmp_path: Path) -> None:
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(tmp_path / "missing.nef", source)
    assert caught.value.code == "invalid_packet_source"
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet(_packet(), source)
    assert caught.value.code == "invalid_packet_source"


def test_packet_must_be_bytes() -> None:
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_hsl_packet("<x/>")
    assert caught.value.code == "invalid_packet"


def test_nested_look_tables_and_local_adjustments_do_not_become_complete_render_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Path, "open", lambda *_args, **_kwargs: pytest.fail("parser attempted file lookup"))
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: pytest.fail("parser invoked a process"))
    result = hsl.inspect_hsl_packet(_packet(children=_look(
        {"HueAdjustmentRed": "4"},
        extra_parameters='<crs:Look crs:UUID="unresolved"/>'
        '<crs:LookTable>unresolved table</crs:LookTable>'
        '<crs:PaintBasedCorrections><rdf:Seq><rdf:li crs:HueAdjustmentRed="77"/></rdf:Seq></crs:PaintBasedCorrections>',
    )))
    assert result["look"]["defaults24"][0] == 4
    assert result["consumer_bound"] is False
    assert result["limitation"] == "consumer_state_unbound"


def _files(tmp_path: Path, packet: bytes | None = None) -> tuple[Path, Path]:
    photo = tmp_path / "Selected.NEF"
    sidecar = tmp_path / "sElEcTeD.XmP"
    photo.write_bytes(b"synthetic photo, never decoded")
    if packet is not None:
        sidecar.write_bytes(packet)
    return photo, sidecar


def _install_exiftool(monkeypatch: pytest.MonkeyPatch, executable: str | None = "/synthetic/exiftool") -> None:
    monkeypatch.setitem(sys.modules, "image_io", SimpleNamespace(_find_exiftool=lambda: executable))


def _stat_identity(path: Path) -> tuple:
    value = path.stat()
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def test_sidecar_is_explicit_case_insensitive_read_only_and_private(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = _packet(
        'other:Personal="private-personal-marker"',
        '<other:GPS>private-location-marker</other:GPS>' + _look(),
    )
    photo, sidecar = _files(tmp_path, packet)
    before = {path: _stat_identity(path) for path in (photo, sidecar)}
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("sidecar attempted embedded fallback"))
    result = hsl.inspect_photo_hsl(photo, "sidecar")
    assert result == hsl.inspect_hsl_packet(packet, "sidecar")
    assert {path: _stat_identity(path) for path in before} == before
    assert photo.read_bytes() == b"synthetic photo, never decoded"
    assert sidecar.read_bytes() == packet
    serialized = json.dumps(result, allow_nan=False)
    for forbidden in (str(tmp_path), str(photo), str(sidecar), "private-personal-marker", "private-location-marker", "<rdf:"):
        assert forbidden not in serialized
    assert "packet" not in result


def test_embedded_uses_exact_exiftool_call_without_reading_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo, _ = _files(tmp_path, b"invalid sidecar must not be used")
    _install_exiftool(monkeypatch)
    packet = _packet('crs:HueAdjustmentBlue="19"')
    before = _stat_identity(photo)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(stdout=packet)

    monkeypatch.setattr(subprocess, "run", run)
    result = hsl.inspect_photo_hsl(photo, "embedded")
    assert calls == [([
        "/synthetic/exiftool", "-b", "-XMP", str(photo),
    ], {"capture_output": True, "check": True, "timeout": 8})]
    assert result == hsl.inspect_hsl_packet(packet)
    assert _stat_identity(photo) == before


@pytest.mark.parametrize("scenario,code", [
    ("missing", "sidecar_missing"), ("ambiguous", "sidecar_ambiguous"), ("directory", "sidecar_unavailable"),
])
def test_sidecar_selection_rejects_missing_ambiguous_or_nonfile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, scenario: str, code: str,
) -> None:
    photo, sidecar = _files(tmp_path)
    if scenario == "ambiguous":
        sidecar.write_bytes(_packet())
        # Model a case-sensitive directory even on case-insensitive macOS.
        original_iterdir = Path.iterdir

        def iterdir(path):
            entries = list(original_iterdir(path))
            return iter(entries + [tmp_path / "Selected.xmp"]) if path == tmp_path else iter(entries)

        monkeypatch.setattr(Path, "iterdir", iterdir)
    elif scenario == "directory":
        sidecar.mkdir()
    # A packet in another folder or with another stem cannot be a fallback.
    (tmp_path / "other.xmp").write_bytes(_packet())
    sibling = tmp_path / "other-folder"
    sibling.mkdir()
    (sibling / "Selected.xmp").write_bytes(_packet())
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: pytest.fail("unexpected fallback"))
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == code


def test_oversized_sidecar_is_rejected_before_open(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo, sidecar = _files(tmp_path)
    with sidecar.open("wb") as stream:
        stream.truncate(MAX_PACKET_BYTES + 1)
    monkeypatch.setattr(Path, "open", lambda *_a, **_k: pytest.fail("oversized sidecar opened"))
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == "packet_too_large"


@pytest.mark.parametrize("source", ["embedded", "sidecar"])
def test_missing_or_nonfile_source_is_a_safe_error(tmp_path: Path, source: str) -> None:
    for path in (tmp_path / "private-missing.nef", tmp_path):
        with pytest.raises(hsl.PhotoHslError) as caught:
            hsl.inspect_photo_hsl(path, source)
        assert caught.value.code == "source_unavailable"
        assert str(path) not in str(caught.value)


@pytest.mark.parametrize("failure", [
    FileNotFoundError("private path"),
    subprocess.CalledProcessError(2, ["private command"], stderr=b"private packet"),
    subprocess.TimeoutExpired(["private command"], 8, stderr=b"private packet"),
])
def test_embedded_extraction_failure_is_not_no_look_or_sidecar_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure,
) -> None:
    photo, _ = _files(tmp_path, _packet())
    _install_exiftool(monkeypatch)

    def run(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "embedded")
    assert caught.value.code == "xmp_extraction_failed"
    assert "private" not in str(caught.value)


def test_missing_exiftool_is_not_no_look(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo, _ = _files(tmp_path)
    _install_exiftool(monkeypatch, None)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "embedded")
    assert caught.value.code == "exiftool_unavailable"


@pytest.mark.parametrize("packet,code", [(b"", "packet_missing"), (b"bad", "malformed_xml"), (b"x" * (MAX_PACKET_BYTES + 1), "packet_too_large")])
def test_embedded_packet_errors_remain_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, packet: bytes, code: str,
) -> None:
    photo, _ = _files(tmp_path, _packet())
    _install_exiftool(monkeypatch)
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=packet))
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "embedded")
    assert caught.value.code == code


@pytest.mark.parametrize("change", ["write", "replace", "delete"])
def test_embedded_rejects_source_changes_during_extraction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str,
) -> None:
    photo, _ = _files(tmp_path)
    _install_exiftool(monkeypatch)

    def run(*_args, **_kwargs):
        if change == "write":
            photo.write_bytes(b"changed source")
        elif change == "replace":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(photo.read_bytes())
            replacement.replace(photo)
        else:
            photo.unlink()
        return SimpleNamespace(stdout=_packet())

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "embedded")
    assert caught.value.code == "source_changed"


@pytest.mark.parametrize("changed", ["photo", "sidecar", "replacement", "new_candidate"])
def test_sidecar_checks_both_source_and_packet_after_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changed: str,
) -> None:
    photo, sidecar = _files(tmp_path, _packet())
    original = hsl._read_sidecar

    def read(path, before):
        packet = original(path, before)
        if changed == "photo":
            photo.write_bytes(b"source changed")
        elif changed == "sidecar":
            sidecar.write_bytes(b"sidecar changed")
        elif changed == "replacement":
            replacement = tmp_path / "replacement"
            replacement.write_bytes(packet)
            replacement.replace(sidecar)
        else:
            # Case-only duplicates cannot be created on every test filesystem.
            original_iterdir = Path.iterdir

            def iterdir(path):
                entries = list(original_iterdir(path))
                return iter(entries + [tmp_path / "Selected.xmp"]) if path == tmp_path else iter(entries)

            monkeypatch.setattr(Path, "iterdir", iterdir)
        return packet

    monkeypatch.setattr(hsl, "_read_sidecar", read)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == {
        "photo": "source_changed", "sidecar": "sidecar_changed",
        "replacement": "sidecar_changed", "new_candidate": "sidecar_ambiguous",
    }[changed]


def test_sidecar_open_failure_is_safe(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    photo, _ = _files(tmp_path, _packet())

    def fail(*_args, **_kwargs):
        raise PermissionError("private file information")

    monkeypatch.setattr(Path, "open", fail)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == "sidecar_read_failed"
    assert "private" not in str(caught.value)


def test_sidecar_results_are_not_cached_across_explicit_reads(tmp_path: Path) -> None:
    photo, sidecar = _files(tmp_path, _packet('crs:HueAdjustmentRed="1"'))
    first = hsl.inspect_photo_hsl(photo, "sidecar")
    sidecar.write_bytes(_packet('crs:HueAdjustmentRed="2"'))
    second = hsl.inspect_photo_hsl(photo, "sidecar")
    assert first["current24"][0] == 1
    assert second["current24"][0] == 2
    assert first["packet_sha256"] != second["packet_sha256"]


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"])
def test_every_required_stat_field_participates_in_change_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    photo, _ = _files(tmp_path)
    _install_exiftool(monkeypatch)
    real_stat = Path.stat
    counts = 0

    def stat(path, *args, **kwargs):
        nonlocal counts
        result = real_stat(path, *args, **kwargs)
        if path == photo:
            counts += 1
            if counts >= 2:
                values = {key: getattr(result, key) for key in (
                    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode",
                )}
                values[field] += 1
                return SimpleNamespace(**values)
        return result

    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(subprocess, "run", lambda *_a, **_k: SimpleNamespace(stdout=_packet()))
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "embedded")
    assert caught.value.code == "source_changed"


@pytest.mark.parametrize("field", ["st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns"])
def test_every_sidecar_stat_field_participates_in_post_read_detection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str,
) -> None:
    photo, sidecar = _files(tmp_path, _packet())
    real_stat = Path.stat
    counts = 0

    def stat(path, *args, **kwargs):
        nonlocal counts
        result = real_stat(path, *args, **kwargs)
        if path == sidecar:
            counts += 1
            if counts >= 2:
                values = {key: getattr(result, key) for key in (
                    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_mode",
                )}
                values[field] += 1
                return SimpleNamespace(**values)
        return result

    monkeypatch.setattr(Path, "stat", stat)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == "sidecar_changed"


def test_sidecar_descriptor_detects_replacement_before_reading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    photo, sidecar = _files(tmp_path, _packet())
    original_open = Path.open

    def open(path, *args, **kwargs):
        if path == sidecar:
            replacement = tmp_path / "replacement"
            replacement.write_bytes(_packet('crs:HueAdjustmentRed="1"'))
            replacement.replace(sidecar)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", open)
    with pytest.raises(hsl.PhotoHslError) as caught:
        hsl.inspect_photo_hsl(photo, "sidecar")
    assert caught.value.code == "sidecar_changed"
