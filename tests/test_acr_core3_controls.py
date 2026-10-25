"""Independent instruction-order checks of the core3 object constructor."""
import ctypes as C
import importlib.util
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


adapter = load('core3_controls_adapter', ROOT / 'tests/test_acr_stage3_core3_float_compose.py')
fma = load('core3_controls_float_ops', ROOT / 'tests/acr_support/balance3_finish_oracle.py').fma32
F = np.float32
P = C.POINTER(C.c_float)


def number(bits):
    return np.asarray([bits], dtype=np.uint32).view(np.float32)[0]


def reference(amount, suppress):
    n = F(amount)
    a, b = F(n - F(1)), F(n - F(12))

    def clamp(v):
        v = np.fmin(v, F(1))
        return F(0) if v < F(0) else v

    x, y = clamp(F(a / F(5))), clamp(F(b / F(-6)))
    m = F(0) if suppress else (x if x < y else y)
    q = fma(clamp(F(a / number(0x40accccd))), number(0xbecccccd), F(.5))
    return np.asarray([number(0x41b40000), number(0x3cb60b61), m, q,
                       number(0x3c3b80bc), number(0x3cbb80bc)], dtype=np.float32)


@pytest.fixture(scope='module', params=['source', 'official_object'])
def lib(request, tmp_path_factory):
    files = adapter.SOURCES
    if request.param == 'official_object':
        for obj, src in zip(adapter.OFFICIAL_OBJECTS, adapter.SOURCES):
            assert obj.is_file()
            assert obj.stat().st_mtime_ns >= max(src.stat().st_mtime_ns, src.with_suffix('.hpp').stat().st_mtime_ns), 'rebuild stale renderer objects'
        files = adapter.OFFICIAL_OBJECTS
    path = tmp_path_factory.mktemp('core3_controls_' + request.param) / ('libcontrols' + adapter.SHARED_SUFFIX)
    library = C.CDLL(str(adapter._compile_bridge(path, files)))
    library.acr_test_core3_controls.argtypes = [C.c_double, C.c_int, C.c_uint32, C.c_uint32, P, C.c_size_t]
    library.acr_test_core3_controls.restype = C.c_int
    return library


@pytest.mark.parametrize('suppress', [0, 1])
def test_constructor_controls_match_float_instructions(lib, suppress):
    rng = np.random.default_rng(197713)
    values = [0., -0., -10., 1., 6., 12., 13., float(np.finfo(np.float32).max)]
    values += rng.uniform(-2, 16, 160).tolist()
    values += [float(np.nextafter(F(x), F(d))) for x in (1, 6, 12) for d in (-100, 100)]
    for i, amount in enumerate(values):
        output = np.full(8, -7., dtype=np.float32)
        assert lib.acr_test_core3_controls(amount, suppress, i & 1, (i >> 1) & 1,
            output.ctypes.data_as(P), 8) == 0
        np.testing.assert_array_equal(output[:6].view(np.uint32), reference(amount, suppress).view(np.uint32))
        assert np.all(output[6:] == F(-7))


def test_fixed_constants_and_negative_zero_mask(lib):
    output = np.empty(6, dtype=np.float32)
    assert lib.acr_test_core3_controls(1., 0, 0, 0, output.ctypes.data_as(P), 6) == 0
    assert output.view(np.uint32).tolist() == [0x41b40000, 0x3cb60b61, 0, 0x3f000000, 0x3c3b80bc, 0x3cbb80bc]
    assert lib.acr_test_core3_controls(12., 0, 0, 0, output.ctypes.data_as(P), 6) == 0
    assert output.view(np.uint32)[2] == 0x80000000
    assert lib.acr_test_core3_controls(12., 1, 0, 0, output.ctypes.data_as(P), 6) == 0
    assert output.view(np.uint32)[2] == 0


def test_q_uses_the_recovered_explicit_fma(lib):
    amount = 4.201414794769306
    output = np.empty(6, dtype=np.float32)
    assert lib.acr_test_core3_controls(amount, 0, 0, 0, output.ctypes.data_as(P), 6) == 0
    assert output.view(np.uint32)[3] == 0x3e869559
    ratio = F(F(F(amount) - F(1)) / number(0x40accccd))
    split = F(F(ratio * number(0xbecccccd)) + F(.5))
    assert split.view(np.uint32) == 0x3e869558


def test_invalid_controls_do_not_publish_partial_output(lib):
    for amount, flag, r, c, capacity in [(np.nan, 0, 0, 0, 6), (np.inf, 0, 0, 0, 6),
            (1e300, 0, 0, 0, 6), (1, 0, 2, 0, 6), (1, 0, 0, 2, 6),
            (1, 2, 0, 0, 6), (1, 0, 0, 0, 5)]:
        out = np.full(8, -17.25, dtype=np.float32)
        before = out.copy()
        assert lib.acr_test_core3_controls(amount, flag, r, c, out.ctypes.data_as(P), capacity) != 0
        np.testing.assert_array_equal(out.view(np.uint32), before.view(np.uint32))
