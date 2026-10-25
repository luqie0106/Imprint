"""Full-frame prepared CPU composition: independent pixels and tile seams."""
import ctypes as C
import importlib.util
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT/'native-renderer'
SPEC = importlib.util.spec_from_file_location(
    'stage3_image_independent', ROOT/'tests/acr_support/stage3_image_oracle.py')
oracle = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(oracle)
SOURCE_NAMES = (
    'acr_stage3_prepared_image', 'acr_bayer_border', 'acr_bayer_gain',
    'acr_float_lut', 'acr_u16_lut', 'acr_stage3_curve', 'acr_hotpixel',
    'acr_stage3_core3', 'acr_stage3_core3_prefix', 'acr_stage3_core3_green',
    'acr_stage3_core3_colour', 'acr_stage3_core3_refine',
    'acr_stage3_core3_format', 'acr_stage3_core3_compose')
U16 = C.POINTER(C.c_uint16)
I32 = C.POINTER(C.c_int32)
D = C.POINTER(C.c_double)
S = C.c_size_t
SENTINEL = 0xABCD


@pytest.fixture(scope='module', params=('source', 'official_object', 'formal_api'))
def bridge(request, tmp_path_factory):
    if request.param == 'formal_api':
        library = NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform == 'darwin'
                                  else 'libimprint_renderer.so')
        assert library.is_file(), 'rebuild renderer for formal API validation'
        dependencies = [NATIVE/'src'/f'{name}.cpp' for name in SOURCE_NAMES]
        dependencies += [NATIVE/'src'/f'{name}.hpp' for name in SOURCE_NAMES]
        dependencies += [NATIVE/'src/acr_stage3_prepared_image_api.cpp',
                         NATIVE/'include/imprint_renderer.h']
        assert library.stat().st_mtime_ns >= max(p.stat().st_mtime_ns for p in dependencies), 'rebuild stale formal renderer'
        lib = C.CDLL(str(library))
        function = lib.im_native_acr_stage3_prepared_image_u16
        function.argtypes = [U16, S, S, I32, D, C.c_uint16, C.c_int, C.c_double,
                             C.c_int, C.c_uint16, C.c_uint16, S, S, U16, S]
        function.restype = C.c_int
        return lib, function
    compiler = shutil.which('clang++') or shutil.which('c++')
    assert compiler, 'C++ compiler required'
    inputs = []
    for name in SOURCE_NAMES:
        source, header = NATIVE/'src'/f'{name}.cpp', NATIVE/'src'/f'{name}.hpp'
        assert source.is_file() and header.is_file(), name
        obj = NATIVE/'build/CMakeFiles/imprint_renderer.dir/src'/f'{name}.cpp.o'
        if request.param == 'official_object':
            assert obj.is_file(), f'rebuild missing {obj}'
            assert obj.stat().st_mtime_ns >= max(source.stat().st_mtime_ns,
                                                header.stat().st_mtime_ns), name
            inputs.append(obj)
        else:
            inputs.append(source)
    target = tmp_path_factory.mktemp('stage3_image_'+request.param)/'bridge.so'
    args = [compiler, '-std=c++17', '-O2', '-fPIC', '-fno-fast-math',
            '-ffp-contract=off', '-dynamiclib' if sys.platform == 'darwin' else '-shared',
            '-I', str(NATIVE/'src'), str(NATIVE/'tools/acr_stage3_prepared_image_bridge.cpp'),
            *map(str, inputs), '-o', str(target)]
    result = subprocess.run(args, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout+result.stderr
    lib = C.CDLL(str(target))
    function = lib.acr_test_stage3_prepared_image
    function.argtypes = [U16, S, S, I32, D, C.c_uint16, C.c_int, C.c_double,
                         C.c_int, C.c_uint16, C.c_uint16, S, S, U16, S]
    function.restype = C.c_int
    return lib, function


def run(bridge, source, rect, *, gains=(1., .45550453738287877, .8786001338393127),
        floor=4100, cubic=0, amount=1., suppress=0, hot=8000, cold=6000,
        tile=(32, 24), destination=None, destination_count=None,
        source_count=None, stride=None):
    top, left, bottom, right = rect
    count = max(0, bottom-top)*max(0, right-left)*3
    if destination is None:
        destination = np.full(count+11, SENTINEL, dtype=np.uint16)
    params = np.asarray(rect, dtype=np.int32)
    gs = np.asarray(gains, dtype=np.float64)
    status = bridge[1](source.ctypes.data_as(U16), source.size if source_count is None else source_count,
                       source.shape[1] if stride is None else stride,
                       params.ctypes.data_as(I32), gs.ctypes.data_as(D), floor,
                       cubic, amount, suppress, hot, cold, *tile,
                       destination.ctypes.data_as(U16),
                       destination.size if destination_count is None else destination_count)
    return status, destination


@pytest.mark.parametrize('cubic,amount,suppress', [(0, 1., 0), (1, 1., 0),
                                                 (0, 6., 0), (1, 6., 0),
                                                 (1, 6., 1)])
def test_full_image_matches_untiled_independent_reference(bridge, cubic, amount, suppress):
    rng = np.random.default_rng(20261020)
    source = np.full((9, 18), 0xD00D, dtype=np.uint16)
    source[:, :13] = rng.integers(4100, 62000, (9, 13), dtype=np.uint16)
    source[4, 6] = 65535
    before = source.copy()
    rect = (-5, 3, 4, 16)
    expected = oracle.reference(source, rect, cubic=cubic, amount=amount,
                                suppress=bool(suppress))
    for tile in ((1, 1), (7, 5), (32, 24)):
        status, output = run(bridge, source, rect, cubic=cubic, amount=amount,
                             suppress=suppress, tile=tile)
        assert status == 0, (status, tile)
        np.testing.assert_array_equal(output[:expected.size].reshape(expected.shape), expected)
        assert np.all(output[expected.size:] == SENTINEL)
        np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize('shape,origin', [((2, 2), (-1, -1)), ((17, 23), (3, 5))])
@pytest.mark.parametrize('kind', ['constant', 'ramp', 'impulse'])
def test_tile_seams_and_image_edges(bridge, shape, origin, kind):
    height, width = shape
    yy, xx = np.indices(shape)
    source = np.full(shape, 17000, np.uint16)
    if kind == 'ramp':
        source[:] = 5000 + yy*501 + xx*103
    elif kind == 'impulse':
        source[height//2, width//2] = 65535
    top, left = origin
    rect = (top, left, top+height, left+width)
    status, baseline = run(bridge, source, rect, tile=(1024, 1024))
    assert status == 0
    for tile in ((1, 1), (7, 5), (8, 8)):
        status, output = run(bridge, source, rect, tile=tile)
        assert status == 0, (status, kind, tile)
        np.testing.assert_array_equal(output, baseline)


@pytest.mark.parametrize('changes', [
    {'cubic': 2}, {'suppress': -1}, {'tile': (0, 8)}, {'tile': (8, 1025)},
    {'gains': (0., 1., 1.)}, {'gains': (float('nan'), 1., 1.)},
    {'floor': 65535}, {'amount': float('inf')}, {'source_count': 3},
    {'source_count': S(-1).value}, {'stride': S(-1).value},
    {'destination_count': 11}, {'destination_count': S(-1).value},
])
def test_rejections_preserve_all_output(bridge, changes):
    source = np.full((2, 2), 12000, np.uint16)
    before = source.copy()
    destination = np.full(23, SENTINEL, np.uint16)
    status, output = run(bridge, source, (0, 0, 2, 2), destination=destination,
                         **changes)
    assert status != 0
    assert np.all(output == SENTINEL)
    np.testing.assert_array_equal(source, before)


def test_unrepresentable_halo_rejects_atomically(bridge):
    source = np.full((2, 2), 12000, np.uint16)
    status, output = run(bridge, source, (-2147483648, 0, -2147483646, 2))
    assert status != 0
    assert np.all(output == SENTINEL)


@pytest.mark.parametrize('overlap', ['source', 'rect', 'gains'])
def test_buffer_api_rejects_alias_before_writing(bridge, overlap):
    source = np.full((2, 2), 12000, np.uint16)
    rect = np.asarray([0, 0, 2, 2], np.int32)
    gains = np.asarray([1., .5, .8], np.float64)
    if overlap == 'source':
        backing = np.full(20, 12000, np.uint16)
        source = backing[:4].reshape(2, 2)
        destination = backing[2:14]
    elif overlap == 'rect':
        backing = np.zeros(48, np.uint8)
        rect = backing[:16].view(np.int32)
        rect[:] = [0, 0, 2, 2]
        destination = backing[:24].view(np.uint16)
    else:
        backing = gains.view(np.uint8)
        destination = backing.view(np.uint16)
    before = backing.copy()
    status = bridge[1](source.ctypes.data_as(U16), source.size, 2,
                       rect.ctypes.data_as(I32), gains.ctypes.data_as(D),
                       4100, 0, 1., 0, 8000, 6000, 8, 8,
                       destination.ctypes.data_as(U16), 12)
    assert status != 0
    np.testing.assert_array_equal(backing, before)


@pytest.mark.parametrize('which', ['source', 'destination'])
def test_buffer_api_rejects_unaligned_samples_atomically(bridge, which):
    source = np.full((2, 2), 12000, np.uint16)
    destination = np.full(13, SENTINEL, np.uint16)
    rect = np.asarray([0, 0, 2, 2], np.int32)
    gains = np.ones(3, np.float64)
    sp = C.cast(source.ctypes.data + (which == 'source'), U16)
    dp = C.cast(destination.ctypes.data + (which == 'destination'), U16)
    before = destination.copy()
    status = bridge[1](sp, 4, 2, rect.ctypes.data_as(I32), gains.ctypes.data_as(D),
                       4100, 0, 1., 0, 8000, 6000, 8, 8, dp, 12)
    assert status != 0
    np.testing.assert_array_equal(destination, before)


def native_module():
    sys.path.insert(0, str(ROOT/'src'))
    import native_renderer
    return native_renderer


def test_python_adapter_uses_official_cpu_pixels(monkeypatch):
    native = native_module()
    path = NATIVE/'build'/('libimprint_renderer.0.1.0.dylib' if sys.platform == 'darwin'
                           else 'libimprint_renderer.so')
    library = C.CDLL(str(path))
    native._configure_library(library)
    monkeypatch.setattr(native, '_get_library', lambda: (library, path))
    source = np.full((5, 8), 12000, np.uint16)[:, ::2]
    before = source.copy()
    gains = (1., .45550453738287877, .8786001338393127)
    actual = native.native_acr_stage3_prepared_image(
        source, gains, source_floor=4100, cubic=False, amount=1.,
        origin=(-3, 5), tile_shape=(3, 2))
    expected = oracle.reference(source, (-3, 5, 2, 9), gains=gains)
    assert actual.dtype == np.uint16 and actual.shape == (5, 4, 3)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, before)


@pytest.mark.parametrize('changes', [
    {'source_floor': True}, {'source_floor': 65535}, {'cubic': 1},
    {'suppress_refine': 0}, {'tile_shape': (1., 2)}, {'tile_shape': (1, 0)},
    {'origin': (False, 0)}, {'origin': (2147483647, 0)},
    {'amount': float('nan')}, {'amount': True}, {'hot_threshold': -1},
])
def test_python_adapter_rejects_before_library_lookup(monkeypatch, changes):
    native = native_module()

    def unexpected_lookup():
        raise AssertionError('invalid inputs must reject before loading native')

    monkeypatch.setattr(native, '_get_library', unexpected_lookup)
    arguments = dict(source_floor=4100, cubic=False, amount=1.)
    arguments.update(changes)
    with pytest.raises(ValueError):
        native.native_acr_stage3_prepared_image(
            np.full((2, 2), 12000, np.uint16), (1., 1., 1.), **arguments)
