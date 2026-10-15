"""DNG wire-format, ordering, orientation and fallback contracts."""
import struct
from dataclasses import replace
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import dng_warp as w
import native_dense
import native_renderer
from test_dng_gainmap import _gainmap_payload


def payload(coeff=None, planes=1, center=(.5,.5)):
    if coeff is None:
        coeff = [1, .06, 0, 0, .001, -.002] * planes
    return struct.pack('>I', planes) + struct.pack('>'+'d'*(6*planes+2), *coeff, *center)


def opcode(kind, data, version=0x01030000, flags=0):
    return struct.pack('>4I', kind, version, flags, len(data)) + data


def dng(tmp_path, ops, orientation=1, crop=False):
    listing = struct.pack('>I', len(ops)) + b''.join(ops)
    tags = [(256,4,1,struct.pack('<I',5)), (257,4,1,struct.pack('<I',5)),
            (274,3,1,struct.pack('<H',orientation)), (51022,7,len(listing),listing)]
    if crop:
        tags += [(50719,4,2,struct.pack('<II',1,1)), (50720,4,2,struct.pack('<II',3,3))]
    tags.sort()
    offset = 8+2+12*len(tags)+4
    entries = []; extra = b''
    for tag, typ, count, data in tags:
        value = data.ljust(4,b'\0') if len(data)<=4 else struct.pack('<I',offset+len(extra))
        entries.append(struct.pack('<HHI4s',tag,typ,count,value))
        if len(data)>4: extra += data
    path=tmp_path/'fixture.dng'
    path.write_bytes(b'II'+struct.pack('<HI',42,8)+struct.pack('<H',len(tags))+b''.join(entries)+struct.pack('<I',0)+extra)
    return path


@pytest.mark.parametrize('planes',[1,3])
def test_actual_opcode_wire_format(tmp_path, planes):
    path=dng(tmp_path,[opcode(1,payload(planes=planes))])
    parsed=w.parse_dng_warp(path)
    assert parsed.planes==planes and parsed.coefficients.shape==(3,6)
    assert not parsed.coefficients.flags.writeable


@pytest.mark.parametrize('bad',[b'', struct.pack('>I',2), payload(center=(float('nan'),.5)), payload(center=(2,.5)),payload()[:-1]])
def test_bad_payload_fails(tmp_path,bad):
    with pytest.raises(w.DNGWarpError): w.parse_dng_warp(dng(tmp_path,[opcode(1,bad)]))


@pytest.mark.parametrize('ops',[[opcode(1,payload(),version=0x02000000)], [opcode(999,b'') ,opcode(1,payload())], [opcode(1,payload()),opcode(1,payload())]])
def test_unsupported_pipeline_does_not_partially_apply(tmp_path,ops):
    image=np.zeros((5,5,3),np.uint16)
    result, warp, gain=w.apply_dng_warp_correction(image,dng(tmp_path,ops))
    assert result is image and not warp and not gain


def test_optional_unknown_can_be_skipped(tmp_path):
    assert w.parse_dng_warp(dng(tmp_path,[opcode(999,b'',flags=1),opcode(1,payload())])) is not None


@pytest.mark.parametrize('gain_first',[True,False])
def test_opcode_order_and_shared_highlight_reference(tmp_path,monkeypatch,gain_first):
    import dng_gainmap
    ops=[opcode(9,_gainmap_payload()),opcode(1,payload())]
    if not gain_first: ops.reverse()
    path=dng(tmp_path,ops)
    image=np.full((5,5,3),1000,np.uint16); reference=np.full_like(image,5000)
    events=[]
    def warp(value, constants, backend):
        events.append('warp'); return value+np.uint16(7),'test_gpu'
    def gain(value, gain_map, **kw):
        events.append('gain')
        expected=reference if gain_first else reference+np.uint16(7)
        np.testing.assert_array_equal(kw['highlight_reference'],expected)
        return value+np.uint16(9)
    monkeypatch.setattr(w,'_apply_native_or_reference',warp)
    monkeypatch.setattr(dng_gainmap,'_apply_gain_map',gain)
    out, applied, gained=w.apply_dng_warp_correction(image,path,highlight_reference=reference)
    assert applied and gained
    assert events==(['gain','warp'] if gain_first else ['warp','warp','gain'])
    np.testing.assert_array_equal(out,image+np.uint16(16))
    assert (image==1000).all() and (reference==5000).all()


@pytest.mark.parametrize('orientation',[1,3,6,8])
def test_orientation_round_trip(tmp_path,monkeypatch,orientation):
    path=dng(tmp_path,[opcode(1,payload())],orientation=orientation)
    sensor=np.arange(75,dtype=np.uint16).reshape(5,5,3)
    to_sensor={1:0,3:2,6:1,8:3}[orientation]
    display=np.ascontiguousarray(np.rot90(sensor,-to_sensor))
    def warp(value, constants, backend):
        np.testing.assert_array_equal(value,sensor)
        return np.flip(value,1),'test_gpu'
    monkeypatch.setattr(w,'_apply_native_or_reference',warp)
    output, applied, _=w.apply_dng_warp_correction(display,path)
    assert applied
    np.testing.assert_array_equal(output,np.rot90(np.flip(sensor,1),-to_sensor))


def test_ambiguous_crop_requires_full_frame_provenance(tmp_path):
    path=dng(tmp_path,[opcode(1,payload())],crop=True)
    image=np.zeros((3,3,3),np.uint16)
    assert not w.apply_dng_warp_correction(image,path)[1]
    assert w.apply_dng_warp_correction(image,path,backend='cpu',full_frame=True)[1]


def test_gpu_cpu_python_fallback_and_fault_boundaries(monkeypatch):
    image=np.full((5,5,3),1000,np.uint16)
    c=np.array([1,0,0,0,0,0]*3+[.5,.5,1],np.float32)
    calls=[]
    def missing(*args):
        calls.append('gpu'); raise native_renderer.NativeRendererError('old library')
    def cpu(*args): calls.append('cpu'); return image.copy()
    monkeypatch.setattr(native_renderer,'native_warp_rectilinear',missing)
    monkeypatch.setattr(native_dense,'warp_rectilinear',cpu)
    assert w._apply_native_or_reference(image,c,'auto')[1]=='cpp_cpu'
    assert calls==['gpu','cpu']
    monkeypatch.setattr(native_dense,'warp_rectilinear',missing)
    out,backend=w._apply_native_or_reference(image,c,'auto')
    assert backend=='python'
    np.testing.assert_array_equal(out,image)
    def invalid(*args): raise ValueError('invalid data')
    monkeypatch.setattr(native_renderer,'native_warp_rectilinear',invalid)
    with pytest.raises(ValueError): w._apply_native_or_reference(image,c,'auto')


def test_reference_matches_cpp_with_tangential_and_pixel_aspect():
    rng=np.random.default_rng(33)
    image=rng.integers(0,65536,(83,127,3),dtype=np.uint16)
    c=np.array([1,.018,-.002,.0001,.0007,-.0004]*3+[.48,.52,.83],np.float32)
    try: native=native_dense.warp_rectilinear(image,c)
    except native_renderer.NativeRendererError as e: pytest.skip(str(e))
    reference=w._warp_reference(image,c)
    assert np.abs(native.astype(int)-reference.astype(int)).max()<=1
