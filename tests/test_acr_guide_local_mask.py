"""Same-invocation Camera/log/mask composition, geometry and atomicity."""
import ctypes as C
import numpy as np
import pytest

import test_acr_guide_camera_black as camera
from test_acr_guide_camera_black import native
from test_acr_guide_gray_log import reference as gray_reference
from acr_support import guide_mask_oracle as oracle


def compose(native,source,controls,order,*,bounds,stride,flags=0,selector=2,ev=.75,
            epsilon=2**-12,mode=0,boundary=2,reuse=0,count=None):
    source=np.ascontiguousarray(source,np.float32).reshape(-1)
    bounds=np.asarray(bounds,np.int32)
    meta=np.full(70,-999.,np.float64);levels=np.zeros(32,np.uint32)
    log=np.full(max(5,source.size),-999.,np.float32);delta=log.copy()
    fn=native.library.acr_test_guide_local_mask
    fn.argtypes=[camera.FLOAT_PTR,camera.SIZE,camera.SIZE,C.POINTER(C.c_int32),camera.DOUBLE_PTR,
        camera.U32_PTR,camera.U32,camera.U32,camera.DOUBLE,camera.DOUBLE,C.c_int,camera.U32,camera.U32,
        camera.DOUBLE_PTR,camera.U32_PTR,camera.FLOAT_PTR,camera.FLOAT_PTR,camera.SIZE]
    fn.restype=C.c_int
    status=fn(camera._float_ptr(source),source.size if count is None else count,stride,
        bounds.ctypes.data_as(C.POINTER(C.c_int32)),camera._double_ptr(controls),camera._u32_ptr(order),
        flags,selector,ev,epsilon,mode,boundary,reuse,camera._double_ptr(meta),camera._u32_ptr(levels),
        camera._float_ptr(log),camera._float_ptr(delta),log.size)
    return status,meta,levels.reshape(16,2),log[:int(meta[38])],delta[:int(meta[39])]


@pytest.mark.parametrize('shape',[(3,5),(1,17),(3,257)])
def test_same_request_full_chain(native,shape):
    h,w=shape;stride=3*w+5
    rng=np.random.default_rng(w)
    source=np.full(h*stride,np.nan,np.float32)
    for y in range(h):source[y*stride:y*stride+3*w]=rng.uniform(.001,.9,3*w)
    before=source.view(np.uint32).copy()
    order=np.array([2,0,1],np.uint32)
    for selector,flags,black,mode,epsilon in [(1,1,.0625,0,2**-12),(2,0,0.,1,3.214e-6),(2,1,.0625,2,2**-12)]:
        controls=camera.make_controls(direct=False,black=black)
        rgb=camera.expected_pixels(source,controls,order,flags,selector,height=h,width=w,source_row_stride=stride)
        log=gray_reference(rgb,.75,epsilon).reshape(h,w)
        expected,stats,sigma,level=oracle.reference(log,epsilon,mode)
        status,m,levels,actual_log,actual=compose(native,source,controls,order,bounds=[0,0,h,w],stride=stride,
            flags=flags,selector=selector,mode=mode,epsilon=epsilon)
        assert status==0
        camera.assert_bits(actual_log,log.ravel());camera.assert_bits(actual,expected.ravel())
        np.testing.assert_array_equal(m[20:38].view(np.uint64),stats.view(np.uint64))
        assert m[10]==sigma and m[9]==m[11]==level
        sizes,selected=oracle.geometry(w,h)
        np.testing.assert_array_equal(levels[:len(sizes)],sizes)
        assert m[8]==len(sizes) and selected==level and np.all(levels[len(sizes):]==0)
        np.testing.assert_array_equal(m[12:20],[epsilon,.75,mode,2,0,0,h,w])
        np.testing.assert_array_equal(m[42:55],controls)
        np.testing.assert_array_equal(m[55:60],[*order,flags,selector])
        assert tuple(m[:3])==(h,w,1) and tuple(m[40:42])==(h,w)
    np.testing.assert_array_equal(source.view(np.uint32),before)


def test_invalid_request_and_alias_preserve_complete_result(native):
    source=np.array([.1,.2,.3,.4,.5,.6],np.float32)
    controls=camera.make_controls(direct=True,black=0.)
    order=np.arange(3,dtype=np.uint32)
    failures=[(dict(bounds=[1,0,2,2]),17),(dict(bounds=[0,1,1,3]),17),
        (dict(bounds=[0,0,1,1]),2),(dict(bounds=[0,0,-1,2]),2),
        (dict(bounds=[0,0,1,65536]),2),(dict(boundary=4),18),(dict(boundary=0),18),
        (dict(mode=3),9),(dict(epsilon=0.),9),(dict(ev=np.nan),9),(dict(stride=5),3),(dict(count=5),4)]
    for reuse in (0,1,2):
        for changes,expected_status in failures:
            args=dict(bounds=[0,0,1,2],stride=6,reuse=reuse);args.update(changes)
            status,m,levels,log,delta=compose(native,source,controls,order,**args)
            assert status==expected_status
            assert tuple(m[:2])==(91,87) and m[8]==13 and m[11]==12
            old_source=source[:args.get('count',source.size)]
            camera.assert_bits(log,old_source if reuse==1 else np.full(5,-1234.5,np.float32))
            camera.assert_bits(delta,old_source if reuse==2 else np.full(5,-4321.5,np.float32))
            np.testing.assert_array_equal(levels,np.tile([43,47],(16,1)))
        status,m,_,log,delta=compose(native,source,controls,order,bounds=[0,0,1,2],stride=6,reuse=reuse)
        assert status==0
        expected_log=gray_reference(source.reshape(-1,3),.75,2**-12)
        expected,_,_,_=oracle.reference(expected_log.reshape(1,2),2**-12,0)
        camera.assert_bits(log,expected_log);camera.assert_bits(delta,expected.ravel())
        bad=source.copy();bad[-1]=np.nan
        status,_,_,_,_=compose(native,bad,controls,order,bounds=[0,0,1,2],stride=6,reuse=reuse)
        assert status==camera.CAMERA_FAILURE


def test_geometry_thresholds_and_limit(native):
    fn=native.library.acr_test_local_tone_geometry
    fn.argtypes=[camera.U32,camera.U32,camera.U32_PTR,camera.U32_PTR,camera.U32_PTR]
    fn.restype=C.c_int
    for w,h in ((1,1),(1,17),(127,3),(128,3),(254,3),(255,3),(256,3),(257,3),(65535,1)):
        values=np.full(32,0xffffffff,np.uint32);count=camera.U32();selected=camera.U32()
        assert fn(w,h,camera._u32_ptr(values),C.byref(count),C.byref(selected))==0
        sizes,index=oracle.geometry(w,h)
        assert count.value==len(sizes) and selected.value==index
        np.testing.assert_array_equal(values.reshape(16,2)[:len(sizes)],sizes)
    for w,h in ((0,3),(65536,1),(65535,65535)):
        values=np.full(32,0xffffffff,np.uint32);count=camera.U32(91);selected=camera.U32(87)
        assert fn(w,h,camera._u32_ptr(values),C.byref(count),C.byref(selected))!=0
        assert np.all(values==0xffffffff) and count.value==91 and selected.value==87
