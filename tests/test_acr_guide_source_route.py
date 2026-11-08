"""Source level, independent strength slot, and cached/current level branches."""
import ctypes as C
import itertools
import numpy as np

import test_acr_guide_camera_black as camera
from test_acr_guide_camera_black import native


def call(native,*,profile=0,ad9=0,adb=0,presence=0,strength=0,cached=0,current=0):
    fn=native.library.acr_test_guide_source_route
    fn.argtypes=[camera.U32]*7+[camera.U32_PTR]
    fn.restype=C.c_int
    output=np.full(6,0xa5a5a5a5,np.uint32)
    status=fn(profile,ad9,adb,presence,strength,cached,current,camera._u32_ptr(output))
    return status,output


def expected(ad9,adb,presence):
    initial=3 if ad9 and adb else 2
    candidates=[level for level in range(initial,0,-1) if presence&(1<<level)]
    level=candidates[0] if candidates else 0
    tested=sum(1<<i for i in range(initial,max(1,level)-1,-1))
    return [initial,level,tested,int(bool(presence&(1<<level)))]


def test_source_availability_and_byte_flags(native):
    for ad9,adb,presence in itertools.product((0,1,2,128,255),(0,1,2,128,255),range(64)):
        status,output=call(native,ad9=ad9,adb=adb,presence=presence)
        assert status==0
        np.testing.assert_array_equal(output[:4],expected(ad9,adb,presence))
    # Levels4/5 are never selected by this loop; missing level0 is observable.
    assert call(native,ad9=1,adb=1,presence=48)[1].tolist()==[3,0,14,0,0,0]


def test_distinct_slot_and_transfer_domains(native):
    for strength,cached,current in itertools.product((0,1,2,3,4,5,0x7fffffff,0x80000000,0xffffffff),
                                                    (0,1,2,3,5,0xffffffff),(0,1,2,3,5,0xffffffff)):
        status,output=call(native,ad9=1,adb=1,presence=8,strength=strength,cached=cached,current=current)
        assert status==0 and output[1]==3
        assert output[4]==min(strength,4)
        assert output[5]==(0 if cached==current else 1 if current<cached else 2)


def test_profile_branch_preserves_plan_and_destination(native):
    for presence in range(64):
        status,out=call(native,profile=1,ad9=1,adb=1,presence=presence,strength=4,cached=2,current=1)
        assert status==19
        np.testing.assert_array_equal(out,np.full(6,0xa5a5a5a5,np.uint32))
