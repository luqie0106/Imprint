"""Saved Gaussian weights and V/H baseline arithmetic on logical borders.

Zero-origin geometry is a static transcription. Host exp/fmaf, no original
matrix/request producer or optimized runtime dispatch execution.
"""
import ctypes as C
import numpy as np
from ordinary_resample_oracle import fma32, _horizontal_baseline

LIBM=C.CDLL(None)
LIBM.exp.argtypes=[C.c_double]
LIBM.exp.restype=C.c_double
Q=1<<32


def weights(q,phase):
    scale=min(q,1.)
    radius=int(2./scale+float.fromhex('0x1.fff2e48e8a71ep-1'))
    raw=[]
    total=0.
    for j in range(2*radius):
        x=scale*((j-radius+1)-phase/128.)
        value=np.float32(LIBM.exp((-2.*x)*x) if x<2. else 0.)
        raw.append(value)
        total+=float(value)
    return np.float32(np.asarray(raw)*np.float32(1./total))


def axis(source_size,output_size):
    if source_size==output_size:return None
    step=int((source_size/output_size)*Q+.5)
    scale=1./(step/Q)
    radius=int(2./min(scale,1.)+.9999)
    lower=(1-radius)*Q
    upper=(source_size-radius)*Q
    origin=(step>>1)+lower-0x7f000000
    coordinates=[min(upper,max(lower,origin+i*step)) for i in range(output_size)]
    return dict(step=step,scale=scale,radius=radius,lower=lower,upper=upper,origin=origin,
                coordinates=[(v//Q,(v>>25)&127) for v in coordinates])


def resize(source,shape):
    source=np.asarray(source,np.float32)
    h,w=source.shape;oh,ow=shape
    v_axis,h_axis=axis(h,oh),axis(w,ow)
    middle=np.empty((oh,w),np.float32)
    if v_axis is None:middle[:]=source
    else:
        table={p:weights(v_axis['scale'],p) for _,p in v_axis['coordinates']}
        for y,(start,phase) in enumerate(v_axis['coordinates']):
            for x in range(w):
                total=np.float32(0.)
                for j,weight in enumerate(table[phase]):
                    if weight!=np.float32(0.):
                        total=fma32(source[min(h-1,max(0,start+j)),x],weight,total)
                middle[y,x]=total
    if h_axis is None:return middle
    output=np.empty((oh,ow),np.float32)
    table={p:weights(h_axis['scale'],p) for _,p in h_axis['coordinates']}
    for x,(start,phase) in enumerate(h_axis['coordinates']):
        row=table[phase]
        indices=np.clip(np.arange(start,start+len(row)),0,w-1)
        for y in range(oh):output[y,x]=_horizontal_baseline(middle[y,indices],row,len(row))
    return output


def pyramid(source):
    levels=[np.array(source,dtype=np.float32,copy=True)]
    while len(levels)<16 and max(levels[-1].shape)>=2:
        levels.append(resize(levels[-1],tuple((n+1)//2 for n in levels[-1].shape)))
    return levels
