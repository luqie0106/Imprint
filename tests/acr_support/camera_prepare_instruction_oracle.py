"""Independent 3-plane camera compensation producer from ARM64 instructions.

00877fa8,00740ea0,0131f884 and0131fa10. This is static prepared
arithmetic; it does not resolve a real profile/white/source or run Adobe.
"""
from __future__ import annotations
import ctypes as C
import math
import struct
import numpy as np

_libm=C.CDLL(None)
fma=_libm.fma
fma.argtypes=[C.c_double]*3
fma.restype=C.c_double


def constant(bits):
    return struct.unpack('>d',int(bits,16).to_bytes(8,'big'))[0]


HUE=constant('3f6b089a02752547')
SAT=constant('3f70624dd2f1a9fc')
TINT=constant('bf2a36e2eb1c432d')


def matmul(left,right):
    out=[]
    for r in range(3):
        for c in range(3):
            accumulator=0.
            for k in range(3):
                accumulator=fma(left[r*3+k],right[k*3+c],accumulator)
            out.append(accumulator)
    return tuple(out)


def matvec(matrix,vector):
    return tuple(fma(matrix[r*3+2],vector[2],
                    fma(matrix[r*3+1],vector[1],
                        fma(matrix[r*3],vector[0],0.))) for r in range(3))


def diagonal(v):
    return (v[0],0.,0.,0.,v[1],0.,0.,0.,v[2])


def prepare(controls):
    if len(controls)!=31 or not all(math.isfinite(float(v)) for v in controls):
        raise ValueError('31 finite controls required')
    x=tuple(map(float,controls))
    white,T,P,source=x[:3],x[3:12],x[12:21],x[21:24]
    calibration,tint=x[24:30],x[30]
    if any(v<.001 or v>1. for v in white) or max(white)<math.nextafter(1.,0.):
        raise ValueError('unsupported white')
    if any(v<=0. for v in source):
        raise ValueError('unsupported source')
    if any(not v.is_integer() or not -100.<=v<=100. for v in (*calibration,tint)):
        raise ValueError('unsupported calibration')
    hr,sr,hg,sg,hb,sb=(calibration[0]*HUE,calibration[1]*SAT,
                        calibration[2]*HUE,calibration[3]*SAT,
                        calibration[4]*HUE,calibration[5]*SAT)
    H=((hg+1.)-hb,-hg,hb,hr,(hb+1.)-hr,-hb,-hr,hg,(hr+1.)-hg)
    one_plus_sr=sr+1.
    S=((sg+1.)+sb,-sg,-sb,-sr,one_plus_sr+sb,-sb,-sr,-sg,one_plus_sr+sg)
    calibration_matrix=matmul(S,H)
    calibrated=matmul(calibration_matrix,P)
    A=matmul(calibrated,T)
    maximum=source[0] if source[0]>source[1] else source[1]
    maximum=maximum if maximum>source[2] else source[2]
    normalized=tuple(v/maximum for v in source)
    scalar=tint*TINT
    tintvector=matvec(diagonal(normalized),(scalar,scalar,scalar))
    vector=matvec(A,tintvector)
    minimum=vector[0] if vector[0]<vector[1] else vector[1]
    minimum=minimum if minimum<vector[2] else vector[2]
    denominator=1.-minimum
    if denominator==0. or not math.isfinite(denominator):
        raise ValueError('unsupported compensation denominator')
    bases=tuple((v-minimum)/denominator for v in vector)
    M=matmul(A,diagonal(white))
    if not all(map(math.isfinite,(*bases,*M))):
        raise ValueError('nonfinite prepared result')
    with np.errstate(over='ignore'):
        matrix=np.asarray(M,dtype=np.float32).astype(np.float64)
    if not np.isfinite(matrix).all():
        raise ValueError('float matrix overflow')
    return {'matrix':matrix,'bases':np.asarray(bases,np.float64),
            'vector':vector,'minimum':minimum,'denominator':denominator,'A':A,
            'calibration_matrix':calibration_matrix,'tint_vector':tintvector}
