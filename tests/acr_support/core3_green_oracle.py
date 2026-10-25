# Independently transcribed from original four-lane instructions 0080f7e4..0080fa34.
import ctypes
import ctypes.util
import numpy as np
_fma = ctypes.CDLL(ctypes.util.find_library('m') or None).fmaf
_fma.argtypes = [ctypes.c_float] * 3
_fma.restype = ctypes.c_float
f = np.float32

def fma(a,b,c): return f(_fma(float(a),float(b),float(c)))
def clamp(a):
    if np.isnan(a): return a
    return f(max(0.,min(float(a),1.)))
def addseq(values):
    a=f(values[0])
    for b in values[1:]: a=f(a+b)
    return a

def angle(a,b):
    # FDIV is reciprocal first, followed by FMUL. The two zero masks
    # replace reciprocal results and the final angle, respectively.
    za=a==0; zb=b==0
    ia=f(0) if za else f(f(1)/a)
    ib=f(0) if zb else f(f(1)/b)
    r0=f(a*ib); r1=f(b*ia)
    c0=np.array([0x4273e173],np.uint32).view(np.float32)[0]
    c1=np.array([0xc17f85cc],np.uint32).view(np.float32)[0]
    p0=f(r0*fma(r0,c1,c0)); p1=f(r1*fma(r1,c1,c0))
    out=f(f(90)-p0) if b>=a else p1
    return f(0) if za and zb else out

def reference(source,planes,y,x,width,phase,parameters):
    t,k,m,q=map(f,parameters); green=np.empty(width,np.float32);aux=np.empty(width,np.float32)
    def s(dy,dx): return source[y+dy,x+j+dx]
    def p(n,dy,dx): return planes[n][y+dy,x+j+dx]
    for j in range(width):
        # v0/v1, then v2/v3 are built independently in instruction order.
        v0=addseq([p(8,0,d) for d in (-2,-1,0,1,2)])
        v1=addseq([p(7,d,0) for d in (-2,-1,0,1,2)])
        v2=addseq([p(1,-3,0),p(1,-2,-1),p(1,-1,0),p(1,0,-1),p(1,1,0),p(1,2,-1),p(1,3,0),p(0,-2,1),p(0,0,1),p(0,2,1)])
        v3=addseq([p(5,-1,-2),p(5,-1,0),p(5,-1,2),p(5,0,-3),p(5,0,-1),p(5,0,1),p(5,0,3),p(2,1,-2),p(2,1,0),p(2,1,2)])
        v0=angle(v0,v1); v1=angle(v3,v2)
        v2=clamp(f(f(f(f(v0+v1)*f(.5))-t)*k))
        if q==0: v2=f(.5)
        elif q<f(.5):
            inv=f(f(1)/f(q+q)); base=v2
            v4=fma(f(base-f(1)),inv,f(1))
            if not base>=f(f(1)-q): v4=f(.5)
            v2=f(base*inv) if q>=base else v4
        twice=f(s(0,0)+s(0,0))
        h=clamp(f(f(.5)*fma(f(f(twice-s(0,-2))-s(0,2)),f(.25),f(s(0,-1)+s(0,1)))))
        v=clamp(f(f(.5)*fma(f(f(twice-s(-2,0))-s(2,0)),f(.25),f(s(-1,0)+s(1,0)))))
        v7=f(h+f(f(v-h)*v2))
        hi=f(f(90)-t)
        c0=v0>hi;c1=t>v0;c2=v1>hi;c3=t>v1
        if c0: v7=v
        if c1: v7=h
        v6=v if c2 else v7
        v5=h if c3 else v6
        green[j]=s(0,0) if ((j^phase)&1) else v5
        if m!=0:
            v1=f(v2*f(2));v1=f(abs(f(v1-f(1))));v1=f(f(1)-v1);v1=f(f(2)*v1);v1=clamp(v1);v1=f(v1*m)
            aux[j]=f(0) if c0 or c1 or c2 or c3 else v1
        else: aux[j]=np.nan
    return green,aux
