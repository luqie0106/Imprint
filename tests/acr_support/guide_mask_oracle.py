"""Independent full-bank mask reference on prepared log guide, alpha=1/beta=0."""
import numpy as np
from guide_resample_oracle import pyramid, resize
from log_scene_stats_oracle import selected_stats, LIBM
from ordinary_resample_oracle import fma32


def sigma(low,high,epsilon,mode):
    xs=(2.,3.5,5.,6.5,12.,14.)
    ys=(1.4,1.6,2.3,2.5,2.5,2.)
    width=high-low
    if width<=xs[0]:value=ys[0]
    elif width>=xs[-1]:value=ys[-1]
    else:
        i=next(i for i,x in enumerate(xs) if x>=width)
        value=LIBM.fma((width-xs[i-1])/(xs[i]-xs[i-1]),ys[i]-ys[i-1],ys[i-1])
    if mode:
        logarithm=LIBM.log(value)/.6931471805599453
        value=LIBM.exp2(LIBM.fma(mode*.5,.4854268271702417-logarithm,logarithm))
    return LIBM.log2(epsilon+value)


def geometry(width,height):
    levels=[(width,height)]
    while len(levels)<16 and max(levels[-1])>=2:
        levels.append(tuple((n+1)//2 for n in levels[-1]))
    selected=0
    for i,shape in enumerate(levels[1:],1):
        if max(shape)<128:break
        selected=i
    return levels,selected


def full_bank_pass(log,epsilon,mode):
    """One ordinary pass, retaining the selected collapse callback's stats."""
    log=np.asarray(log,np.float32)
    guide=pyramid(log)
    _,level=geometry(log.shape[1],log.shape[0])
    first,_=selected_stats(guide[level].flat,epsilon)
    s=sigma(first[1],first[2],epsilon,mode)
    low=max(np.float32(first[1]),np.float32(np.float32(first[3])-np.float32(20)))
    high=min(np.float32(first[2]),np.float32(np.float32(first[3])+np.float32(20)))
    n=max(1,int(np.ceil(np.float32(high-low))))
    step=np.float32(np.float32(high-low)/np.float32(n))
    inv=np.float32(np.float32(1)/step) if step else np.float32(0)
    anchors=[fma32(step,np.float32(i),low) for i in range(n+1)]
    bank=[pyramid(np.clip(log,np.float32(a-np.float32(s)),np.float32(a+np.float32(s)))) for a in anchors]
    bands=[]
    for lev,g in enumerate(guide[:-1]):
        band=np.zeros_like(g)
        for j,a in enumerate(anchors):
            residual=np.float32(bank[j][lev]-resize(bank[j][lev+1],g.shape))
            before,after=np.float32(a-step),np.float32(a+step)
            for k in np.ndindex(g.shape):
                x=g[k];r=residual[k]
                if j==0:
                    band[k]=r if x<a else np.float32(np.float32(np.float32(x-after)*-inv)*r) if x<after else np.float32(0)
                elif j==n and x>=a:band[k]=np.float32(band[k]+r)
                elif x>=before and (j==n or x<after):
                    weight=np.float32(inv*np.float32(x-before)) if x<a else np.float32(np.float32(x-after)*-inv)
                    band[k]=fma32(r,weight,band[k])
        bands.append(band)
    collapsed=guide[-1]
    second=None
    for lev in range(len(bands)-1,-1,-1):
        collapsed=np.float32(resize(collapsed,bands[lev].shape)+bands[lev])
        if lev==level:second,_=selected_stats(collapsed.flat,epsilon)
    assert second is not None
    return collapsed,first,second,s,level


def swap_left_right(log,inverse=False):
    log=np.asarray(log,np.float32)
    width=log.shape[1]
    assert width>=2
    split=width//2 if inverse else (width+1)//2
    return np.concatenate((log[:,split:],log[:,:split]),axis=1)


def blend_horizontal(first,second):
    first=np.asarray(first,np.float32)
    second=np.asarray(second,np.float32)
    assert first.shape==second.shape and first.shape[1]>=2
    inverse=np.float32(np.float32(1)/np.float32(first.shape[1]-1))
    output=np.empty_like(first)
    for x in range(first.shape[1]):
        t=np.clip(np.float32(np.float32(x)*inverse),np.float32(0),np.float32(1))
        a=abs(fma32(t,np.float32(2),np.float32(-1)))
        wide=float(a)
        weight=np.float32((wide*wide)*LIBM.fma(-2.,wide,3.))
        for y in range(first.shape[0]):
            difference=np.float32(second[y,x]-first[y,x])
            output[y,x]=fma32(weight,difference,first[y,x])
    return output


def reference(log,epsilon,mode,boundary=2):
    log=np.asarray(log,np.float32)
    if boundary not in (2,4):raise ValueError('unsupported boundary mode')
    collapsed,first,second,s,level=full_bank_pass(log,epsilon,mode)
    if boundary==4 and log.shape[1]>=2:
        other,first,second,s,level=full_bank_pass(swap_left_right(log),epsilon,mode)
        collapsed=blend_horizontal(collapsed,swap_left_right(other,inverse=True))
    # Both stats objects are the second pass writes for boundary4. Neither
    # the inverse permutation nor the final blend invokes another callback.
    aligned=list(second)
    shift=first[0]-second[0]
    for i in range(3):aligned[i]+=shift
    delta=np.minimum(np.float32(np.float32(np.float32(first[0])-np.float32(second[0]))+collapsed)-log,np.float32(4))
    return delta,np.asarray([*first,*second,*aligned],np.float64),s,level
