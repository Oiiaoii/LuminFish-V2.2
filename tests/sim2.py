import numpy as np
from reel_sim import *

class P2(Precision):
    def __init__(s,G,D,learn_gd=False,learn_lat=False,**k):
        super().__init__(G,D,**k); s.lg=learn_gd; s.ll=learn_lat
    def see(s,o,now):
        _,ts,tip,zc,zw,hidden=o
        s.lastz=(zc,zw)
        if hidden: return
        if s.est is not None:
            pts,pp=s.est
            dt=(ts-pts)/1000.0
            if 0<dt<0.25:
                landed=[t for t in s.taps if pts<t+s.L<=ts]
                pred=pp-s.D*dt+s.G*len(landed)
                e=tip-pred
                s.err=0.9*s.err+0.1*abs(e)
                if s.lg:
                    if len(landed)==0 and dt>=0.06:
                        d=(pp-tip)/dt
                        if 0.5*s.D<d<1.6*s.D: s.D=0.95*s.D+0.05*d
                    elif len(landed)==1 and dt<0.12:
                        g=(tip-pp)+s.D*dt
                        if 0.4*s.G<g<1.8*s.G: s.G=0.95*s.G+0.05*g
        s.est=(ts,tip)

def test(name,fac,subpx=True):
    cases=[dict(zw=80),dict(zw=80,zv=120),dict(zw=80,L=80),dict(zw=80,mis=(1.18,0.85)),dict(zw=80,fps=20),dict(zw=80,L=80,mis=(1.18,0.85),zv=80)]
    r=[]
    for kw in cases:
        r.append(np.mean([run(fac,seed=s,subpx=subpx,**kw) for s in range(1,6)],0))
    print("%-34s"%name," ".join("%5.1f/%4.1f"%(x[0],x[1]) for x in r))

print("%-34s"%"cases: 80px | moving | L80 | G/D off | 20fps | all"," ")
test("old timed",lambda G,D:OldTimed(G,D),subpx=False)
test("new, no learning",lambda G,D:P2(G,D))
test("new, learn G/D",lambda G,D:P2(G,D,learn_gd=True))
