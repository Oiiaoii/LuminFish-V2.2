import numpy as np
from reel_sim import *
from sim2 import P2
class V(P2):
    def __init__(s,G,D,trig_off=0.0,sp=0.4,**k): super().__init__(G,D,**k); s.to=trig_off; s.sp=sp
    def step(s,now):
        if s.est is None or s.lastz is None: return False
        pts,pp=s.est; zc,zw=s.lastz
        if now-pts>300: return False
        la=sum(1 for t in s.taps if t+s.L>pts)
        T=now+s.L+s.hold/2
        pT=pp-s.D*(T-pts)/1000+s.G*la
        if pT<zc-0.5*s.G+s.to and now-s.last>=max(s.minGap,s.sp*s.G/s.D*1000):
            s.last=now; s.taps.append(now); return True
        return False
def sc(fac,**kw):
    return np.mean([run(fac,seed=s,subpx=True,**kw) for s in range(1,6)],0)
for to in (-8,-4,0,4):
  for sp in (0.4,0.5,0.6):
    r=sc(lambda G,D:V(G,D,to,sp),zw=80)
    print("off",to,"spacing",sp,"-> %.1f / %.1f"%(r[0],r[1]))
