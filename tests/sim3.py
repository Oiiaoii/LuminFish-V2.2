import numpy as np
from reel_sim import *
from sim2 import P2
def bias(fac,subpx,**kw):
    rng=np.random.RandomState(3); ph=Physics(rng,**kw); cap=Capture(rng,subpx=subpx); c=fac(ph.G,ph.D)
    off=[]; 
    for ms in range(40000):
        ph.step(1.0); o=cap.tick(ph)
        if o: c.see(o,ph.t)
        if ms%4==0 and c.step(ph.t): ph.tap(ph.t)
        if ms>3000: off.append(ph.p-ph.zc)
    off=np.array(off); return off.mean(),off.std(),off.min(),off.max()
print("old",bias(lambda G,D:OldTimed(G,D),False,zw=80))
print("new",bias(lambda G,D:P2(G,D),True,zw=80))
