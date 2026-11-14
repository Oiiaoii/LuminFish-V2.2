import numpy as np
from reel_sim import *
from sim5 import Ledger

class Ledger2(Ledger):
    """+ cancelled taps are removed from the ledger the moment Android reports the cancel, + the zone is followed
    ahead of where it was seen (speed from the last pictures)"""
    def __init__(s,*a,**k):
        super().__init__(*a,**k); s.zh=[]; s.zv=0.0
    def cancelled(s,fire):
        for t in s.taps:
            if t['fire']==fire and t['st']==0: t['st']=2
    def see(s,o,now):
        super().see(o,now)
        _,ts,tip,zc,zw,hidden=o
        s.zh.append((ts,zc)); s.zh=[z for z in s.zh if ts-z[0]<400]
        old=[z for z in s.zh if 100<=ts-z[0]<=260]
        if old:
            z0=old[0]; v=(zc-z0[1])/((ts-z0[0])/1000.0)
            s.zv=0.0 if abs(v)<25 else 0.6*s.zv+0.4*v
    def step(s,now):
        if s.pts is None or s.zc is None: return False
        if now-s.pts>300: return False
        n=len(s.inflight()); T=now+s.L+s.hold/2
        pT=s.pp-s.D*(T-s.pts)/1000+s.G*n
        zT=s.zc+max(-s.zw,min(s.zw,s.zv*(T-s.pts)/1000))
        if pT<zT-0.5*s.G and now-s.last>=s.minGap:
            s.last=now; s.taps.append(dict(fire=now,st=0)); return True
        return False

# hook the cancel callback into the simulator: Physics knows when a tap was dropped
import reel_sim
def run2(fac,zw=80,zv=0.0,fps=45,L=45.0,mis=(1,1),secs=40,seed=1,cancel=0.03,tell=True):
    rng=np.random.RandomState(seed); ph=Physics(rng,zw=zw,zv=zv,L=L,cancel=cancel); cap=Capture(rng,fps=fps,subpx=True)
    c=fac(ph.G*mis[0],ph.D*mis[1]); ins=tot=taps=ctr=0; pend_cancel=[]
    for ms in range(secs*1000):
        ph.step(1.0); o=cap.tick(ph)
        if o: c.see(o,ph.t)
        for fire,at in list(pend_cancel):
            if ph.t>=at and hasattr(c,'cancelled') and tell: c.cancelled(fire); pend_cancel.remove((fire,at))
        if ms%4==0 and c.step(ph.t):
            n0=len(ph.pending); ph.tap(ph.t); taps+=1
            if len(ph.pending)==n0: pend_cancel.append((ph.t,ph.t+60))   # Android says 'cancelled' ~60 ms later
        if ms>3000: tot+=1; ins+=ph.inzone(); ctr+=abs(ph.p-ph.zc)<=ph.zw/4
    return 100*ins/tot,100*ctr/tot,taps/secs
if __name__=='__main__':
    cases=[("80px",dict(zw=80)),("moving",dict(zw=80,zv=120)),("L=80",dict(zw=80,L=80)),("G/D off",dict(zw=80,mis=(1.18,0.85))),
           ("20fps",dict(zw=80,fps=20)),("all",dict(zw=80,L=80,mis=(1.18,0.85),zv=80)),("cancel 15%",dict(zw=80,cancel=0.15)),("60px",dict(zw=60)),("60px moving",dict(zw=60,zv=120))]
    print("%-12s %-18s %-18s"%("case","ledger","ledger+cancel+zone"))
    for n,kw in cases:
        a=np.mean([run2(lambda G,D:Ledger(G,D),seed=s,**kw) for s in range(1,6)],0)
        b=np.mean([run2(lambda G,D:Ledger2(G,D),seed=s,**kw) for s in range(1,6)],0)
        print("%-12s %5.1f / %5.1f         %5.1f / %5.1f"%(n,a[0],a[1],b[0],b[1]))
