import numpy as np
from reel_sim import *

class Ledger:
    """New engine: every tap is entered in a ledger and counted as 'in flight' until a picture shows its lift.
    Nothing depends on a guessed tap-to-game delay: the delay is MEASURED (fire time -> lift seen) and only used
    for the look-ahead. Gain and drain are learned from the same pictures."""
    def __init__(s,G,D,delay=35.0,hold=35.0,minGap=70.0,fps=45):
        s.G=G;s.D=D;s.L=delay;s.hold=hold;s.minGap=max(minGap,hold+20)
        s.taps=[]            # dict(fire, state 0=in flight 1=seen 2=lost)
        s.last=-1e9;s.pts=None;s.pp=None;s.acc=0.0;s.zc=None;s.zw=None;s.err=2.0;s.seenAt=0
        s.quiet=0.0;s.quietP=None;s.quietT=None;s.dt_ema=25.0
    def inflight(s): return [t for t in s.taps if t['st']==0]
    def see(s,o,now):
        _,ts,tip,zc,zw,hidden=o
        s.zc,s.zw=zc,zw
        if hidden: return
        if s.pts is not None and ts>s.pts:
            dt=(ts-s.pts)/1000.0
            if dt<0.3:
                s.dt_ema=0.9*s.dt_ema+0.1*dt*1000
                r=tip-(s.pp-s.D*dt)                 # lift since the last picture (px)
                s.acc+=r
                fl=s.inflight()
                # attribute lift to the oldest taps still in flight, one tap per 0.6 G
                while fl and s.acc>=0.6*s.G and fl[0]['fire']<ts:
                    t=fl.pop(0); t['st']=1
                    lat=ts-t['fire']-0.5*s.dt_ema
                    if 5<lat<200: s.L=0.95*s.L+0.05*lat
                    one=(len(fl)==0 and s.acc<1.6*s.G)
                    if one:
                        s.G=0.96*s.G+0.04*min(1.6*s.G,max(0.5*s.G,s.acc))
                    s.acc-=s.G
                s.acc=max(-0.5*s.G,min(s.acc,2.5*s.G)) if fl else 0.0
                # tap never showed a lift: drop it from the in-flight count after a generous time
                for t in fl:
                    if ts-t['fire']>max(220.0,3*s.L): t['st']=2
                # drain from quiet stretches (no tap in flight)
                if not s.inflight() and s.quietT is not None and s.quietT<=ts:
                    qd=(ts-s.quietT)/1000.0
                    if qd>=0.08:
                        d=(s.quietP-tip)/qd
                        if 0.5*s.D<d<1.6*s.D: s.D=0.96*s.D+0.04*d
                        s.quietT=ts;s.quietP=tip
                else:
                    s.quietT=None
                if not s.inflight() and s.quietT is None: s.quietT=ts;s.quietP=tip
                s.err=0.92*s.err+0.08*abs(r- (0 if not fl else 0))*0
        s.pts=ts;s.pp=tip
        s.taps=[t for t in s.taps if t['st']==0 or t['fire']>ts-1500]
    def step(s,now):
        if s.pts is None or s.zc is None: return False
        if now-s.pts>300: return False
        n=len(s.inflight())
        T=now+s.L+s.hold/2
        pT=s.pp-s.D*(T-s.pts)/1000+s.G*n
        trig=s.zc-0.5*s.G
        if pT<trig and now-s.last>=s.minGap:
            s.last=now; s.taps.append(dict(fire=now,st=0)); return True
        return False

def sc(fac,**kw): return np.mean([run(fac,seed=s,subpx=True,**kw) for s in range(1,6)],0)
if __name__=='__main__':
    cases=[("80px",dict(zw=80)),("moving",dict(zw=80,zv=120)),("L=80",dict(zw=80,L=80)),("G/D off",dict(zw=80,mis=(1.18,0.85))),
           ("20fps",dict(zw=80,fps=20)),("all",dict(zw=80,L=80,mis=(1.18,0.85),zv=80)),("cancel 15%",dict(zw=80,cancel=0.15)),("60px",dict(zw=60))]
    print("%-12s %-16s %-16s"%("case","old timed","ledger"))
    for n,kw in cases:
        o=np.mean([run(lambda G,D:OldTimed(G,D),seed=s,subpx=False,**kw) for s in range(1,6)],0)
        l=sc(lambda G,D:Ledger(G,D),**kw)
        print("%-12s %5.1f / %5.1f (%.1f/s)   %5.1f / %5.1f (%.1f/s)"%(n,o[0],o[1],o[2],l[0],l[1],l[2]))
