# Reeling-bar physics simulator + tap controllers (desktop mirror of the tap engine in MacroService.kt).
# Run: python3 reel_sim.py
# Physics (measured on a 2340 px screen): one tap lifts the fill by G ~58 px (after a tap-to-game delay L),
# the fill drains by itself at D ~215 px/s. The zone is a window [zl, zr]; score = % of time the tip is inside it.
import numpy as np, sys

class Physics:
    def __init__(s, rng, G=58.0, D=215.0, L=45.0, Ljit=12.0, zw=180.0, zv=0.0, cancel=0.03):
        s.rng=rng; s.G=G; s.D=D; s.L=L; s.Ljit=Ljit; s.cancel=cancel
        s.p=700.0; s.zc=1200.0; s.zw=zw; s.zv=zv; s.pending=[]   # (land_time_ms, amount)
        s.t=0.0
    def tap(s,t):
        if s.rng.rand()<s.cancel: return
        s.pending.append((t+s.L+s.rng.randn()*s.Ljit, s.G*(1+0.04*s.rng.randn())))
    def step(s,dt=1.0):
        s.t+=dt
        s.p=max(0.0,s.p-s.D*dt/1000.0)
        keep=[]
        for lt,a in s.pending:
            if lt<=s.t: s.p+=a
            else: keep.append((lt,a))
        s.pending=keep
        s.zc+=s.zv*dt/1000.0
        if s.zc<900 or s.zc>1500: s.zv=-s.zv
    def inzone(s): return abs(s.p-s.zc)<=s.zw/2

class Capture:
    """frames at fps, picture shows the state at its time stamp; usable by the controller after pipe+scan ms."""
    def __init__(s,rng,fps=45,pipe=28.0,scan=12.0,noise=1.5,step=2,miss=0.03,subpx=False):
        s.rng=rng; s.per=1000.0/fps; s.pipe=pipe; s.scan=scan; s.noise=noise; s.step=step; s.miss=miss; s.subpx=subpx
        s.next=0.0; s.q=[]
    def tick(s,ph):
        out=None
        if ph.t>=s.next:
            ts=ph.t; s.next+=s.per*(1+0.15*s.rng.randn()*0.3)
            tip=ph.p+s.rng.randn()*s.noise
            if not s.subpx: tip=round(tip/s.step)*s.step     # whole-pixel tip (scan step)
            hidden = s.rng.rand()<s.miss
            s.q.append((ts+s.pipe+s.scan,ts,tip,ph.zc,ph.zw,hidden))
        if s.q and s.q[0][0]<=ph.t: out=s.q.pop(0)
        return out

class OldTimed:
    """port of the v2.0 timed branch (engineStep): trigger on predicted position, fixed margin"""
    def __init__(s,G,D,delay=35.0,hold=35.0,minGap=70.0,margin=0.05):
        s.G=G;s.D=D;s.delay=delay;s.hold=hold;s.minGap=max(minGap,hold+20);s.margin=margin
        s.taps=[];s.last=-1e9;s.obs=None
    def see(s,o,now): s.obs=o
    def step(s,now):
        if s.obs is None: return False
        _,ts,tip,zc,zw,hidden=s.obs
        if hidden or now-s.obs[0]>300: return False
        tMs=min(1500,max(80,s.G/s.D*1000))
        lead=s.delay+s.hold/2
        nn=sum(1 for t in s.taps if t>ts-s.delay)
        p0=tip-s.D*(now-ts)/1000+s.G*nn
        at=p0-s.D*lead/1000
        trig=zc-0.5*s.G-s.margin*zw
        spacing=max(s.minGap,0.5*tMs)
        if at<trig and now-s.last>=spacing:
            s.last=now; s.taps.append(now); return True
        return False

class Precision:
    """new: state observer (tip, drain, gain, tap latency learned online), sub-pixel tip, look-ahead trigger that
    aims the sawtooth at the centre of the zone and scales its safety margin with the measured prediction error."""
    def __init__(s,G,D,delay=35.0,hold=35.0,minGap=70.0,zone_w=180.0):
        s.G=G;s.D=D;s.L=delay;s.hold=hold;s.minGap=max(minGap,hold+20)
        s.taps=[];s.last=-1e9;s.est=None;s.err=3.0;s.lastz=None
        s.Gn=0;s.Dn=0;s.lat_hist=[];s.pend=None
    def see(s,o,now):
        _,ts,tip,zc,zw,hidden=o
        if hidden: s.lastz=(zc,zw); return
        s.lastz=(zc,zw)
        if s.est is not None:
            pts,pp=s.est
            dt=(ts-pts)/1000.0
            if 0<dt<0.25:
                landed=[t for t in s.taps if pts<t+s.L<=ts]
                pred=pp-s.D*dt+s.G*len(landed)
                e=tip-pred
                s.err=0.9*s.err+0.1*abs(e)
                if len(landed)==0 and dt>=0.06:                 # quiet stretch: learn the drain
                    d=(pp-tip)/dt
                    if 0.5*s.D<d<1.6*s.D: s.D=0.93*s.D+0.07*d; s.Dn+=1
                elif len(landed)==1 and dt<0.12:               # exactly one tap: learn the gain and the latency
                    g=(tip-pp)+s.D*dt
                    if 0.4*s.G<g<1.8*s.G: s.G=0.94*s.G+0.06*g; s.Gn+=1
                    # latency: was the lift already visible? residual sign tells if the model landed it too early / late
                    s.L=min(140,max(5,s.L+0.5*np.clip(-e/s.D*1000/ max(1,len(landed)),-6,6)*0))  # (see latency below)
                # latency learner: a tap whose lift is not yet in the picture although predicted -> L larger
                for t in s.taps[-4:]:
                    if pts<t+s.L<=ts:
                        rise=(tip-pp)+s.D*dt
                        if rise<0.3*s.G: s.L=min(140,s.L+3)       # lift missing: tap lands later than assumed
                        elif rise>1.6*s.G: s.L=max(5,s.L-2)
        s.est=(ts,tip)
        s.obs=o
    def step(s,now):
        if s.est is None or s.lastz is None: return False
        pts,pp=s.est
        zc,zw=s.lastz
        if now-pts>300-0: return False
        landed_after=sum(1 for t in s.taps if t+s.L>pts)           # taps whose lift the picture can not show yet
        T=now+s.L+s.hold/2                                         # when a tap fired now would take effect
        pT=pp-s.D*(T-pts)/1000+s.G*landed_after
        margin=min(0.2*zw,1.0*s.err)
        trig=zc-0.5*s.G+0.0
        # fire early enough that, with the measurement error, the tip never falls under the lower edge
        low=zc-0.5*zw+margin
        spacing=max(s.minGap,0.4*s.G/s.D*1000)
        if (pT<trig or pT<low) and now-s.last>=spacing:
            s.last=now; s.taps.append(now); return True
        return False

def run(ctrl_factory,zw=180,zv=0.0,fps=45,L=45.0,mis=(1.0,1.0),secs=40,seed=1,subpx=False,pipe=28.0,noise=1.5,cancel=0.03):
    rng=np.random.RandomState(seed)
    ph=Physics(rng,zw=zw,zv=zv,L=L,cancel=cancel)
    cap=Capture(rng,fps=fps,pipe=pipe,subpx=subpx,noise=noise)
    c=ctrl_factory(ph.G*mis[0],ph.D*mis[1])
    ins=0;tot=0;taps=0;ctr=0
    for ms in range(int(secs*1000)):
        ph.step(1.0)
        o=cap.tick(ph)
        if o is not None: c.see(o,ph.t)
        if ms%4==0 and c.step(ph.t):
            ph.tap(ph.t); taps+=1
        if ms>3000:
            tot+=1; ins+=ph.inzone(); ctr+=abs(ph.p-ph.zc)<=ph.zw/4
    return 100*ins/tot, 100*ctr/tot, taps/secs

if __name__=='__main__':
    cases=[("exact model, zone 180px",dict(zw=180)),
           ("zone 120px",dict(zw=120)),
           ("zone 80px",dict(zw=80)),
           ("zone 80px, moving zone 120px/s",dict(zw=80,zv=120)),
           ("zone 80px, tap delay 80ms (assumed 35)",dict(zw=80,L=80)),
           ("zone 80px, gain +18% drain -15% off",dict(zw=80,mis=(1.18,0.85))),
           ("zone 80px, 20 fps capture",dict(zw=80,fps=20)),
           ("zone 80px, delay 80ms + gain/drain off + moving",dict(zw=80,L=80,mis=(1.18,0.85),zv=80))]
    print("%-52s %18s %18s"%("case","old timed (in/centre %)","precision (in/centre %)"))
    for name,kw in cases:
        o=[];n=[]
        for sd in range(1,6):
            o.append(run(lambda G,D:OldTimed(G,D),seed=sd,**kw))
            n.append(run(lambda G,D:Precision(G,D),seed=sd,subpx=True,**kw))
        o=np.mean(o,0);n=np.mean(n,0)
        print("%-52s   %5.1f / %5.1f  (%3.1f/s)   %5.1f / %5.1f  (%3.1f/s)"%(name,o[0],o[1],o[2],n[0],n[1],n[2]))
