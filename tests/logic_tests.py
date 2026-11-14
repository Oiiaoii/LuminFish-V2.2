# Desktop mirror of the v1.9 decision logic (same rules as MacroService.kt / Stats.kt). Run: python3 logic_tests.py
import numpy as np

# ---- auto-sell trigger (sellDue)
def sell_due(mode, every, mins, catches, at_sell, now_min, last_min):
    by_count = every > 0 and catches - at_sell >= every
    by_time = mins > 0 and (now_min - last_min) >= mins
    return by_count if mode == 0 else by_time if mode == 1 else (by_count or by_time)
assert sell_due(0, 10, 10, 10, 0, 3, 0) and not sell_due(0, 10, 10, 9, 0, 99, 0)
assert sell_due(1, 10, 10, 1, 0, 10, 0) and not sell_due(1, 10, 10, 500, 0, 9, 0)
assert sell_due(2, 10, 10, 10, 0, 1, 0) and sell_due(2, 10, 10, 1, 0, 10, 0) and not sell_due(2, 10, 10, 9, 0, 9, 0)
assert not sell_due(0, 10, 10, 25, 20, 0, 0) and sell_due(0, 10, 10, 30, 20, 0, 0)   # counts from the last sell
print('auto-sell trigger: ok')

# ---- catch verification (panel seen -> counted; none seen -> not counted; never seen on this phone -> fall back after 4)
def run_verify(events):
    miss=0; ever=False; off=False; counted=0; unver=0
    for panel_seen in events:
        verify_on = not off
        if not verify_on or panel_seen:
            if panel_seen: ever=True; miss=0
            counted+=1
        else:
            miss+=1
            if not ever and miss>=4: off=True; counted+=1
            else: unver+=1
    return counted, unver
assert run_verify([True]*5)==(5,0)
assert run_verify([True,False,True])==(2,1)
assert run_verify([False]*6)==(3,3)          # 3 not counted, 4th switches verification off, then everything counts
print('catch verification: ok', run_verify([False]*6))

# ---- anti-freeze: a frozen screen changes nothing, a live game does
def motion(a,b): return float((np.abs(a-b)>6).mean())
rng=np.random.RandomState(0)
base=rng.randint(0,255,(36,64))
frozen=[base.copy() for _ in range(20)]
live=[]; cur=base.copy()
for i in range(20):
    cur=cur.copy(); cur[20:30,:]=rng.randint(0,255,(10,64)); live.append(cur)   # water moving
ticking=[]; 
for i in range(20):
    f=base.copy(); f[0,0]=(i*40)%255; ticking.append(f)                           # only a clock changes
def frozen_for(frames, limit_frames=10):
    since=0
    for i in range(1,len(frames)):
        if motion(frames[i-1],frames[i])>0.003: since=i
        if i-since>=limit_frames: return True
    return False
assert frozen_for(frozen) and not frozen_for(live) and frozen_for(ticking)
print('anti-freeze: frozen screen detected, live game not flagged, a lone clock does not hide a freeze')

# ---- click recovery: 8 failed taps in a row trigger it, any good tap resets the streak
def streak(res):
    s=0; fires=0
    for ok in res:
        if ok: s=0
        else:
            s+=1
            if s>=8: fires+=1; s=0
    return fires
assert streak([False]*7+[True]+[False]*7)==0 and streak([False]*8)==1 and streak([False]*16)==2
print('click recovery: ok')

# ---- stats math
catches,fails,unver,sec=42,6,2,1800
print('success rate',catches*100//(catches+fails+unver),'% | per hour',catches*3600//sec,'| time per catch',sec//catches,'s')
