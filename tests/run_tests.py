# Desktop test of the reeling-bar detector and the result-panel timing (same algorithm as Cv.kt / Pn in Cv.kt).
# Run:  pip install opencv-python-headless numpy   then   python3 run_tests.py
import random, numpy as np, proto_detector as P
color=0xFF1A8522; color2=0xFF2181A9
errs=[];miss=0;mem=0;N=100
for seed in range(N):
    rnd=random.Random(seed)
    tipf=rnd.uniform(0.05,0.93); zc=rnd.uniform(0.2,0.8); br=rnd.uniform(0.8,1.25)
    P.MEM.update(box=None,t=-1)
    img0,_=P.make_frame(zone_c=zc,tip_frac=max(0.02,zc-0.15),bright=br,seed=seed)
    P.scan_full(P.downsample(img0,4),4,color,color2,28,t=0)
    img,(zl,zw,tx)=P.make_frame(zone_c=zc,tip_frac=tipf,bright=br,seed=seed+100)
    r=P.scan_full(P.downsample(img,4),4,color,color2,28,t=33)
    if r is None: miss+=1; continue
    if r.get('mem'): mem+=1
    errs.append(abs(r['tip']-tx))
print('bar frames',N,'| missed',miss,'| found via bar memory',mem,'| tip error px max',max(errs),'mean',round(sum(errs)/len(errs),1))
fp=0
for seed in range(30):
    P.MEM.update(box=None,t=-1)
    img,_=P.make_frame(bar=False,seed=seed+7)
    if P.scan_full(P.downsample(img,4),4,color,color2,28,t=1000+seed) is not None: fp+=1
print('false bar on frames without a bar:',fp,'/30')
