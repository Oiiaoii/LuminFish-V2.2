import numpy as np, cv2, random

def rgb2hsv_cv(r,g,b):
    px = np.uint8([[[r,g,b]]])
    return tuple(int(x) for x in cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0,0])

def int_rgb(c):
    c &= 0xffffffff
    return (c>>16)&255,(c>>8)&255,c&255

def mask_for(hsv, color, tol):
    r,g,b = int_rgb(color)
    hz,sz,vz = rgb2hsv_cv(r,g,b)
    ht = 4 + tol//3
    lo = np.array([max(0,hz-ht), max(45,int(sz*0.45)), max(25,int(vz*0.5))],np.uint8)
    hi = np.array([min(179,hz+ht), 255, min(255,int(vz*1.6)+1)],np.uint8)
    return cv2.inRange(hsv, lo, hi)

MEM={'box':None,'zl':0,'zr':0,'t':-1}
def scan_full(rgb, step, color, color2, tol, t=0):
    h,w = rgb.shape[:2]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    zm = mask_for(hsv, color, tol)
    zm = cv2.morphologyEx(zm, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT,(5,3)))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(zm, connectivity=8)
    barL = w/2.0 - 0.386*h*1.0  # same geometry as the old scanner (in downsampled units use h*... see note)
    barR = w/2.0 + 0.386*h
    best=None; bscore=-1
    for i in range(1,n):
        x,y,cw,ch,area = stats[i]
        if ch<3 or cw<2*ch: continue
        if ch<0.025*h or ch>0.06*h: continue
        if cw<0.02*w or cw>w/2: continue
        if area < cw*ch*0.5: continue
        cx=x+cw/2.0; cy=y+ch/2.0
        if cx<barL or cx>barR or cy<0.4*h or cy>0.8*h: continue
        if area>bscore: bscore=area; best=(x,y,cw,ch,area)
    if best is None:
        m=MEM
        if m['box'] is not None and t-m['t']<=400:
            bx,by_,bw,bh_=m['box']
            ya=by_+bh_//4; yb=by_+bh_-1-bh_//4
            bm=mask_for(hsv,color2,tol)
            band=bm[ya:yb+1,bx:bx+bw]
            ok=((band>0).sum(axis=0)*2>=(yb-ya+1))
            tip=-1;run=0;rs=-1
            for k in range(len(ok)+1):
                o=ok[k] if k<len(ok) else False
                if o:
                    if run==0: rs=k
                    run+=1
                else:
                    if run>=3 and rs<=0.03*w and tip<0: tip=bx+k; 
                    run=0
            if tip>=0:
                tk=((hsv[ya:yb+1,bx:bx+bw,1]<80)&(hsv[ya:yb+1,bx:bx+bw,2]<120)).sum(axis=0)*2>=(yb-ya+1)
                if tip-bx>=bw-3: return None
                rest=tk[max(0,tip-bx):]
                if len(rest)>=6 and rest.mean()<0.5: return None
                return dict(zone_area=m['area'],zl=m['zl']*step,zr=m['zr']*step,blue=1,tip=tip*step,red=0,mem=True)
        return None
    x,y,cw,ch,area = best
    MEM.update(box=(max(0,int(barL)-1),y,int(2*0.386*h)+2,ch),zl=x,zr=x+cw,t=t,area=area*step*step/2.0)
    ya = y+ch//4; yb = y+ch-1-ch//4
    if yb<ya: yb=ya
    xa = max(0,int(barL)-1); xb=min(w-1,x+cw-1+cw)
    bm = mask_for(hsv, color2, tol)
    band = bm[ya:yb+1, xa:xb+1]
    colsum = (band>0).sum(axis=0)
    need = (yb-ya+1)
    ok = colsum*2>=need
    tip=-1; bn=0; run=0; rs=-1
    for k in range(len(ok)+1):
        xx = xa+k
        o = ok[k] if k<len(ok) else False
        if o:
            if run==0: rs=xx
            run+=1
        else:
            if run>=3 and (rs-xa)<=0.03*w: tip=xx; bn=run
            run=0
    red = cv2.inRange(rgb, np.array([215,24,25],np.uint8), np.array([255,104,105],np.uint8))
    redn = int(cv2.countNonZero(red))
    return dict(zone_area=area*step*step/2.0, zl=x*step, zr=(x+cw)*step, blue=bn, tip=(tip*step if tip>=0 else -1), red=redn)

def make_frame(W=2340,H=1080, zone_c=0.5, zone_w=94, tip_frac=0.4, bright=1.0, noise=6, distract=True, seed=0, bar=True, panel=False):
    rnd=np.random.RandomState(seed)
    img = np.zeros((H,W,3),np.uint8)
    # sky/water background gradient
    for y in range(H):
        img[y,:,:]=(int(60+40*y/H),int(110+30*y/H),int(160-20*y/H))
    if distract:
        # grass patches (greenish) and blue water chunk
        cv2.rectangle(img,(100,700),(700,1000),(40,140,50),-1)
        cv2.rectangle(img,(1700,300),(2200,420),(30,120,40),-1)   # wide green HUD thing out of geometry
        cv2.rectangle(img,(1000,880),(1400,930),(26,133,34),-1)   # green bar-like but below 0.8h
    if bar:
        barW=int(0.772*H); bl=W//2-barW//2; by=int(0.55*H); bh=44
        cv2.rectangle(img,(bl-4,by-4),(bl+barW+4,by+bh+4),(20,20,24),-1)
        cv2.rectangle(img,(bl,by),(bl+barW,by+bh),(50,50,58),-1)
        zl=int(bl+zone_c*barW-zone_w/2)
        cv2.rectangle(img,(zl,by),(zl+zone_w,by+bh),(26,133,34),-1)
        tipx=int(bl+tip_frac*barW)
        cv2.rectangle(img,(bl,by),(tipx,by+bh),(33,129,169),-1)
    if panel:
        cv2.rectangle(img,(700,150),(1640,900),(25,25,30),-1)
        cv2.rectangle(img,(760,220),(1580,600),(70,70,80),-1)
    f = img.astype(np.float32)*bright + rnd.normal(0,noise,img.shape)
    return np.clip(f,0,255).astype(np.uint8), (zl if bar else None, zone_w, tipx if bar else None)

def downsample(img, step):
    return img[::step, ::step]

if __name__=="__main__":
    color=0xFF1A8522; color2=0xFF2181A9
    results=[]
    for seed in range(40):
        rnd=random.Random(seed)
        tipf=rnd.uniform(0.1,0.9); zc=rnd.uniform(0.2,0.8); br=rnd.uniform(0.8,1.25)
        img,(zl,zw,tx)=make_frame(zone_c=zc,tip_frac=tipf,bright=br,seed=seed)
        step=max(2,2340//500)
        ds=downsample(img,step)
        r=scan_full(ds,step,color,color2,28)
        if r is None: results.append((seed,'MISS',br)); continue
        ez=abs(r['zl']-zl); et=abs(r['tip']-tx) if r['tip']>=0 else 9999
        results.append((seed,ez,et,br,r['zone_area']))
    miss=[x for x in results if x[1]=='MISS']
    ok=[x for x in results if x[1]!='MISS']
    print('frames',len(results),'miss',len(miss), [m[0] for m in miss][:10], [round(m[2],2) for m in miss][:10])
    print('zone edge err px  max',max(x[1] for x in ok),' tip err px max',max(x[2] for x in ok if x[2]<9999),' tip lost',sum(1 for x in ok if x[2]==9999))
    # negatives
    neg=0
    for seed in range(20):
        img,_=make_frame(bar=False,seed=seed)
        r=scan_full(downsample(img,4),4,color,color2,28)
        if r is not None: neg+=1
    print('false positives on frames without bar:',neg,'/20')
