# Desktop mirror of Cv.scanRoi (v2.2): HSV match on the small box around the bar, column counts, tip with decimals.
# Frames are drawn at 4x size and shrunk to full size, so the edge of the blue fill is anti-aliased like a real screen.
import numpy as np, cv2, proto_detector as P
color=0xFF1A8522; color2=0xFF2181A9
MEM={'zl':None}
def scan_roi(rgb,x0,y0,screen_w,tol=28):
    h,w=rgb.shape[:2]
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV)
    zm=P.mask_for(hsv,color,tol); bm=P.mask_for(hsv,color2,tol)
    ya=h//4; yb=max(ya,h-1-h//4); rows=yb-ya+1
    zc=(zm[ya:yb+1]>0).sum(axis=0); bc=(bm[ya:yb+1]>0).sum(axis=0)
    cells=0; zl=-1; zr=-1; run=0; zs=-1
    for xx in range(w+1):
        zn=zc[xx] if xx<w else 0
        if xx<w and zn*2>=rows:
            if run==0: zs=xx
            run+=1; cells+=zn
        else:
            if run>=2:
                if zl<0: zl=zs
                zr=xx
            run=0
    mem=False
    if zl>=0: MEM.update(zl=x0+zl,zr=x0+zr,cells=cells)
    elif MEM['zl'] is not None: mem=True
    else: return None
    tip=-1.0; brun=0; bs=-1
    for xx in range(w+1):
        ok=xx<w and bc[xx]*2>=rows
        if ok:
            if brun==0: bs=xx
            brun+=1
        else:
            if brun>=3 and bs<=0.03*screen_w:
                last=bc[xx-1]/rows; nxt=bc[xx]/rows if xx<w else 0.0
                tip=min(max((xx-1)+last+nxt,xx-1.0),xx+1.0)
            brun=0
    if mem and (tip<0 or tip>=w-3): return None
    return dict(zl=(MEM['zl'] if mem else x0+zl),zr=(MEM['zr'] if mem else x0+zr),tip=x0+tip if tip>=0 else -1,cells=cells,mem=mem)
if __name__=='__main__':
    rng=np.random.RandomState(0); nm=0; e_int=[];e_sub=[];miss=0;covered=0;N=200
    for i in range(N):
        zc=rng.uniform(0.3,0.7); tipf=rng.uniform(0.1,0.9)
        img,(zl,zw,tx)=P.make_frame(zone_c=zc,tip_frac=tipf,bright=rng.uniform(0.85,1.2),seed=i)
        H,W=img.shape[:2]
        barW=int(0.772*H); bl=W//2-barW//2; by=int(0.55*H)
        y0=by-6; y1=by+44+6; x0=bl-2; x1=min(W,zl+2*zw)
        MEM['zl']=None if i%1==0 and False else MEM['zl']
        r=scan_roi(img[y0:y1,x0:x1],x0,y0,W)
        if r is None or r['tip']<0:
            if tx>=zl: covered+=1
            else: miss+=1
            continue
        nm+=bool(r.get('mem'))
        e_sub.append(abs(r['tip']-tx)); e_int.append(abs(round(r['tip']-0.5)-tx))
    print("box scan: frames",N,"| zone hidden under the fill",covered,"| of the rest found via zone memory",nm,"| missed while the zone was visible",miss,"| tip error px mean %.2f max %.1f"%(np.mean(e_sub),np.max(e_sub)))
