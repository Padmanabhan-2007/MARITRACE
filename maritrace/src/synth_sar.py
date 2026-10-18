"""Domain-randomized synthetic SAR-like tiles (not physical SAR simulation)."""
import argparse, json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter, rotate
from PIL import Image, ImageDraw
from .config import IMAGE_SIZE, SEED

def generate_sample(size: int = IMAGE_SIZE, seed: int | None = None, forced_spill=None):
    rng=np.random.default_rng(seed); low=gaussian_filter(rng.normal(size=(size,size)),size/24); low=(low-low.min())/(np.ptp(low)+1e-8)
    image=np.clip(.32 + .14*low + rng.normal(0,.035,(size,size)),0,1).astype(np.float32); mask=np.zeros((size,size),np.uint8)
    yy,xx=np.mgrid[:size,:size]
    landfield=gaussian_filter(rng.normal(size=(size,size)),size/12)
    if rng.random()<.25:
        edge=int(rng.integers(0,4)); field=landfield.copy()
        if edge==0: field[:size//3]+=1
        elif edge==1: field[-size//3:]+=1
        elif edge==2: field[:,:size//3]+=1
        else: field[:,-size//3:]+=1
        mask[field>np.quantile(field,.82)]=4; image[mask==4]=.72+rng.random(np.count_nonzero(mask==4))*.2
    def patch(cx,cy,rx,ry,angle,cls,damp):
        x=xx-cx; y=yy-cy; a=np.deg2rad(angle); u=x*np.cos(a)+y*np.sin(a); v=-x*np.sin(a)+y*np.cos(a)
        n=gaussian_filter(rng.normal(size=(size,size)),max(2,rx/4)); n=(n-n.mean())/(n.std()+1e-8)
        shape=(u/rx)**2+(v/ry)**2 < (1+.18*n)
        shape &= mask!=4
        if cls==1: shape &= (mask==0)|(mask==2); image[shape]*=damp*(.8+.35*rng.random(np.count_nonzero(shape))); mask[shape]=1
        else: shape &= mask==0; image[shape]*=damp*(.82+.4*rng.random(np.count_nonzero(shape))); mask[shape]=2
    if forced_spill is not None:
        patch(int(forced_spill[0]),int(forced_spill[1]),size*.12,size*.035,25,1,.32)
    elif rng.random()<.7:
        margin=max(2,int(size*.08)); hi=max(margin+1,size-margin)
        for _ in range(int(rng.integers(1,4))): patch(int(rng.integers(margin,hi)),int(rng.integers(margin,hi)),rng.uniform(size*.05,size*.16),rng.uniform(size*.018,size*.05),rng.uniform(0,180),1,rng.uniform(.2,.5))
    if rng.random()<.5:
        for _ in range(int(rng.integers(1,3))):
            # Include both round and elongated look-alikes, with damping overlapping oil.
            rx=rng.uniform(size*.04,size*.095); ry=rng.uniform(size*.03,size*.067)
            if rng.random()<.45: rx,ry=max(rx,ry*1.2),min(rx,ry)
            margin=max(2,int(size*.08)); hi=max(margin+1,size-margin)
            patch(int(rng.integers(margin,hi)),int(rng.integers(margin,hi)),rx,ry,rng.uniform(0,180),2,rng.uniform(.25,.58))
    if rng.random()<.4:
        for _ in range(int(rng.integers(1,6))):
            cx,cy=int(rng.integers(5,size-5)),int(rng.integers(5,size-5)); mask[max(0,cy-2):cy+3,max(0,cx-2):cx+3]=3; image[max(0,cy-2):cy+3,max(0,cx-2):cx+3]=rng.uniform(.8,1)
    looks=rng.uniform(1,4); image*=rng.gamma(looks,1/looks,(size,size)); image=np.clip(image,0,1)
    return image[None].astype(np.float32),mask

def generate_dataset(out_dir, n_train=2000,n_val=400,n_test=400,size=256,seed=42):
    out=Path(out_dir); ids={}
    for split,n,offset in [('train',n_train,0),('val',n_val,1000000),('test',n_test,2000000)]:
        d=out/split; d.mkdir(parents=True,exist_ok=True); seeds=[]
        for i in range(n):
            s=seed+offset+i; im,ma=generate_sample(size,s); np.save(d/f'{i:05}.npy',im[0]); Image.fromarray(ma).save(d/f'{i:05}.png'); seeds.append(s)
        ids[split]=seeds
    (out/'metadata.json').write_text(json.dumps({'simulated':True,'seed_ranges':{k:[min(v),max(v)] if v else [] for k,v in ids.items()}},indent=2))
    return ids

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--out',default='data/synthetic'); p.add_argument('--n_train',type=int,default=2000); p.add_argument('--n_val',type=int,default=400); p.add_argument('--n_test',type=int,default=400); a=p.parse_args(); generate_dataset(a.out,a.n_train,a.n_val,a.n_test)
