"""Class mask to GeoJSON polygon conversion."""
import numpy as np
def mask_to_geojson(mask,origin_lon=-10,origin_lat=50,pixel_size_deg=.001,min_pixels=3):
    try:
        from skimage.measure import label, regionprops
        labels=label(np.asarray(mask)==1,connectivity=2); feats=[]
        for r in regionprops(labels):
            if r.area<min_pixels: continue
            y0,x0,y1,x1=r.bbox; ring=[[origin_lon+x0*pixel_size_deg,origin_lat-y0*pixel_size_deg],[origin_lon+x1*pixel_size_deg,origin_lat-y0*pixel_size_deg],[origin_lon+x1*pixel_size_deg,origin_lat-y1*pixel_size_deg],[origin_lon+x0*pixel_size_deg,origin_lat-y1*pixel_size_deg],[origin_lon+x0*pixel_size_deg,origin_lat-y0*pixel_size_deg]]
            feats.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},'properties':{'pixel_area':int(r.area),'area_km2':float(r.area*pixel_size_deg**2*12321)}})
        return {'type':'FeatureCollection','features':feats}
    except ImportError:
        # Dependency-free connected components fallback.
        a=np.asarray(mask)==1; seen=set(); feats=[]
        for y,x in zip(*np.where(a)):
            if (y,x) in seen: continue
            stack=[(y,x)]; seen.add((y,x)); pts=[]
            while stack:
                cy,cx=stack.pop(); pts.append((cy,cx))
                for ny,nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0<=ny<a.shape[0] and 0<=nx<a.shape[1] and a[ny,nx] and (ny,nx) not in seen: seen.add((ny,nx)); stack.append((ny,nx))
            if len(pts)>=min_pixels:
                ys,xs=zip(*pts); y0,y1=min(ys),max(ys)+1; x0,x1=min(xs),max(xs)+1
                ring=[[origin_lon+x0*pixel_size_deg,origin_lat-y0*pixel_size_deg],[origin_lon+x1*pixel_size_deg,origin_lat-y0*pixel_size_deg],[origin_lon+x1*pixel_size_deg,origin_lat-y1*pixel_size_deg],[origin_lon+x0*pixel_size_deg,origin_lat-y1*pixel_size_deg],[origin_lon+x0*pixel_size_deg,origin_lat-y0*pixel_size_deg]]
                feats.append({'type':'Feature','geometry':{'type':'Polygon','coordinates':[ring]},'properties':{'pixel_area':len(pts),'area_km2':len(pts)*pixel_size_deg**2*12321}})
        return {'type':'FeatureCollection','features':feats}
