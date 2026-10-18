"""AIS sanitation and coarse per-vessel summaries."""
import pandas as pd
def clean_ais(data):
    df=pd.read_csv(data) if isinstance(data,(str,bytes)) else pd.DataFrame(data).copy()
    df['sog']=pd.to_numeric(df.sog,errors='coerce'); df['lat']=pd.to_numeric(df.lat,errors='coerce'); df['lon']=pd.to_numeric(df.lon,errors='coerce')
    return df[df.sog.between(0,60)&df.lat.between(-90,90)&df.lon.between(-180,180)].drop_duplicates(['mmsi','timestamp']).reset_index(drop=True)
def vessel_features(data,origin_lat,origin_lon):
    df=clean_ais(data); out=[]
    for m,g in df.groupby('mmsi'):
        d=((g.lat-origin_lat)*111).pow(2)+((g.lon-origin_lon)*71).pow(2); out.append({'mmsi':m,'minimum_distance_km':float(d.min()**.5),'loiter_fraction':float((g.sog<2).mean()),'max_gap_hours':0.,'is_tanker_cargo':int(g.ship_type.iloc[0] in ('Tanker','Cargo'))})
    return out
