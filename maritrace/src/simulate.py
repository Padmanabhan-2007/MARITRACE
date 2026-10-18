"""Deterministic synthetic maritime scenario builder."""
from datetime import datetime,timedelta,timezone
import numpy as np,pandas as pd
def scenario(seed=42):
    rng=np.random.default_rng(seed); lat,lon=50.,-10.; t=datetime(2026,1,1,12,tzinfo=timezone.utc); rows=[]
    for j in range(20):
        mmsi=f'235{j:06}'; typ='Tanker' if j%4==0 else ('Cargo' if j%3==0 else 'Fishing'); cx=lat+rng.uniform(-.4,.4); cy=lon+rng.uniform(-.5,.5)
        for h in range(-12,1):
            # One true source loiters at the source and then departs; two decoys have a gap or pass nearby.
            near=(j==0 and h>=-5) or (j==1 and -5<=h<=-2)
            la=cx+rng.normal(0,.002) if near else cx+h*rng.uniform(-.001,.001); lo=cy+rng.normal(0,.002) if near else cy+h*rng.uniform(-.001,.001)
            if j==0 and h in (-3,-2): continue
            rows.append({'mmsi':mmsi,'timestamp':(t+timedelta(hours=h)).isoformat(),'lat':la,'lon':lo,'sog':rng.uniform(.3,1.5) if near else rng.uniform(4,15),'cog':rng.uniform(0,360),'ship_type':typ,'name':f'Synthetic vessel {j:02d}'})
    ais=pd.DataFrame(rows); features=[]
    for j,g in ais.groupby('mmsi'):
        distance=float(np.hypot((g.lat-lat)*111,(g.lon-lon)*71).min()); gap=3 if j.endswith('000000') else 0
        features.append({'mmsi':j,'name':g['name'].iloc[0],'ship_type':g.ship_type.iloc[0],'minimum_distance_km':distance,'loiter_fraction':float((g.sog<2).mean()),'max_gap_hours':gap,'is_tanker_cargo':int(g.ship_type.iloc[0] in ('Tanker','Cargo'))})
    return {'seed':seed,'timestamp':t.isoformat(),'lat':lat,'lon':lon,'wind_u':4.,'wind_v':2.,'cur_u':.2,'cur_v':.1,'ais':ais,'features':features,'true_source':'235000000'}
