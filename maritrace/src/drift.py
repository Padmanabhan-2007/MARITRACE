"""Simple constant-flow backward drift approximation."""
import math
from .config import WIND_FACTOR, BACK_DRIFT_HOURS
def back_drift(lat,lon,wind_u,wind_v,cur_u,cur_v,hours,wind_factor=WIND_FACTOR):
    east=(cur_u+wind_factor*wind_u)*hours*3600; north=(cur_v+wind_factor*wind_v)*hours*3600
    return lat-north/111320, lon-east/(111320*max(.01,math.cos(math.radians(lat))))
def drift_path(lat,lon,wind_u,wind_v,cur_u,cur_v,hours=BACK_DRIFT_HOURS,wind_factor=WIND_FACTOR):
    return [{'hours':h,'lat':back_drift(lat,lon,wind_u,wind_v,cur_u,cur_v,h,wind_factor)[0],'lon':back_drift(lat,lon,wind_u,wind_v,cur_u,cur_v,h,wind_factor)[1]} for h in hours]
