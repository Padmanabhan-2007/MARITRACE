"""Optional FastAPI wrapper around deterministic sample scenes."""
from fastapi import FastAPI,HTTPException
from src.simulate import scenario
from src.score import rank_suspects
from src.drift import drift_path
app=FastAPI(title='MARITRACE simulated analysis API')
@app.get('/health')
def health(): return {'ok':True,'data':'simulated'}
@app.post('/analyze')
def analyze(scene_id:str='demo'):
    if scene_id not in ('demo','scene-1','scene-2','scene-3'): raise HTTPException(404,'Unknown synthetic scene')
    s=scenario({'demo':42,'scene-1':42,'scene-2':43,'scene-3':44}[scene_id]); r=rank_suspects(s['features'])
    return {'simulated':True,'spill':{'type':'Point','coordinates':[s['lon'],s['lat']]},'origin_path':drift_path(s['lat'],s['lon'],s['wind_u'],s['wind_v'],s['cur_u'],s['cur_v']),'suspects':r.to_dict(orient='records')}
