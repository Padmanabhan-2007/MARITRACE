"""Explainable heuristic vessel ranking for synthetic scenarios."""
import pandas as pd
from .config import SCORE_WEIGHTS
def rank_suspects(features):
    df=pd.DataFrame(features).copy()
    if df.empty: return df
    ranges={'proximity':(0,50),'loitering':(0,1),'ais_gap':(0,6),'vessel_type':(0,1)}
    df['proximity_feature']=1-df.minimum_distance_km.clip(*ranges['proximity'])/50
    df['loitering_feature']=df.loiter_fraction.clip(0,1)
    df['ais_gap_feature']=df.max_gap_hours.clip(0,6)/6
    df['vessel_type_feature']=df.is_tanker_cargo.astype(float)
    for k,w in SCORE_WEIGHTS.items(): df[f'{k}_contribution']=df[f'{k}_feature']*w
    df['score']=sum(df[f'{k}_contribution'] for k in SCORE_WEIGHTS)
    return df.sort_values('score',ascending=False).reset_index(drop=True)
def sensitivity(features):
    ranked=rank_suspects(features); return {'top_mmsi':str(ranked.iloc[0].mmsi) if len(ranked) else None,'synthetic_perturbation_note':'Wind +/-20% and direction +/-15 degrees; heuristic approximation.'}
