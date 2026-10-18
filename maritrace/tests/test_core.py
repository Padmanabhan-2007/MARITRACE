import numpy as np
import json
from src.synth_sar import generate_sample
from src.synth_sar import generate_dataset
from src.dataset import SynthSARDataset
from src.drift import back_drift
from src.detect import mask_to_geojson
from src.score import rank_suspects
from src.simulate import scenario
from src.ais import clean_ais

def test_generator_contract():
    a,m=generate_sample(64,3); assert a.shape==(1,64,64) and a.dtype==np.float32 and m.shape==(64,64) and set(np.unique(m))<=set(range(5))
def test_drift_direction():
    lat,lon=back_drift(0,0,0,0,1,0,1); assert lat==0 and lon<0
def test_polygons():
    m=np.zeros((12,12),np.uint8); m[2:5,3:6]=1; assert mask_to_geojson(m)['features']
def test_rank():
    r=rank_suspects(scenario(42)['features']); assert r.iloc[0].mmsi=='235000000'
def test_ais_clean():
    d=clean_ais([{'mmsi':'1','timestamp':'a','lat':0,'lon':0,'sog':100,'ship_type':'Cargo'},{'mmsi':'2','timestamp':'a','lat':0,'lon':0,'sog':3,'ship_type':'Cargo'}]); assert len(d)==1

def test_dataset_crop_shape_and_class_frequencies(tmp_path):
    generate_dataset(tmp_path,n_train=2,n_val=1,n_test=1,size=256,seed=77)
    ds=SynthSARDataset(tmp_path,'train',augment=True,crop_size=128)
    x,y=ds[0]; assert tuple(x.shape)==(1,128,128) and tuple(y.shape)==(128,128)
    assert x.dtype==ds.torch.float32
    assert y.dtype==ds.torch.long and np.isclose(ds.class_frequencies().sum(),1)

def test_generated_split_seeds_disjoint(tmp_path):
    generate_dataset(tmp_path,n_train=2,n_val=2,n_test=2,size=32,seed=9)
    ranges=json.loads((tmp_path/'metadata.json').read_text())['seed_ranges']
    all_seeds=[set(range(v[0],v[1]+1)) for v in ranges.values()]
    assert all_seeds[0].isdisjoint(all_seeds[1]) and all_seeds[0].isdisjoint(all_seeds[2]) and all_seeds[1].isdisjoint(all_seeds[2])

def test_model_forward_backward():
    import torch
    import segmentation_models_pytorch as smp
    torch.set_num_threads(min(2,torch.get_num_threads()))
    model=smp.Unet(encoder_name='resnet18',encoder_weights=None,in_channels=1,classes=5)
    logits=model(torch.rand(1,1,64,64)); loss=logits.mean(); loss.backward()
    assert tuple(logits.shape)==(1,5,64,64)
