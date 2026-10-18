"""Write three compact synthetic demonstration scenes."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .synth_sar import generate_sample
from .simulate import scenario
from .detect import mask_to_geojson
from .config import SAMPLE_DIR

def main():
    SAMPLE_DIR.mkdir(parents=True,exist_ok=True)
    for seed in (42,43,44):
        folder=SAMPLE_DIR/f'scene-{seed-41}'; folder.mkdir(exist_ok=True)
        image,mask=generate_sample(seed=seed,forced_spill=(128,128))
        np.save(folder/'image.npy',image[0]); Image.fromarray((image[0]*255).astype('uint8')).save(folder/'image.png'); Image.fromarray(mask).save(folder/'mask.png')
        s=scenario(seed); s['ais'].to_csv(folder/'ais.csv',index=False)
        (folder/'polygons.geojson').write_text(json.dumps(mask_to_geojson(mask),indent=2))
        (folder/'metadata.json').write_text(json.dumps({'simulated':True,'seed':seed,'image_timestamp':s['timestamp'],'spill_lat':s['lat'],'spill_lon':s['lon']},indent=2))
    # Grid of image and class-color mask examples for visual inspection.
    from PIL import ImageDraw
    canvas=Image.new('RGB',(768,768),'white'); colors=np.array([[20,45,70],[230,50,30],[240,190,40],[240,240,230],[90,140,80]],dtype='uint8')
    for i in range(12):
        im,ma=generate_sample(seed=800+i); x=(i%3)*256; y=(i//3)*192
        canvas.paste(Image.fromarray((im[0]*255).astype('uint8')).convert('RGB').resize((128,128)),(x,y)); canvas.paste(Image.fromarray(colors[ma]).resize((128,128)),(x+128,y))
    Path('runs').mkdir(exist_ok=True); canvas.save('runs/sample_grid.png')
if __name__=='__main__': main()
