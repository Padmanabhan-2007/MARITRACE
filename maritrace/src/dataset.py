"""PyTorch dataset adapter; importable without torch for data preparation."""
from pathlib import Path
import numpy as np
from PIL import Image
class SynthSARDataset:
    def __init__(self,root,split='train',augment=False,crop_size=None):
        import torch
        self.torch=torch; self.files=sorted((Path(root)/split).glob('*.npy')); self.augment=augment; self.crop_size=crop_size
    def __len__(self): return len(self.files)
    def __getitem__(self,i):
        torch=self.torch; f=self.files[i]; x=np.load(f).astype('float32'); y=np.asarray(Image.open(f.with_suffix('.png'))).astype('int64')
        if x.ndim==2: x=x[None]
        if self.crop_size is not None:
            h,w=y.shape; ch=cw=self.crop_size
            if h<ch or w<cw: raise ValueError(f'crop_size {ch} exceeds tile dimensions {y.shape}')
            top=np.random.randint(0,h-ch+1); left=np.random.randint(0,w-cw+1); x=x[:,top:top+ch,left:left+cw]; y=y[top:top+ch,left:left+cw]
        if self.augment:
            if np.random.rand()<.5: x=x[:,:,::-1]; y=y[:,::-1]
            if np.random.rand()<.5: x=x[:,::-1,:]; y=y[::-1,:]
        return torch.from_numpy(np.ascontiguousarray(x)),torch.from_numpy(np.ascontiguousarray(y))
    def class_frequencies(self):
        counts=np.zeros(5,dtype=np.int64)
        for f in self.files: counts+=np.bincount(np.asarray(Image.open(f.with_suffix('.png'))).ravel(),minlength=5)
        return counts/counts.sum() if counts.sum() else counts
