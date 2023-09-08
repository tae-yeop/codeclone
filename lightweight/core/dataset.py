import os
from pathlib import Path

import cv2
import PIL.Image

import numpy as np
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

# 일단 jpg이라서 고려x
try:
    import pyspng
except ImportError:
    pyspng = None


######### Transformation ##############
class CenterCropMargin(object):
    def __init__(self, fraction=0.95):
        super().__init__()
        self.fraction=fraction
        
    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size)*self.fraction)

    def __repr__(self):
        return self.__class__.__name__

    

class PairedDataset(Dataset):
    
    def __init__(self, root, src_name='source', cond_name='target', tgt_name='condition',
                 crop=False, resize=None,
                 ):
        super().__init__()
        self.root = Path(root)
        self.src_path = self.root.joinpath(src_name)
        self.tgt_path = self.root.joinpath(tgt_name)
        self.cond_path = self.root.joinpath(cond_name)

        PIL.Image.init()
        self.src_list = list(self.src_path.rglob('*[!_r].png'))


        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.trsf_list = []

        if crop:
            self.crop = CenterCropMargin(fraction=0.95)
            self.trsf_list.append(self.crop)
            
    def __len__(self):
        return len()

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        with open(os.path.join(self._path, fname), 'rb') as f:
            if pyspng is not None and self._file_ext(fname) == '.png':
            image = pyspng.load(f.read())
        else:
            image = np.array(PIL.Image.open(f))
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image


    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)