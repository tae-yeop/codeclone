import os
import numpy as np
import zipfile # 
import PIL.Image
import json
import torch
import dnnlib
from training.utils import get_poseangle

try:
    import pyspng
except ImportError:
    pyspng = None


class MaskLabeledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 img_path,  # Path to directory or zip.
                 seg_path,  # Path to directory or zip.
                 min_yaw=None,  # Minimum yaw angle
                 max_yaw=None,  # Maximum yaw angle
                 max_pitch=None,  # Maximum pitch angle
                 back_repeat=None,  # repeat back images how many times
                 name=None,
                 raw_shape=None, # Shape of the raw image data (NCHW). [총 이미지 갯수, C, H, W]
                 max_size=None,  # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
                 use_labels=False,  # Enable conditioning labels? False = label dimension is zero.
                 xflip=False,  # Artificially double the size of the dataset via x-flips. Applied after max_size.
                 random_seed=0  # Random seed to use when applying max_size.
                 ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self._img_path = img_path
        self._seg_path = seg_path
        self.min_yaw = 0 if min_yaw is None else min_yaw
        self.max_yaw = 180 if max_yaw is None else max_yaw
        self.max_pitch = 90 if max_pitch is None else max_pitch
        self.back_repeat = 1 if back_repeat is None else back_repeat

        if os.path

        # 내부적으로 segmentation dataset을 포함
        self._seg_dataset = Image

        
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        self._raw_idx = self._filter_samples()
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _filter_samples(self):
        raise NotImplementedError

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def _load_raw_image(self, raw_idx):  # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self):  # to be overridden by subclass
        raise NotImplementedError

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        # 먼저 image path에서 얻음
        image = self._load_raw_image(self._raw_idx[idx])
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3  # CHW
            image = image[:, :, ::-1]
        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3  # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3  # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64


class MaskLabeledDataset(torch.utils.data.Dataset):
    def __init__(self,
                 xflip=False, # mirror horizontal flip을 getitem할 때 수행
                 ):
        
    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx])
        mask = self._seg_dataset._load_raw_image(self._seg_raw_idx[idx])
        label = self.get_label(idx)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            # 이걸 수행하려면 image가 numpy array여야 한다
            # torch tensor로는 ::-1이 되지 않ㅇ므
            # 역방향으로 width를 조회해서 flip 되는 효과를 낸다
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
            mask = mask[:, :, ::-1]
            if self._use_labels:
                assert label.shape == (25,)
                label[[1,2,3,4,8]] *= -1
        return image.copy(), mask.copy(), label