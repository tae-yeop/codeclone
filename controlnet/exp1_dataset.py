import json
import cv2
import numpy as np

from torch.utils.data import Dataset
import os
from pathlib import Path


class DFDataset(Dataset):
    def __init__(self):
        self.data = []
        with open('./training/fashion_mm.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        source = cv2.imread(os.path.join('./training',source_filename))
        target = cv2.imread(os.path.join('./training',target_filename))

        source = cv2.resize(source, (256, 256), interpolation=cv2.INTER_LANCZOS4)
        target = cv2.resize(target, (256, 256), interpolation=cv2.INTER_LANCZOS4)

        # Do not forget that OpenCV read images in BGR order.
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)

if __name__ == '__main__':
	# print('dsad')
	# with open('/purestorage/project/tyk/0_experiments/0_smp/ControlNet/training/captions.json', 'r') as f:
	# 	data = json.load(f)
	# root = '/purestorage/project/tyk/0_experiments/0_smp/ControlNet/training'
	# img_list = list(Path(root, 'images').rglob('*.jpg'))
	# print(img_list[:10])
	# seg_list = list(Path(root, 'seg').rglob('*.png'))
	# print(img_list[0].name)

	# import os
	# dir_path = '/purestorage/project/tyk/0_experiments/0_smp/ControlNet/training/images'
	# img_list = [f for f in os.listdir(dir_path) if f.endswith('.jpg')]

	# print(img_list[:10])
	# dir_path = '/purestorage/project/tyk/0_experiments/0_smp/ControlNet/training/segm'
	# seg_list = [f for f in os.listdir(dir_path) if f.endswith('.png')]
	# print(seg_list[:10])

	# print(len(img_list), len(seg_list))



    from PIL import Image

    print(Image.open('/purestorage/project/tyk/project0/0_smp/ControlNet/training/images/MEN-Denim-id_00000080-01_7_additional.jpg').size)

    print(Image.open('/purestorage/project/tyk/project0/0_smp/ControlNet/training/segm/MEN-Denim-id_00000080-01_7_additional_segm.png').size)
