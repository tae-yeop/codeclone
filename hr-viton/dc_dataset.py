
import torch
import torch.utils.data as data
import torchvision.transforms as transforms


from PIL import Image, ImageDraw
import json
import os
import os.path as osp
import numpy as np


class DCDataset(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        
        self.fine_height = opt.fine_height
        self.fine_width = opt.fine_width
        # 총 채널 갯수
        self.semantic_nc = opt.semantic_nc
        self.transform = transforms.Compose([  \
                transforms.ToTensor(),   \
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        # sub_category = ['dresses', 'lower_body', 'upper_body']
        # sub_folder = ['images', 'dense', 'label_maps']
        # self.image_list = [osp.join(osp.join(self.root, c), d) for c in sub_category for d in sub_folder]
        # im_names = [f for f in os.listdir(self.image_list[0]) if f.endswith('.jpg')]
        
        
        subfolders = {
            # 지금 dense와 skeletons 번호가 5로 같음
            # cm : _1, sk : _5, pa : _7
            'upper_body': {'images', 'dense', 'label_maps', 'cloth_mask', 'skeletons', 'parse_agnostic'},
            'lower_body': {'images', 'dense', 'label_maps', 'mask'},
            'dresses': {'images', 'dense', 'label_maps', 'mask'},
        }
        
        self.path_dict = {f"{subfolder}_{nested_subfolder}": os.path.join(self.root, subfolder, nested_subfolder)
            for subfolder in subfolders
            for nested_subfolder in subfolders[subfolder]}
        # path_dict['lower_images'] = osp.join(self.root, 'lower_body', 'images')
        # path_dict['upper_images'] = osp.join(self.root, 'upper_body', 'images')
        # path_dict['dress_images']
        # self.image_path = osp.join()
        # im_names = []
        # c_names = []
        # for category in sub_category:
        #     im_names.extend([f for f in os.listdir(osp.joint(self.root, category, 'images')) if f.endswith('_0.jpg')])
        #     c_names.extend([f for f in os.listdir(osp.joint(self.root, category, 'images')) if f.endswith('_1.jpg')])
        
        
        # self.upper_i_names = [f for f in os.listdir(self.path_dict['upper_images']) if f.endswith('_0.jpg') for key in self.path_dict if key.endswith('images')]
        # self.upper_c_names = [f for f in os.listdir(self.path_dict['upper_images']) if f.endswith('_1.jpg')] 
        
        # self.i_names = [f for f in os.listdir(self.path_dict[key]) if f.endswith('_0.jpg') for key in self.path_dict if key.endswith('images')]

        self.i_names = [osp.join(self.path_dict[key], f) for key in self.path_dict if key.endswith('images') for f in os.listdir(self.path_dict[key]) if f.endswith('_0.jpg')]
        self.c_names = [osp.join(self.path_dict[key], f) for key in self.path_dict if key.endswith('images') for f in os.listdir(self.path_dict[key]) if f.endswith('_1.jpg')]
        self.cm_names = [osp.join(self.path_dict[key], f) for key in self.path_dict if key.endswith('mask') for f in os.listdir(self.path_dict[key]) if f.endswith('_1.jpg')]
        self.label_maps = [osp.join(self.path_dict[key], f) for key in self.path_dict if key.endswith('label_maps') for f in os.listdir(self.path_dict[key]) if f.endswith('_4.png')]
        
        
    def __len__(self):
        return len(self.im_names)
        
    def __getitem__(self, index):
        
        i_name = self.i_names[index]
        c_name = self.c_names[index]
        cm_name = self.cm_names[index]
        label_maps = self.label_maps[index]

        # cloth
        c = Image.open(c_name).convert('RGB')
        c = transforms.Resize(self.fine_width, interpolation=2)(c)
        c = self.transform(c) # [-1, 1]
        
        # cloth_mask
        cm = Image.open(cm_name)
        cm = transforms.Resize(self.fine_width, interpolation=0)(cm)
        cm_array = np.array(cm)
        cm_array = (cm_array >= 128).astype(np.float32)
        cm = torch.from_numpy(cm_array)  # [0,1]
        cm.unsqueeze_(0)
        
        
        # person image
        im_pil_big = Image.open(i_name).convert('RGB')
        im_pil = transforms.Resize(self.fine_width, interpolation=2)(im_pil_big)
        im = self.transform(im_pil)


        # load parsing image
        im_parse_pil_big = Image.open(label_maps)
        im_parse_pil = transforms.Resize(self.fine_width, interpolation=0)(im_parse_pil_big)
        parse = torch.from_numpy(np.array(im_parse_pil)[None]).long()
        im_parse = self.transform(im_parse_pil.convert('RGB'))
        
        # Dreecode parse map 
        labels = {
            0:  ['background',  [0]],
            1:  ['hair',        [1, 2]], # hat, hair
            2:  ['face',        [3, 11]], # sunglasses + head
            3:  ['upper',       [4]], # upper_clothes
            4:  ['bottom',      [5, 6]], # skirt + pants
            5:  ['left_arm',    [14]],
            6:  ['right_arm',   [15]],
            7:  ['left_leg',    [12]],
            8:  ['right_leg',   [13]],
            9:  ['left_shoe',   [9]],
            10: ['right_shoe',  [10]],
            11: ['belt',       [8]], # belt
            12: ['bag',        [16]],
            13: ['dress' ,     [7]],
            14: ['scarf',      [17]]
        }
        
        parse_map = torch.FloatTensor(18, self.fine_height, self.fine_width).zero_()
        parse_map = parse_map.scatter_(0, parse, 1.0)
        new_parse_map = torch.FloatTensor(self.semantic_nc, self.fine_height, self.fine_width).zero_()

        for i in range(len(labels)):
            for label in labels[i][1]: # 번호
                new_parse_map[i] += parse_map[label]
                
        parse_onehot = torch.FloatTensor(1, self.fine_height, self.fine_width).zero_()
        for i in range(len(labels)):
            for label in labels[i][1]:
                parse_onehot[0] += parse_map[label] * i


        result = {
            #'c_name':   c_name,     # for visualization
            #'im_name':  im_name,    # for visualization or ground truth
            # intput 1 (clothfloww)
            'cloth':    c,          # for input
            'cloth_mask':     cm,   # for input
            # intput 2 (segnet)
            'parse_agnostic': new_parse_agnostic_map,
            'densepose': densepose_map,
            'pose': pose_map,       # for conditioning
            # GT
            'parse_onehot' : parse_onehot,  # Cross Entropy
            'parse': new_parse_map, # GAN Loss real
            'pcm': pcm,             # L1 Loss & vis
            'parse_cloth': im_c,    # VGG Loss & vis
            # visualization & GT
            'image':    im,         # for visualization
            }

        return result


from torch.utils.data import DistributedSampler

class DCDataLoader(object):
    def __init__(self, opt, dataset):
        super().__init__()

        if opt.ddp:
            train_sampler = DistributedSampler(dataset, drop_last=True)
        else:
            train_sampler = None

        self._dataloader = data.DataLoader(dataset, batch_size=opt.batch_size, 
                                           num_workers=opt.workers, shuffle=(train_sampler is None), pin_memory=True,  sampler=train_sampler)
        self.dataset = dataset

    # 
    def next_batch(self):

        