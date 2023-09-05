import torch
import torch.nn as nn
from .facial_recognition.model_irse import Backbone
import torchvision

class IDLoss(nn.Module):
    def __init__(self, weight_path):
        super(IDLoss, self).__init__()
#         print('Loading ResNet ArcFace for ID Loss')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(weight_path))
        self.pool = torch.nn.AdaptiveAvgPool2d((256, 256))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()

        self.to_tensor = torchvision.transforms.ToTensor()

    def extract_feats(self, x):
        if x.shape[2] != 256:
            x = self.pool(x)
        x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats
    
    def extract_feats_not_align(self, x):
        # if x.shape[2] != 256:
        #     x = self.pool(x)
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        x = self.face_pool(x)
        x_feats = self.facenet(x)
        return x_feats

    def get_residual(self, image, ref):
        img_feat = self.extract_feats(image)
        ref_feat = self.extract_feats(ref)
        return ref_feat - img_feat