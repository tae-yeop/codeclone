# import torch
# import torch.nn as nn
# from torch.utils.checkpoint import checkpoint

# from transformers import T5Tokenizer, T5EncoderModel, CLIPTokenizer, CLIPTextModel

# import open_clip
# from ldm.util import default, count_params


# CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet/models")
# CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14", cache_dir="/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet/models")

from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from exp1_dataset import DFDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from pytorch_lightning import seed_everything

seed_everything(1)


# Configs
resume_path = './models/control_sd15_exp3_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15_exp3.yaml').cpu()