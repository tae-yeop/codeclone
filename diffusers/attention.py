from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn



class Attention(nn.Module):
    def batch_to_head_dim(self, tensor):
        head_size = self.heads
        batch_size, seq_lm 


