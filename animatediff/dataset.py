import decord
decord.bridge.set_bridge('torch')

from torch.utils.data import Dataset
from einops import rearrange

# 이건 하나의 영상에 대해서 수행하는 것임
class TuneAVideoDataset(Dataset):
    def __init__(self, 
                 ):
        

        self.prompt = prompt
        self.prompt_ids = None

        self.width = width
        
    def __len__(self):
        return 1
    
    def __getitem__(self, index):
        vr = decord.VideoReader(self.video_path, width=self.width, height=self.height)
        sample_index = list(range(self.sample_start_idx, len(vr), self.sample_frame_rate))[]