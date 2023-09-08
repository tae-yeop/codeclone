import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
import sys
import os

os.chdir('/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet')
sys.path.append('/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet')

resume_path = '/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet/exp1.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = False
only_mid_control = False


model = create_model('/home/aiteam/tykim/generative_model/framework/diffusion/ControlNet/models/cldm_v15_exp1.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


dataset = MyDataset()
dataloader = DataLoader(dataset, num_workers=2, batch_size=2, shuffle=False)

batch = next(iter(dataloader))
batch['jpg'] = batch['jpg'].to('cuda')
batch['hint'] = batch['hint'].to('cuda')

# from pytorch_lightning.utilities.model_summary import ModelSummary

# # ModelSummary(model, max_depth=-1)
model = model.to('cuda')
model.training_step(batch, 0)

# logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=32, callbacks=[logger])
# # Train!
# trainer.fit(model, dataloader)