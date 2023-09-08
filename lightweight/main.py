import argparse
import os
import torch
from torch.utils.data import DataLoader
import wandb


import pytorch_lightning as pl
import pytorch_lightning.loggers as pl_loggers
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import seed_everything

import core.coach as coach
from core.dataset import PairedDataset
from core.utils import create_model, create_coach, load_state_dict

seed_everything(1)

wandb_key = ''
wandb_host = ''


parser = argparse.ArgumentParser()
parser.add_argument("--gpus", type=int, default=1, help="number of available GPUs")
parser.add_argument("--checkpoint_dir", type=str, default=None, help="path to checkpoint_dir")
parser.add_argument("--max_steps", type=int, default=25000, help="training step")
parser.add_argument("--model_config", type=str, default=None, help="training model configuration path")
parser.add_argument("--batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--num_nodes", type=int, default=2, help="num nodes")    
args = parser.parse_args()


# Model
coach_model = create_coach(args.model_config).cpu()
# model.load_state_dict(load_state_dict())
# DataLoader
dataset = PairedDataset()
dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)

wandb.login(key=wandb_key, host=wandb_host, force=True,)
wandb_logger = WandbLogger(name='test_1', project='ai_service_model', log_model=True)

trainer = pl.Trainer(accelerator="gpu", devices=args.gpus, precision=32, 
                     num_nodes=args.num_nodes, strategy='ddp',
                     logger=wandb_logger, max_steps=args.max_steps) # callbacks=[logger])

trainer.fit(coach_model, dataloader)