# from omegaconf import DictConfig
# from omegaconf import OmegaConf
# import os
# from pathlib import Path


# print(__file__)
# print(os.path.abspath(__file__))
# print(Path(os.path.abspath(__file__)).parent)


# dict = OmegaConf.create({'Age':7})

# # dict = {'Name': 'Zabra', 'Age': 7}
# print(dict.get('Age'))
# print(dict.get('Education', None))


from tqdm import tqdm
from time import sleep
import logging

logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", 
                    datefmt="%m%d%Y %H:%M:%S",level=logging.INFO)

streamhandler = logging.StreamHandler()
logger = logging.getLogger('')
logger.addHandler(streamhandler)


bar = tqdm(range(5, 10), initial=5, total=10)# , initial=5,total=20, ncols=10)#, desc='Phase 1')#, total=300, smoothing=0)

bar.set_description('Phase 1')
for idx in bar:
    # print(idx)
    logger.info(idx)
    sleep(0.2)
    bar.set_postfix(refresh=True)


# test = dict()
# test['ads'] = 10
# test['dsad'] = 20
# bar = tqdm(range(13, 10), initial=13, total=10)
# for i in bar:
#     # print(i, end='\r', flush=True)
#     logger.info(i)
#     sleep(1)
#     print('dsad')
#     bar.set_postfix(refresh=True)