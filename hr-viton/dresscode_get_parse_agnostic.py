import json
from os import path as osp
import os
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
from pathlib import Path
from PIL import Image

import typer


# for item in dataroot.iterdir():
# 	print(item.suffix)
# 	print(item)
# 	print(item.name)
# def get_image():
# 	im_parse = Image.open('/home/aiteam/tykim/generative_model/human/HR-VITON/data/dresscode/label_maps/020714_4.png')


# Dresscode는 얼굴 부분이 없어서 키포인트가 적게 나옴 : 24개
# 그래도 사용할 수 있을까?

def handle_hand(key_point_path, agnostic, label_array):
    with open(key_point_path, 'r') as f:
        pose_label = json.load(f)
        pose_data = pose_label['keypoints']
        print(pose_data)
        pose_data = np.array(pose_data)
        pose_data = pose_data[:,0:2]
        pose_data = pose_data*2
        
    w=768
    h=1024
    r = 10

    for parse_id, pose_ids in [(14, [2, 5, 6, 7]), (15, [5, 2, 3, 4])]:
        # 14번에서 [2,5,6,7]
        mask_arm = Image.new('L', (w, h), 'black')
        mask_arm_draw = ImageDraw.Draw(mask_arm)
        i_prev = pose_ids[0] # i_prev 처음엔 2
        for i in pose_ids[1:]:
            # i_prev가 keypoint를 나타내는 듯
            # 키포인트 값이 없으면 무시
            if (pose_data[i_prev, 0] == 0.0 and pose_data[i_prev, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
                continue

            # key point가 있는 곳 두 점을 이어서 라인을 만듬
            mask_arm_draw.line([tuple(pose_data[j]) for j in [i_prev, i]], 'white', width=r*10) # [2, 5]
            pointx, pointy = pose_data[i]
            radius = r*4 if i == pose_ids[-1] else r*15
            mask_arm_draw.ellipse((pointx-radius, pointy-radius, pointx+radius, pointy+radius), 'white', 'white')
            i_prev = i
        parse_arm = (np.array(mask_arm) / 255) * (label_array == parse_id).astype(np.float32)
        agnostic.paste(0, None, Image.fromarray(np.uint8(parse_arm * 255), 'L'))
        

    return agnostic


def get_agnostic(src_path):

    src_path = Path(src_path)
    # dest_path = Path(dest_path)
    
    print(src_path.name)
    labels_path = os.path.join(src_path, 'label_maps')
    keypoints_path = os.path.join(src_path, 'keypoints')
    
    dest_path = Path(os.path.join(src_path, 'parse_agnostic'))
    dest_path.mkdir(parents=True, exist_ok=True)
    label_map_list = list(Path(labels_path).rglob('*.png'))
    print(label_map_list)

    for label_path in label_map_list:
        print(label_path)
        im_parse = Image.open(label_path)
        label_array = np.array(im_parse)
        
        key_path = os.path.join(keypoints_path, label_path.name.replace('_4.png', '_2.json'))
        print(key_path)
        
        agnostic = im_parse.copy()
        # garment category specific
        if src_path.name == 'dresses':
            exclusion = ((label_array == 7).astype(np.float32) + (label_array == 8).astype(np.float32))
        elif src_path.name == 'lower_body':
            exclusion = ((label_array == 5).astype(np.float32) + (label_array == 6).astype(np.float32))
        elif src_path.name == 'upper_body':
            exclusion = ((label_array == 4).astype(np.float32) + (label_array == 7).astype(np.float32))
        else:
            pass
        
        agnostic.paste(0, None, Image.fromarray(np.uint8(exclusion * 255), 'L'))
    
        agnostic = handle_hand(key_path, agnostic, label_array)
        
        agnostic.save(dest_path.joinpath(label_path.name.replace('_4.png', '_7.png')))
        
if __name__ == '__main__':
	typer.run(get_agnostic)