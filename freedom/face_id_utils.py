import torch
import numpy as np

import dlib
from skimage import transform as trans
from skimage import io


def get_points_and_rec(img, detector, shape_predictor, size_threshold=999):
    dets = detector(img, 1)
    if len(dets) == 0:
        return None, None
    
    all_points = []
    rec_list = []
    for det in dets:
        if isinstance(detector, dlib.cnn_face_detection_model_v1):
            rec = det.rect # for cnn detector
        else:
            rec = det
        if rec.width() > size_threshold or rec.height() > size_threshold: 
            break
        rec_list.append(rec)
        shape = shape_predictor(img, rec) 
        single_points = []
        for i in range(5):
            single_points.append([shape.part(i).x, shape.part(i).y])
        all_points.append(np.array(single_points))
    if len(all_points) <= 0:
        return None, None
    else:
        return all_points, rec_list
    

def align_and_save(img, src_points, template_path, template_scale=1, img_size=256):
    out_size = (img_size, img_size)
    reference = np.load(template_path) / template_scale * (img_size / 256)

    for idx, spoint in enumerate(src_points):
        tform = trans.SimilarityTransform()
        tform.estimate(spoint, reference)
        M = tform.params[0:2,:]
        return M, img.shape

        

def align_and_save_dir(src_path, template_path='./pretrain_models/FFHQ_template.npy', template_scale=4, use_cnn_detector=True, img_size=256):
    if use_cnn_detector:
        detector = dlib.cnn_face_detection_model_v1('./pretrain_models/mmod_human_face_detector.dat')
    else:
        detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor('./pretrain_models/shape_predictor_5_face_landmarks.dat')

    img_path = src_path
    img = dlib.load_rgb_image(img_path)

    points, rec_list = get_points_and_rec(img, detector, sp)
    if points is not None:
        return align_and_save(img, points, template_path, template_scale, img_size=img_size)


    

def get_tensor_M(src_path):
    M, s = align_and_save_dir(src_path)
    h, w = s[0], s[1]
    a = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    Mt = torch.Tensor(
        [  
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]
    )
    Mt[:2, :] = torch.Tensor(M)
    Mt = torch.inverse(Mt)
    h, w = 256, 256
    b = torch.Tensor(
        [
            [2/(w-1), 0, -1],
            [0, 2/(h-1), -1],
            [0, 0, 1]
        ]
    )
    b = torch.inverse(b)
    Mt = a.matmul(Mt)
    Mt = Mt.matmul(b)[:2].unsqueeze(0)
    return Mt