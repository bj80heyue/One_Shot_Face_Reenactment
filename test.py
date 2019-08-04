import time
import scipy.misc as m
import numpy as np
import cv2
import torch
import torchvision.utils as vutils
import argparse
from tqdm import *
from model.spade_model import SpadeModel
from opt.configTrain import TrainOptions
from loader.dataset_loader_demo import DatasetLoaderDemo
from fusion.affineFace import *


parser = argparse.ArgumentParser()
parser.add_argument('--pose_path', type=str, default='data/poseGuide/imgs', help='path to pose guide images')
parser.add_argument('--ref_path', type=str, default='data/reference/imgs', help='path to appearance/reference images')
parser.add_argument('--pose_lms', type=str, default='data/poseGuide/lms_poseGuide.out', help='path to pose guide landmark file')
parser.add_argument('--ref_lms', type=str, default='data/reference/lms_ref.out', help='path to reference landmark file')
args = parser.parse_args()


if __name__ == '__main__':
    trainConfig = TrainOptions()
    opt = trainConfig.get_config()  # namespace of arguments
    # init test dataset
    dataset = DatasetLoaderDemo(gaze=(opt.input_nc == 9), imgSize=256)

    root = args.pose_path  # root to pose guide img
    path_Appears = args.pose_lms.format(root)  # root to pose guide dir&landmark
    dataset.loadBounds([path_Appears], head='{}/'.format(root))

    root = args.ref_path  # root to reference img
    path_Appears = args.ref_lms.format(root)   # root to reference dir&landmark
    dataset.loadAppears([path_Appears], '{}/'.format(root))
    dataset.setAppearRule('sequence')

    # dataloader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=opt.batch_size,
                                              shuffle=False,
                                              num_workers=12, drop_last=False)
    print('dataset size: {}\n'.format(dataset.shape()))

    # output sequence: ref1-pose1, ref1-pose2,  ref1-pose3, ... ref2-pose1, ref2-pose2, ref2-pose3, ...
    boundNew = []
    appNew = []
    for aa in dataset.appearList:
        for bb in dataset.boundList:
            boundNew.append(bb)
            appNew.append(aa)
    dataset.boundList = boundNew
    dataset.appearList = appNew

    model = SpadeModel(opt)  # define model
    model.setup(opt)  # initilize schedules (if isTrain), load pretrained models
    model.set_logger(opt) # set writer to runs/test_res
    model.eval()

    iter_start_time = time.time()
    cnt = 1
    with torch.no_grad():
        for step, data in tqdm(enumerate(data_loader)):
            model.set_input(data)  # set device for data
            model.forward()
            # fusionNet
            for i in range(data['img_src'].shape[0]):
                img_gen = model.fake_B.cpu().numpy()[i].transpose(1, 2, 0)
                img_gen = (img_gen * 0.5 + 0.5) * 255.0
                img_gen = img_gen.astype(np.uint8)
                img_gen = dataset.gammaTrans(img_gen, 2.0) # model output image, 256*256*3
                # cv2.imwrite('output_noFusion/{}.jpg'.format(cnt), img_gen)

                lms_gen = data['pt_dst'].cpu().numpy()[i] / 255.0 # [146, 2]
                img_ref = data['img_src_np'].cpu().numpy()[i]
                lms_ref = data['pt_src'].cpu().numpy()[i] / 255.0
                lms_ref_parts, img_ref_parts = affineface_parts(img_ref, lms_ref, lms_gen)

                # fusion
                fuse_parts, seg_ref_parts, seg_gen = fusion(img_ref_parts, lms_ref_parts, img_gen, lms_gen, 0.1)
                fuse_eye, mask_eye, img_eye = lightEye(img_ref, lms_ref, fuse_parts, lms_gen, 0.1)
                # res = np.hstack([img_ref, img_pose, img_gen, fuse_eye])
                cv2.imwrite('output/{}.jpg'.format(cnt), fuse_eye)
                cnt += 1
    iter_end_time = time.time()

    print('length of dataset:', len(dataset))
    print('time per img: ', (iter_end_time - iter_start_time) / len(dataset))




