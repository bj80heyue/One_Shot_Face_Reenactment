from loader.dataset_basic import *
from utils.transforms import initAlignTransfer
#from utils.transforms import shakeCurve
import random
import numpy as np
import torch as th
import copy

class DatasetLoaderTrain(DatasetBasic):
	def __init__(self, imgSize=256, gaze=True):
		super(DatasetLoaderTrain, self).__init__(imgSize)
		self.transformAlign = initAlignTransfer(self.imgSize,mirror=False, gaze=gaze)
		self.dataList = None
		self.isTransform = True
		self.SampleCurveType = 'Bound'
		if gaze:
			self.heatmap_num = 17
		else:
			self.heatmap_num = 15
	
	def setSampleCurve(self, flag='Bound'):
		self.SampleCurveType = flag

	def transform(self, img, pt):
		pack = self.transformAlign(pt, img)
		pt = pack[0]
		img = pack[1]
		return img, pt

	def loaddata(self, pathList, head):
		# pathList[list] : [path1,path2,...]
		# head[string]:	head of subpath
		self.dataList = self.loadtxtList(pathList, head)

	def sampleCurve(self, img, pt, flag='Bound'):
		# if flag == 'None':
		# 	return img_src.copy(), pt_src.copy()
		if flag == 'Bound':
			_, pt_sample = random.sample(self.dataList, 1)[0]
			pt_sample = (np.array(pt_sample) - 64).reshape(-1, 2)
			pt_ = pt.copy()
			pt_[:33] = pt_sample[:33]
			img_warped = self.warp(img, pt, pt_)
			return img_warped, pt_
		# if flag == 'Shake':
		# 	pt_ = shakeCurve(pt)
		# 	img_warped = self.warp(img, pt, pt_)
		# 	return img_warped, pt_
		return None

	def add_nose_bridge(self, boundary, heatmap):
		# add nose bridge boundary and dilate
		nose_bridge = copy.copy(boundary[3:4])
		kernel = np.ones((4, 4), np.uint8)
		nose_bridge = 255 * torch.from_numpy(cv2.dilate(nose_bridge.squeeze(0).numpy(), kernel)).unsqueeze(0).float()  # todo
		heatmap = torch.cat((heatmap, nose_bridge), 0)
		return heatmap

	def __getitem__(self, index):
		path, pt = self.dataList[index]
		img_src = cv2.imread(path, 1)[64:64+256, 64:64+256] # [256, 256, 3]
		pt_src = (np.array(pt) - 64).reshape(-1, 2) # [146,2]
		img_src = self.gammaTrans(img_src, 0.5)
		# sample strategy
		img_dst, pt_dst = self.sampleCurve(img_src, pt_src, flag=self.SampleCurveType)
		# img_dst = copy.copy(img_src)
		# pt_dst = copy.copy(pt_src)
		# data augmentation
		if self.isTransform:
			img_dst, pt_dst = self.transform(img_dst, pt_dst)
		# src
		heatmap_src, curves_src, boundary_src = self.points2heatmap(pt_src, self.imgSize, sigma=self.sigma) # heatmap_src: tensor [8, 256, 256], curve_src: list of 17, each eliemnt: an array
		heatmap_src = self.add_nose_bridge(boundary_src, heatmap_src)
		weighted_mask_src = heatmap_src[0:1] + 2 * heatmap_src[1:2] + 3 * heatmap_src[2:3] + 4 * heatmap_src[3:4] + 2 * heatmap_src[4:5] + \
							3 * heatmap_src[5:6] + 3 * heatmap_src[6:7] + 2 * heatmap_src[7:8] + heatmap_src[8:]
		# dst
		heatmap_dst, curves_dst, boundary_dst = self.points2heatmap(pt_dst, self.imgSize, sigma=self.sigma)
		heatmap_dst = self.add_nose_bridge(boundary_dst, heatmap_dst)  # add nose bridge boundary and dilate
		weighted_mask_dst = heatmap_dst[0:1] + 2 * heatmap_dst[1:2] + 3 * heatmap_dst[2:3] + 4 * heatmap_dst[3:4] + 2 * heatmap_dst[4:5] + \
							3 * heatmap_dst[5:6] + 3 * heatmap_dst[6:7] + 2 * heatmap_dst[7:8] + heatmap_dst[8:]

		img_src = self.np2tensor(img_src)
		img_dst = self.np2tensor(img_dst)
		return {'img_src':img_src,'seg_src':heatmap_src,'face_mask_src':heatmap_src[0:1], 'boundary_dst': boundary_dst,
				'img_dst':img_dst,'seg_dst':heatmap_dst,'face_mask_dst':heatmap_dst[0:1], 'weighted_mask_dst': weighted_mask_dst}

	def __len__(self):
		return len(self.dataList)
