from loader.dataset_basic import *
import random
import numpy as np
import copy
import torch as th
from utils.affineFace import affineface

class DatasetLoaderDemo(DatasetBasic):
	def __init__(self, imgSize=256, gaze=True):
		super(DatasetLoaderDemo, self).__init__(imgSize)
		self.boundList = None
		self.appearList = None
		self.rule = 'sequence'
		self.indexAppear = 0

	def loadBounds(self, pathList, head):
		self.boundList = self.loadtxtList(pathList, head)

	def loadAppears(self, pathList, head):
		self.appearList = self.loadtxtList(pathList, head)

	def setAppearRule(self, flag='random'):
		# flag[string]: random / similar / sequence
		# call this function after load data
		if self.appearList == None:
			print('please call setAppearRule after load data!')
		if flag != 'random' and flag != 'similar' and flag != 'sequence':
			print('rule: ', 'random / similar / sequence')
		else:
			self.rule = flag
			if flag == 'random':
				self.indexAppear = random.randint(0, len(self.appearList)-1)
			else:
				pass

	def findSimilar(self, pt_dst):
		minVal = 1e5
		res = 0
		for index in range(len(self.appearList)):
			_, pt = self.appearList[index]
			pt = (np.array(pt) - 64).reshape(-1, 2)
			diff = np.linalg.norm(pt[:106] - pt_dst[:106])
			if diff < minVal:
				res = index
				minVal = diff
		return res

	def adjustPose(self, img_src, pt_src, pt_dst):
		img_align, pt_align = affineface(img_src, pt_src, pt_dst)
		return img_align, pt_align

	def add_nose_bridge(self, boundary, heatmap):
		# add nose bridge boundary and dilate
		nose_bridge = copy.copy(boundary[3:4])
		kernel = np.ones((4, 4), np.uint8)
		nose_bridge = 255 * torch.from_numpy(cv2.dilate(nose_bridge.squeeze(0).numpy(), kernel)).unsqueeze(0).float()
		heatmap = torch.cat((heatmap, nose_bridge), 0)
		return heatmap

	def __getitem__(self, index):
		# load dst
		path, pt = self.boundList[index]
		img_dst = cv2.imread(path, 1)[64:64+256, 64:64+256]
		pt_dst = (np.array(pt) - 64).reshape(-1, 2)
		# dst
		heatmap_dst, curves_dst, boundary_dst = self.points2heatmap(pt_dst, self.imgSize, sigma=self.sigma)
		heatmap_dst = self.add_nose_bridge(boundary_dst, heatmap_dst)  # add nose bridge boundary and dilate
		weighted_mask_dst = heatmap_dst[0:1] + 2 * heatmap_dst[1:2] + 3 * heatmap_dst[2:3] + 4 * heatmap_dst[3:4] + 2 * heatmap_dst[4:5] + \
							3 * heatmap_dst[5:6] + 3 * heatmap_dst[6:7] + 2 * heatmap_dst[7:8]  +  heatmap_dst[8:]
		#select reference
		if self.rule == 'random':
			index = self.indexAppear
		elif self.rule == 'similar':
			index = self.findSimilar(pt_dst)
		elif self.rule == 'sequence':
			index = min(index, len(self.appearList)-1)

		# load src
		path, pt = self.appearList[index]
		img_src = cv2.imread(path, 1)[64:64+256, 64:64+256]
		img_src_np = img_src
		img_src = self.gammaTrans(img_src, 0.5)
		pt_src = (np.array(pt) - 64).reshape(-1, 2)
		pt_src_np = pt_src

		# align pose src 2 dst
		img_src,pt_src = self.adjustPose(img_src,pt_src/256.0,pt_dst/256.0)
		img_src = self.warp(img_src, pt_src, np.vstack([pt_dst[:33], pt_src[33:]]))

		# src
		heatmap_src, curves_src, boundary_src = self.points2heatmap(pt_src, self.imgSize, sigma=self.sigma)

		#np 2 tensor scale = [-1,1]
		img_src = self.np2tensor(img_src)
		img_dst = self.np2tensor(img_dst)

		return {'img_src': img_src, 'face_mask_src': heatmap_src[0:1],
				'img_dst': img_dst, 'face_mask_dst': heatmap_dst[0:1], 'seg_dst': heatmap_dst, 'weighted_mask_dst': weighted_mask_dst,
				'pt_src': pt_src_np, 'pt_dst': pt_dst, 'img_src_np': img_src_np}

	def __len__(self):
		return len(self.boundList)
