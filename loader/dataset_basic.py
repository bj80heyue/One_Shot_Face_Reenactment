# coding:utf-8
import sys
from utils.points2heatmap import curves2segments,points2curves
from utils import warper
import os
import numpy as np
import torch
import torch.utils.data
import cv2
from tqdm import *


class DatasetBasic(torch.utils.data.Dataset):
	def __init__(self, imgSize=256):
		# imgSize[int]
		self.boundList = None
		self.appearList = None
		self.imgSize = imgSize
		self.sigma = 3

	def __len__(self):
		return -1

	def shape(self):
		return self.__len__()

	def loadtxt(self, path, head=''):
		# path[string] : path to lms.txt
		#		format:	subpath of img
		#				landmarks [106*2 + 20*2] or 40*2
		# head[string] : head of subpath
		res = list()
		with open(path, 'r') as fin:
			data = fin.read().splitlines()
			N = len(data)//2
			for i in tqdm(range(N)):
				imgPath = os.path.join(head, data[2*i+0])
				landmarks = list(map(float, data[2*i+1].split()))
				res.append((imgPath, landmarks))
		return res

	def loadtxtList(self, pathList, head):
		res = list()
		for path in pathList:
			res += self.loadtxt(path, head)
		return res

	def warp(self, img, srcPt, dstPt):
		# img[ndarray]: shape = (3,D,D)
		# srcPt[ndarray]: shape = (K,2) ,K key points
		# dstPt[ndarray]: shape = (K,2) ,K key points
		return warper.warping(img, srcPt.reshape(-1, 2), dstPt.reshape(-1, 2), (self.imgSize, self.imgSize))

	def np2tensor(self, img, scale=1/255.0):
		#========input=======
		#img[ndarray][H,W,C] (0.0,1/scale)
		#========output======
		#img[ndarray][C,H,W] (-1.0,1.0)
		img = img.transpose((2, 0, 1))
		img = torch.from_numpy(img).float() * scale
		img = (img - 0.5) / 0.5
		return img

	def points2heatmap(self, landmarks, mapSize, sigma, landmarkSize=255.0, heatmap_num=17):
		# landmarks[ndarray]	: shape = (K,2), K = 106 + 20*2
		# landmarkSize [float]
		# mapSize[int] 		: output size
		# sigma[float] 		: gaussian sigma
		# _________Return__________
		# heatmap[tensor]: [C,H,W]
		# curve[list] :	list[list]
		if landmarks.max() > 1:
			landmarks /= landmarkSize
			landmarks[landmarks > 1] = 1
		curves, boundary = points2curves(landmarks, mapSize, sigma, heatmap_num)
		# [C,H,W] (0.0,1.0)
		heatmap = curves2segments(curves)
		# np 2 tensor
		heatmap = torch.from_numpy(heatmap).float()

		# boundary heatmap
		boundary = boundary.transpose([2, 0, 1])
		boundary = torch.from_numpy(boundary).float()

		return heatmap, curves, boundary



	'''
	def getRois(self, curve, sigma, onlyMask=False):
		# curves[list[list]] :
		# sigma[float]		: gaussian sigma
		# onlyMask[bool] 	: for train ,just need mask of face
		bound = np.vstack([curve[1], curve[2], curve[0]])

		mask_bound = genROI(bound, D=5, sigma=5)
		if onlyMask:
			return None, mask_bound
		browL = np.vstack([curve[1], curve[13]])
		browR = np.vstack([curve[2], curve[14]])
		eyeL = np.vstack([curve[5], curve[6]])
		eyeR = np.vstack([curve[7], curve[8]])
		nose = np.vstack([curve[3], curve[4]])
		teeth = np.vstack([curve[10], curve[11]])
		mouth = np.vstack([curve[9], curve[12]])

		mask_browL = genROI(browL)
		mask_browR = genROI(browR)
		mask_eyeL = genROI(eyeL, )
		mask_eyeR = genROI(eyeR)
		mask_nose = genROI(nose)
		mask_mouth = genROI(mouth)
		mask_teeth = genROI(teeth)
		mask_skin = (1-mask_eyeL)*(1-mask_eyeR)*(1-mask_nose)*(1-mask_browL)*(1-mask_browR)\
			* (1-mask_teeth)*(1-mask_mouth)

		if len(curve) == 17:
			# gaze
			gazeL = curve[15]
			gazeR = curve[16]
			mask_gazeL = genROI(gazeL,erode=True)
			mask_gazeR = genROI(gazeR,erode=True)
			return {'browL': mask_browL, 'browR': mask_browR, 'eyeL': mask_eyeL, 'eyeR': mask_eyeR, 'nose': mask_nose,
					'mouth': mask_mouth, 'teeth': mask_teeth, 'skin': mask_skin, 'gazeL': mask_gazeL, 'gazeR': mask_gazeR}, mask_bound
		else:
			return {'browL': mask_browL, 'browR': mask_browR, 'eyeL': mask_eyeL, 'eyeR': mask_eyeR, 'nose': mask_nose,
					'mouth': mask_mouth, 'teeth': mask_teeth, 'skin': mask_skin, }, mask_bound

	def fix_gaze(self, eye_roi, gaze_roi):
		intersect = eye_roi * gaze_roi
		return intersect
	'''

	def gammaTrans(self, img, gamma):
		gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
		gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
		return cv2.LUT(img, gamma_table)

	def __getitem__(self, index):
		pass
