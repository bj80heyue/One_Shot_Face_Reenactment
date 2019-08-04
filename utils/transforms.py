"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""

import math
import random
import torch as th
import cv2

from utils.affine_util import th_affine2d, exchange_landmarks
import numpy as np
#from utils.points2heatmap import getCurve17


def initAlignTransfer(size, mirror=False, gaze=True):
	corr_list_base = np.array([0, 32, 1, 31, 2, 30, 3, 29, 4, 28, 5, 27, 6, 26, 7, 25, 8, 24, 9, 23,	 10, 22, 11, 21, 12, 20, 13, 19, 14, 18, 15, 17, 33, 42, 34, 41, 35, 40, 36, 39, 37, 38, 64, 71, 65,	 70, 66, 69, 67,
							   68, 52, 61, 53, 60, 72, 75, 54, 59, 55, 58, 56, 63, 73, 76, 57, 62, 74, 77, 104, 105, 78, 79, 80, 81, 82, 83, 47, 51, 48, 50, 84, 90, 96, 100, 85, 89, 86, 88, 95, 91, 94, 92, 97, 99, 103, 101, ]).reshape([-1, 2])

	corr_list_gaze = 106 + np.array([0, 20, 1, 21, 2, 22, 3, 23, 4, 24, 5, 25, 6, 26, 7, 27, 8, 28, 9		, 29,
									 10, 30, 11, 31, 12, 32, 13, 33, 14, 34, 15, 35, 16, 36, 17, 37, 18, 38, 19, 39]).reshape(-1,	 2)
	if gaze:
		corr_list = np.vstack([corr_list_base, corr_list_gaze])
	else:
		corr_list = corr_list_base
	transform_align = AffineCompose(rotation_range=10,
									translation_range=10,
									zoom_range=[0.9, 1.1],
									fine_size=size,
									mirror=mirror,
									corr_list=corr_list
									)
	return transform_align

# input pt: numpy N*2
# input img: numpy, 3*256*256, 0-255


class AffineCompose(object):

	def __init__(self,
				 rotation_range,
				 translation_range,
				 zoom_range,
				 fine_size,
				 mirror=False,
				 corr_list=None
				 ):

		self.fine_size = fine_size
		self.rotation_range = rotation_range
		self.translation_range = translation_range
		self.zoom_range = zoom_range
		self.mirror = mirror
		self.corr_list = corr_list

	def __call__(self, *inputs):
		rotate = random.uniform(-self.rotation_range, self.rotation_range)
		trans_x = random.uniform(-self.translation_range,
								 self.translation_range)
		trans_y = random.uniform(-self.translation_range,
								 self.translation_range)
		if not isinstance(self.zoom_range, list) and not isinstance(self.zoom_range, tuple):
			raise ValueError('zoom_range must be tuple or list with 2 values')
		zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

		# rotate
		transform_matrix = np.zeros([3, 3])
		center = (self.fine_size/2.-0.5, self.fine_size/2-0.5)
		M = cv2.getRotationMatrix2D(center, rotate, 1)
		transform_matrix[:2, :] = M
		transform_matrix[2, :] = np.array([[0, 0, 1]])
		# translate
		transform_matrix[0, 2] += trans_x
		transform_matrix[1, 2] += trans_y
		# zoom
		for i in range(3):
			transform_matrix[0, i] *= zoom
			transform_matrix[1, i] *= zoom
		transform_matrix[0, 2] += (1.0 - zoom) * center[0]
		transform_matrix[1, 2] += (1.0 - zoom) * center[1]

		# mirror about x axis in cropped image
		do_mirror = False
		'''
		if self.mirror:
			mirror_rng = random.uniform(0., 1.)
			if mirror_rng > 0.5:
				do_mirror = True

		do_mirror = True
		if do_mirror:
			transform_matrix[0, 0] = -transform_matrix[0, 0]
			transform_matrix[0, 1] = -transform_matrix[0, 1]
			transform_matrix[0, 2] = float(
				self.fine_size)-transform_matrix[0, 2]
		'''

		outputs = []
		for idx, _input in enumerate(inputs):
			# input: pt, face_256
			if _input.ndim == 3:
				is_landmarks = False
			else:
				is_landmarks = True
			input_tf = th_affine2d(_input,
								   transform_matrix,
								   output_img_width=self.fine_size,
								   output_img_height=self.fine_size,
								   is_landmarks=is_landmarks)
			'''
			if do_mirror:
				if is_landmarks:
					input_tf = exchange_landmarks(input_tf, self.corr_list)
				else:
					#input_tf = cv2.flip(input_tf, 1)
					pass
			'''
			outputs.append(input_tf)
		return outputs if idx >= 1 else outputs[0]
	
def dealcurve(curve):
	cmean = curve.mean(0)
	angle = (random.random()*10)-5
	scale = ((random.random()-0.5)*0.1)+1.0
	m = cv2.getRotationMatrix2D((0,0),angle,scale)
	m = np.vstack([m,[0,0,1]])
	dmean = (np.random.rand(1,2)-0.5)*10
	curve = curve - cmean
	curve = cv2.perspectiveTransform(np.array([curve]),m)
	curve += cmean
	curve += dmean
	return curve[0]

'''
def shakeCurve(points):
	pts = points.copy()
	curves,_ = getCurve17(pts)
	Lbrow = np.vstack([curves[1],curves[13]])
	Rbrow = np.vstack([curves[2],curves[14]])
	Leye = np.vstack([curves[5],curves[6],curves[15]])
	Reye = np.vstack([curves[7],curves[8],curves[16]])
	Nose = np.vstack([curves[3],curves[4]])
	Mouth = np.vstack([curves[9],curves[10],curves[11],curves[12]])
	Bound = curves[0]
	
	Lbrow = dealcurve(Lbrow)
	Rbrow = dealcurve(Rbrow)
	Leye = dealcurve(Leye)
	Reye = dealcurve(Reye)
	Nose = dealcurve(Nose)
	Mouth= dealcurve(Mouth)
	Bound= dealcurve(Bound)

	for i in range(33):
		pts[i] = Bound[i]
	for i in range(5):
		pts[i+33] = Lbrow[i]
	for i in range(5):
		pts[i+38] = Rbrow[i]
	for i in range(4):
		pts[i+43] = Nose[i]
	pts[80] = Nose[4+0]
	pts[82] = Nose[4+1]
	for i in range(5):
		pts[i+47] = Nose[6+i]
	pts[83] = Nose[11]
	pts[81] = Nose[12]
	
	pts[52] = Leye[0]
	pts[53] = Leye[1]
	pts[72] = Leye[2]
	pts[54] = Leye[3]
	pts[55] = Leye[4]
	
	pts[55] = Leye[5]
	pts[56] = Leye[6]
	pts[73] = Leye[7]
	pts[57] = Leye[8]
	pts[52] = Leye[9]
	
	pts[58] = Reye[0]
	pts[59] = Reye[1]
	pts[75] = Reye[2]
	pts[60] = Reye[3]
	pts[61] = Reye[4]
	
	pts[61] = Reye[5]
	pts[62] = Reye[6]
	pts[76] = Reye[7]
	pts[63] = Reye[8]
	pts[58] = Reye[9]
	for i in range(7):
		pts[i+84] = Mouth[i]
	for i in range(5):
		pts[96+i] = Mouth[7+i]
	pts[96] = Mouth[12]
	pts[103] = Mouth[13]
	pts[102] = Mouth[14]
	pts[101] = Mouth[15]
	pts[100] = Mouth[16]
	
	pts[84] = Mouth[17]
	pts[95] = Mouth[18]
	pts[94] = Mouth[19]
	pts[93] = Mouth[20]
	pts[92] = Mouth[21]
	pts[91] = Mouth[22]

	pts[90] = Mouth[23]
	pts[33] = Lbrow[5+0]
	pts[64] = Lbrow[5+1]
	pts[65] = Lbrow[5+2]
	pts[66] = Lbrow[5+3]
	pts[67] = Lbrow[5+4]
	
	pts[68] = Rbrow[5+0]
	pts[69] = Rbrow[5+0]
	pts[70] = Rbrow[5+0]
	pts[71] = Rbrow[5+0]
	pts[42] = Rbrow[5+0]
	
	for i in range(20):
		pts[106+i] = Leye[10+i]
		pts[106+20+i] = Reye[10+i]
	return pts
'''

	
