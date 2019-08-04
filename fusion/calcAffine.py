# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 13:43:03 2017
"""
import numpy as np
import cv2


#affine points via least square method
#src_p[input] -- np.array([[x,y],...])
#dst_p[input] -- list[float]
#affine_mat[output] -- np.array() | matrix of affine 
#pt_align[output] -- np.array() | aligned points
def calAffine(src_p, dst_p):
	p_N = len(src_p)
	U = np.mat(list(dst_p[:,0]) + list(dst_p[:,1]))
	xx_src,yy_src = list(src_p[:,0]),list(src_p[:,1])

	X = np.mat(np.stack([xx_src + yy_src, yy_src + [-ii for ii in xx_src], \
	[1 for ii in range(p_N)] + [0 for ii in range(p_N)], \
	[0 for ii in range(p_N)] + [1 for ii in range(p_N)]], axis=1))

	result = np.linalg.pinv(X) * U.T

	affine_mat = np.zeros([2, 3])
	affine_mat[0][0] = result[0][0]
	affine_mat[0][1] = result[1][0]
	affine_mat[0][2] = result[2][0]
	affine_mat[1][0] = -result[1][0]
	affine_mat[1][1] = result[0][0]
	affine_mat[1][2] = result[3][0]
	return affine_mat

def affinePts(affine_mat,pt):
	src_align = pt.T
	new_align = np.mat(affine_mat[:2, :2]) * np.mat(src_align) + np.reshape(affine_mat[:, 2], (-1, 1))
	pt_align = np.array(np.reshape(new_align.T, -1))[0].reshape(-1,2)
	return pt_align
	
#affine Image from pt_src to pt_mean
#img[input] -- np.array()
#pt_src,pt_mean[input] -- list[float] format = x0,y0,x1,y1,...,xn,yn
#img_align -- np.array() | aligned image
def affineImg(img,TransMat,dsize = 256):
	img_align = cv2.warpAffine(img, TransMat, (dsize, dsize), borderValue=(155, 155, 155) )
	return img_align

	
if __name__ == '__main__':

	path_src = '/media/heyue/8d1c3fac-68d3-4428-af91-bc478fbdd541/Project/Face2Face/detectface/samples/common/output/landmarks.txt'
	output_pt = 'lms/lms.txt' 
	output_img = 'imgs'
	affineList(path_src, output_pt,output_img,'meanpose384.txt',k=2,head='/media/heyue/8d1c3fac-68d3-4428-af91-bc478fbdd541/Project/Face2Face/Data/test')
	
	'''
	path_src = 'alignedPoints_256.txt'
	output_pt = 'output/AU_points.txt'
	output_img = 'output/AU'
	head = '/media/heyue/8d1c3fac-68d3-4428-af91-bc478fbdd541/Project/Face2Face/net/GANimation/dataset_emo'
	affineList(path_src,output_pt,output_img,'meanpose384.txt',k=2,head = head)
	print('done')
	'''
