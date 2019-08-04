from fusion.points2heatmap import *
from fusion.calcAffine import *
from fusion.warper import warping as warp
import matplotlib.pyplot as plt
from fusion.parts2lms import parts2lms
import time
from tqdm import *
import random
import multiprocessing
import sys


def gammaTrans(img, gamma):
	gamma_table = [np.power(x/255.0, gamma)*255.0 for x in range(256)]
	gamma_table = np.round(np.array(gamma_table)).astype(np.uint8)
	return cv2.LUT(img, gamma_table)

def erodeAndBlur(img,kernelSize=21,blurSize=21):
	#img : ndarray float32
	kernel = np.ones((int(kernelSize), int(kernelSize)), np.uint8)
	res = cv2.erode(img,kernel)
	res = cv2.GaussianBlur(res, (blurSize, blurSize), math.sqrt(blurSize))
	return res

def affineface(img,src_pt,dst_pt,heatmapSize=256,needImg=True):
	#src/dst_pt[ndarray] : [...,[x,y],...] in [0.0,1.0],with gaze
	#naive mode: align 5 parts 
	curves_src,_ = points2curves(src_pt.copy())
	pts_fivesense_src = np.vstack(curves_src[1:])
	curves_dst,_ = points2curves(dst_pt.copy())
	pts_fivesense_dst = np.vstack(curves_dst[1:])
	affine_mat = calAffine(pts_fivesense_src,pts_fivesense_dst)

	pt_aligned = affinePts(affine_mat,src_pt*255.0)/255.0
	if needImg:
		img_aligned = affineImg(img,affine_mat)
		return pt_aligned,img_aligned
	else:
		return pt_aligned

def affineface_parts(img,src_pt,dst_pt):
	curves_src,_ = points2curves(src_pt.copy())
	curves_dst,_ = points2curves(dst_pt.copy())#[0,255]

	parts_src = curves2parts(curves_src)
	parts_dst = curves2parts(curves_dst)	#[0,255]

	partsList = []
	for i in range(len(parts_src)-2):
		affine_mat = calAffine(parts_src[i],parts_dst[i])
		parts_aligned = affinePts(affine_mat,parts_src[i])	#[0,255]
		partsList.append(parts_aligned)
	partsList.append(parts_src[-2])
	partsList.append(parts_src[-1])
	
	'''
	A = []
	B = []
	for i in range(len(parts_src)):
		A.append(parts_src[i])
		B.append(partsList[i])
	A = np.vstack(A)
	B = np.vstack(B)
	res = warp(img,A,B)
	'''
	lms = parts2lms(partsList)
	#bound
	lms[:33] = dst_pt[:33]*256
	res = warp(img,src_pt[:106]*256,lms[:106])

	return lms/255.0,res

def lightEye(img_ref,lms_ref,img_gen,lms_gen,ratio=0.1):
	#get curves
	curves_ref,_ = points2curves(lms_ref.copy())
	curves_gen,_ = points2curves(lms_gen.copy())

	parts_ref = curves2parts(curves_ref)
	parts_gen = curves2parts(curves_gen)	#[0,255]

	#get rois
	gaze_ref = curves2gaze(curves_ref)
	gaze_gen = curves2gaze(curves_gen)

	#img_gazeL = np.dot(gaze_ref[0],  img_ref)
	img_gazeL = multi(img_ref,gaze_ref[0])
	#img_gazeR = np.dot(gaze_ref[1] , img_ref)
	img_gazeR = multi(img_ref,gaze_ref[1])

	affine_mat = calAffine(parts_ref[-2],parts_gen[-2])
	img_gazeL_affined = affineImg(img_gazeL,affine_mat)
	affine_mat = calAffine(parts_ref[-1],parts_gen[-1])
	img_gazeR_affined = affineImg(img_gazeR,affine_mat)

	img_ref = img_gazeL_affined + img_gazeR_affined
	
	mask = gaze_gen[0] + gaze_gen[1]
	mask = erodeAndBlur(mask,5,5)

	R = img_gen[:,:,0] * (1-mask) + mask* (img_gen[:,:,0]*ratio + img_ref[:,:,0]*(1-ratio))
	G = img_gen[:,:,1] * (1-mask) + mask* (img_gen[:,:,1]*ratio + img_ref[:,:,1]*(1-ratio))
	B = img_gen[:,:,2] * (1-mask) + mask* (img_gen[:,:,2]*ratio + img_ref[:,:,2]*(1-ratio))

	res = np.stack([R,G,B]).transpose((1,2,0))
	seg = mask
	seg = seg * 127
	return res,seg,img_ref

def multi(img,mask):
	R = img[:,:,0] * mask
	G = img[:,:,1] * mask
	B = img[:,:,2] * mask
	res = np.stack([R,G,B]).transpose((1,2,0))
	return res


def fusion(img_ref,lms_ref,img_gen,lms_gen,ratio=0.2):
	#img*: ndarray(np.uint8) [0,255]
	#lms*: ndarray , [...,[x,y],...] in [0,1]
	#ratio: weight of gen 
	#--------------------------------------------
	#get curves
	curves_ref,_ = points2curves(lms_ref.copy())
	curves_gen,_ = points2curves(lms_gen.copy())
	#get rois
	roi_ref = curves2segments(curves_ref)
	roi_gen = curves2segments(curves_gen)
	#get seg
	seg_ref = roi_ref.sum(0)
	seg_gen = roi_gen.sum(0)
	seg_ref = seg_ref / seg_ref.max() * 255
	seg_gen = seg_gen / seg_gen.max() * 255
	#get skin mask
	skin_src = roi_ref[0] - roi_ref[2:].max(0)
	skin_gen = roi_gen[0] - roi_gen[2:].max(0)
	#blur edge
	skin_src = erodeAndBlur(skin_src,7,7)
	skin_gen = erodeAndBlur(skin_gen,7,7)
	#fusion 
	skin = skin_src * skin_gen

	R = img_gen[:,:,0] * (1-skin) + skin * (img_gen[:,:,0]*ratio + img_ref[:,:,0]*(1-ratio))
	G = img_gen[:,:,1] * (1-skin) + skin * (img_gen[:,:,1]*ratio + img_ref[:,:,1]*(1-ratio))
	B = img_gen[:,:,2] * (1-skin) + skin * (img_gen[:,:,2]*ratio + img_ref[:,:,2]*(1-ratio))

	res = np.stack([R,G,B]).transpose((1,2,0))
	return res,seg_ref,seg_gen


def loaddata(head,path_lms,flag=256,num = 50000):
	#head: head of img
	#return res:[[path,lms[0,1]]]
	fin = open(path_lms,'r')
	data = fin.read().splitlines()
	res = []
	for i in tqdm(range(min(len(data)//2,num))):
		name = data[2*i]
		path = os.path.join(head,name)
		lms = list(map(float,data[2*i+1].split()))
		if flag==256:
			lms = np.array(lms).reshape(-1,2) / 255.0
		else:
			lms = (np.array(lms).reshape(-1,2)-64) / 255.0
		res.append((path,lms))
	return res

def gray2rgb(img):
	res = np.stack([img,img,img]).transpose((1,2,0))
	return res.astype(np.uint8)

def process(index, album_ref, album_gen, album_pose):
	# 30ms
	img_gen = cv2.imread(album_gen[index][0])
	lms_gen = album_gen[index][1]
	img_ref = cv2.imread(album_ref[index // 100][0])[64:64 + 256, 64:64 + 256, :]
	lms_ref = album_ref[index // 100][1]
	img_pose = cv2.imread(album_pose[index % 100][0])[64:64 + 256, 64:64 + 256, :]
	lms_pose = album_pose[index % 100][1]

	# affine
	# 4ms
	lms_ref_, img_ref_ = affineface(img_ref, lms_ref, lms_gen)
	# 200ms
	lms_ref_parts, img_ref_parts = affineface_parts(img_ref, lms_ref, lms_gen)

	# fusion
	# fuse_all,seg_ref_,seg_gen = fusion(img_ref_,lms_ref_,img_gen,lms_gen,0.1)
	fuse_parts, seg_ref_parts, seg_gen = fusion(img_ref_parts, lms_ref_parts, img_gen, lms_gen, 0.1)
	fuse_eye, mask_eye, img_eye = lightEye(img_ref, lms_ref, fuse_parts, lms_gen, 0.1)

	res = np.hstack([img_ref, img_pose, img_gen, fuse_eye])
	cv2.imwrite('proposed_wild/fuse/%d.jpg' % (index), fuse_eye)

	


