import numpy as np
import cv2
import os
import math

def curve_interp(points, heatmapSize=256, sigma=3):
	sigma = max(1,(sigma // 2)*2 + 1)
	img = np.zeros((heatmapSize, heatmapSize), np.uint8)
	for ii in range(1, points.shape[0]):
		cv2.line(img, tuple(points[ii-1].astype(np.int32)),tuple(points[ii].astype(np.int32)), (255), sigma)
	img = cv2.GaussianBlur(img, (sigma, sigma), sigma)
	return img.astype(np.float64)/255.0

def curve_fill(points, heatmapSize=256, sigma=3, erode=False):
	sigma = max(1,(sigma // 2)*2 + 1)
	points = points.astype(np.int32)
	canvas = np.zeros([heatmapSize, heatmapSize])
	cv2.fillPoly(canvas,np.array([points]),255)
	'''
	kernel = np.ones((sigma, sigma), np.uint8)
	if erode:
		erode_kernel = np.ones((int(0.5*sigma), int(0.5*sigma)), np.uint8)
		canvas = cv2.erode(canvas, erode_kernel)
	else:
		canvas = cv2.dilate(canvas, kernel)
	'''
	canvas = cv2.GaussianBlur(canvas, (sigma,sigma), sigma)
	return canvas.astype(np.float64)/255.0

def curves2heatmap(curves,heatmapSize=256,sigma=3,flag='line'):
	#-----------------------input--------------------------
	# curves [list of ndarray] : points coordinate in [0,heatmapSize]
	# heatmapSize[int]: the size of the generated heatmap 
	# sigma[float]: Boundary vagueness
	# flag[string]: 'line' or 'segment'
	#-----------------------output----------------
	# heatmap[ndarray,float64]: [D,D,num of curves],range in (0.0,1.0)
	#=============================================
	heatmap = np.zeros((heatmapSize, heatmapSize, len(curves)),np.float64)
	for i in range(len(curves)):
		if flag == 'line':
			heatmap[:, :, i] = curve_interp(curves[i], heatmapSize, sigma)
		else:
			heatmap[:, :, i] = curve_fill(curves[i], heatmapSize, sigma)
	return heatmap

def curves2segments(curves,heatmapSize=256,sigma=3):
	#res[ndarray]: range in (0,1) [Channel,Size,Size]
	face = curve_fill(np.vstack([curves[0],curves[2][::-1],curves[1][::-1]]),heatmapSize,sigma)
	browL = curve_fill(np.vstack([curves[1],curves[13][::-1]]),heatmapSize,sigma)
	browR = curve_fill(np.vstack([curves[2],curves[14][::-1]]),heatmapSize,sigma)
	eyeL = curve_fill(np.vstack([curves[5],curves[6]]),heatmapSize,sigma)
	eyeR = curve_fill(np.vstack([curves[7],curves[8]]),heatmapSize,sigma)
	gazeL = curve_fill(curves[15],heatmapSize,sigma)
	gazeR = curve_fill(curves[16],heatmapSize,sigma)

	#intersect eye and gaze
	gazeL = gazeL * eyeL
	gazeR = gazeR * eyeR
	#2 to 1
	eye = np.max([eyeL,eyeR],axis=0)
	gaze = np.max([gazeL,gazeR],axis=0)
	brow = np.max([browL,browR],axis=0)

	nose = curve_fill(np.vstack([curves[3][0:1],curves[4]]),heatmapSize,sigma)
	lipU= curve_fill(np.vstack([curves[9],curves[10][::-1]]),heatmapSize,sigma)
	lipD= curve_fill(np.vstack([curves[11],curves[12][::-1]]),heatmapSize,sigma)
	tooth = curve_fill(np.vstack([curves[10],curves[11][::-1]]),heatmapSize,sigma)
	return np.stack([face,brow,eye,gaze,nose,lipU,lipD,tooth])

def curves2gaze(curves,heatmapSize=256,sigma=3):
	eyeL = curve_fill(np.vstack([curves[5],curves[6]]),heatmapSize,sigma)
	eyeR = curve_fill(np.vstack([curves[7],curves[8]]),heatmapSize,sigma)
	gazeL = curve_fill(curves[15],heatmapSize,sigma)
	gazeR = curve_fill(curves[16],heatmapSize,sigma)
	#intersect eye and gaze
	gazeL = gazeL * eyeL
	gazeR = gazeR * eyeR
	return np.stack([gazeL,gazeR])
	
def curves2parts(curves):
	bound = curves[0]
	browL = np.vstack([curves[1],curves[13]])
	browR = np.vstack([curves[2],curves[14]])
	eyeL = np.vstack([curves[5],curves[6]])
	eyeR = np.vstack([curves[7],curves[8]])
	gazeL = curves[15]
	gazeR = curves[16]
	nose = np.vstack([curves[3],curves[4]])
	lipU= np.vstack([curves[9],curves[10]])
	lipD= np.vstack([curves[11],curves[12]])
	return [bound,browL,browR,eyeL,eyeR,nose,lipU,lipD,gazeL,gazeR]
	


def points2curves(points, heatmapSize=256,  sigma=1, heatmap_num=17):
	#-----------------------input--------------------------
	# points[ndarray]: [...,[x,y],...],range in (0.0,1.0)
	# heatmapSize[int]: the size of the generated heatmap 
	# heatmapNum: number of heatmap channels
	#-----------------------output----------------
	# curves [list of ndarray] : points coordinate in [0,heatmapSize]
	# =====================================================
	# resize points (0-1) to heatmapSize(0-D)
	for i in range(points.shape[0]):
		points[i] *= (float(heatmapSize))
	# curve define
	curves = [0]*heatmap_num
	curves[0] = np.zeros((33, 2))  # contour
	curves[1] = np.zeros((5, 2))  # left top eyebrow
	curves[2] = np.zeros((5, 2))  # right top eyebrow
	curves[3] = np.zeros((4, 2))  # nose bridge
	curves[4] = np.zeros((9, 2))  # nose tip
	curves[5] = np.zeros((5, 2))  # left top eye
	curves[6] = np.zeros((5, 2))  # left bottom eye
	curves[7] = np.zeros((5, 2))  # right top eye
	curves[8] = np.zeros((5, 2))  # right bottom eye
	curves[9] = np.zeros((7, 2))  # up up lip
	curves[10] = np.zeros((5, 2))  # up bottom lip
	curves[11] = np.zeros((5, 2))  # bottom up lip
	curves[12] = np.zeros((7, 2))  # bottom bottom lip
	curves[13] = np.zeros((5, 2))  # left bottom eyebrow
	curves[14] = np.zeros((5, 2))  # left bottom eyebrow
	if heatmap_num == 17:
		curves[15] = np.zeros((20, 2))  # left gaze
		curves[16] = np.zeros((20, 2))  # right gaze
	# assignment proccess
	# countour
	for i in range(33):
		curves[0][i] = points[i]
	for i in range(5):
		# left top eyebrow
		curves[1][i] = points[i+33]
		# right top eyebrow
		curves[2][i] = points[i+38]
	# nose bridge
	for i in range(4):
		curves[3][i] = points[i+43]
	# nose tip
	curves[4][0] = points[80]
	curves[4][1] = points[82]
	for i in range(5):
		curves[4][i+2] = points[i+47]
	curves[4][7] = points[83]
	curves[4][8] = points[81]
	# left top eye
	curves[5][0] = points[52]
	curves[5][1] = points[53]
	curves[5][2] = points[72]
	curves[5][3] = points[54]
	curves[5][4] = points[55]
	# left bottom eye
	curves[6][0] = points[55]
	curves[6][1] = points[56]
	curves[6][2] = points[73]
	curves[6][3] = points[57]
	curves[6][4] = points[52]
	# right top eye
	curves[7][0] = points[58]
	curves[7][1] = points[59]
	curves[7][2] = points[75]
	curves[7][3] = points[60]
	curves[7][4] = points[61]
	# right bottom eye
	curves[8][0] = points[61]
	curves[8][1] = points[62]
	curves[8][2] = points[76]
	curves[8][3] = points[63]
	curves[8][4] = points[58]
	# up up lip
	for i in range(7):
		curves[9][i] = points[i+84]
	# up bottom lip
	for i in range(5):
		curves[10][i] = points[i+96]
	# bottom up lip
	curves[11][0] = points[96]
	curves[11][1] = points[103]
	curves[11][2] = points[102]
	curves[11][3] = points[101]
	curves[11][4] = points[100]
	# bottom bottom lip
	curves[12][0] = points[84]
	curves[12][1] = points[95]
	curves[12][2] = points[94]
	curves[12][3] = points[93]
	curves[12][4] = points[92]
	curves[12][5] = points[91]
	curves[12][6] = points[90]
	# left bottom eyebrow
	curves[13][0] = points[33]
	curves[13][1] = points[64]
	curves[13][2] = points[65]
	curves[13][3] = points[66]
	curves[13][4] = points[67]
	# right bottom eyebrow
	curves[14][0] = points[68]
	curves[14][1] = points[69]
	curves[14][2] = points[70]
	curves[14][3] = points[71]
	curves[14][4] = points[42]
	if heatmap_num == 17:
		# left gaze
		for i in range(20):
			curves[15][i] = points[106+i]
		# right gaze
		for i in range(20):
			curves[16][i] = points[106+20+i]

	return curves,None

def distance(p1, p2):
	return math.sqrt((p1[0]-p2[0])*(p1[0]-p2[0])+(p1[1]-p2[1])*(p1[1]-p2[1]))

def curve_fitting(points, heatmap_size, sigma):
	curve_tmp = curve_interp(points, heatmap_size, sigma)
	return curve_tmp


if __name__ == '__main__':
	import matplotlib.pyplot as plt
	res = list()
	path = '../2019CVPR_reconstruct/data/celebHQ/lms.txt'
	head = '../2019CVPR_reconstruct/data/celebHQ/align_384'
	with open(path, 'r') as fin:
		data = fin.read().splitlines()
		N = len(data)//2
		for i in range(N):
			imgPath = os.path.join(head, data[2*i+0])
			landmarks = list(map(float, data[2*i+1].split()))
			res.append((imgPath, landmarks))
	for path,landmark in res:
		points = (np.array(landmark).reshape(-1,2).astype(np.float32)-64)/256.0
		curves = points2curves(points)
		segments = curves2segments(curves)
		img= np.sum(segments,axis=0)

		plt.figure()
		plt.imshow(img)
		for i in range(len(points)):
			plt.plot(points[i][0],(points[i][1]),'.',255,1)
			if i<=106:
				plt.text(points[i][0], (points[i][1]), str(i), fontsize=5)
			else:
				plt.text(points[i][0], (points[i][1]), str(i), fontsize=3)
		plt.show()
	
