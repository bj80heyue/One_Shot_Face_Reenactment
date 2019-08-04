import numpy as np

def parts2lms(parts):
	bound,browL,browR,eyeL,eyeR,nose,lipU,lipD,gazeL,gazeR = parts
	res = list()
	res.append(bound)	#0-32
	res.append(browL[:5]) #33- 37
	res.append(browR[:5])	#38-42
	res.append(nose[:4])	#43,44,45,46
	res.append(nose[6:6+5])	#47,48,49,50,51
	res.append(eyeL[:2])	#52,53
	res.append(eyeL[3:3+2])	#54,55
	res.append(eyeL[6])		#56
	res.append(eyeL[8])		#57
	res.append(eyeR[:2])	#58,59
	res.append(eyeR[3:3+2])	#60,61
	res.append(eyeR[6])		#62
	res.append(eyeR[8])		#63
	res.append(browL[6:6+4])#64,65,66,67
	res.append(browR[5:5+4])#68,69,70,71
	res.append(eyeL[2])	#72
	res.append(eyeL[7])	#73
	res.append((eyeL[2]+eyeL[7])/2)	#74 useless
	res.append(eyeR[2])	#75
	res.append(eyeR[7])	#76
	res.append((eyeR[2]+eyeR[7])/2)	#77 useless
	res.append((nose[0]+eyeL[4])/2)	#78
	res.append((nose[0]+eyeR[0])/2)	#79
	res.append(nose[4])	#80
	res.append(nose[12])	#81
	res.append(nose[5])	#82
	res.append(nose[11])	#83
	res.append(lipU[:7])	#84,85,86,87,88,89,90
	res.append(lipD[10])	#91
	res.append(lipD[9])	#92
	res.append(lipD[8])	#93
	res.append(lipD[7])	#94
	res.append(lipD[6])	#95
	res.append(lipU[7:7+5]) #96,97,98,99,100
	res.append(lipD[3])	#101
	res.append(lipD[2])	#102
	res.append(lipD[1])	#103
	res.append((eyeL[2]+eyeL[7])/2)	#104
	res.append((eyeR[2]+eyeR[7])/2)	#105
	res.append(gazeL)
	res.append(gazeR)
	res = np.vstack(res)
	return res



