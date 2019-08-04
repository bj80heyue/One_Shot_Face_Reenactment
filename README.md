# Learning One-shot Face Reenactment
Official test script for 2019 BMVC paper 'Learning One-shot Face Reenactment' in PyTorch.

<img src="https://github.com/bj80heyue/Learning_One_Shot_Face_Reenactment/blob/master/pics/main.png" width = 900 align=middle>

## Installation

### Requirements
- Linux
- Python 3.6
- PyTorch 1.0+
- CUDA 9.0+
- GCC 4.9+

### Easy Install
```shell
pip install -r requirements.txt
```

## Getting Started

### Prepare Data
It is recommended to symlink the dataset root to `$PROJECT/data`.
```shell
Project
├── data
│   ├── poseGuide
│   │   ├── imgs
│   │   ├── lms
│   ├── reference
│   │   ├── imgs
│   │   ├── lms
```
- imgs : store images that match lms
- lms : store landmarks extracted from images
	- format : 106 common facial key points & 20+20 gaze key points
	
	![image](https://github.com/bj80heyue/Learning_One_Shot_Face_Reenactment/blob/master/pics/lms.png)

Example input data is organized in folder 'data'. Please organize your data in the format the same as the example input data if you want to test with your own data. 

Output images are saved in folder 'output'.

### Inference with pretrained model
```
python test.py --pose_path PATH/TO/POSE/GUIDE/IMG/DIR --ref_path PATH/TO/REF/IMG/DIR --pose_lms PATH/TO/POSE/LANDMARK/FILE --ref_lms PATH/TO/REF/LANDMARK/FILE
```

```
output sequence: 
		ref1-pose1, ref1-pose2,  ref1-pose3, ... &
		ref2-pose1, ref2-pose2,  ref2-pose3, ... &
		ref3-pose1, ref3-pose2,  ref3-pose3, ... &
		    .				
		    .				
		    .					
```

### Pretrained model
You can download the model from [here](https://drive.google.com/open?id=1Wnc2TGwFQM4PdCdeSn-trI75UeGbuY_E) 

### Visualization of results
You can download our sample data and corresponding results from [here](https://drive.google.com/open?id=1Ia8YJrtYTvNRwBfcKK7iBSAf5vb8gkqw)





 
