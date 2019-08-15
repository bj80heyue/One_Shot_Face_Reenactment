# One-shot Face Reenactment

[[Project]](https://wywu.github.io/projects/ReenactGAN/OneShotReenact.html) [[Paper]](https://arxiv.org/abs/1908.03251) [[Demo]](https://www.youtube.com/watch?v=FE-D6wh11_A)  

Official test script for 2019 BMVC spotlight paper 'One-shot Face Reenactment' in PyTorch.

<img src="https://github.com/bj80heyue/Learning_One_Shot_Face_Reenactment/blob/master/pics/main.png" width = 900 align=middle>

## Installation

### Requirements
- Linux
- Python 3.6
- PyTorch 0.4+
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
- imgs : store images
- lms : store landmarks extracted from images
	- format : 106 common facial key points & 20+20 gaze key points
	
<div align="center"><img src="https://github.com/bj80heyue/Learning_One_Shot_Face_Reenactment/blob/master/pics/lms.png" width = 500></div>

Example input data is organized in folder 'data'. Please organize your data in the format the same as the example input data if you want to test with your own data. 

Output images are saved in folder 'output'.

Due to the protocol of company, the model to extract 106 + 40 facial landmarks cannot be released, however, if you want to get access to the following dataset, please fill in the license file in the repo (license/celebHQlms_license.pdf), then email the signed copy to siwei.1995@163.com to get access to the annotation dataset. 
- our preprocessed 106 + 40 facial landmark annotations of celebHQ dataset
- additional 80 images as pose guide with corresponding 106 + 40 facial landmark annotations


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
You can download models from [here](https://drive.google.com/open?id=1Wnc2TGwFQM4PdCdeSn-trI75UeGbuY_E) 
```shell
Project
├── pretrainModel
│   ├── id_200.pth
│   ├── vgg16-397923af.pth
├── trained_model
│   ├── latest_net_appEnc.pth
│   ├── latest_net_appDnc.pth
│   ├── latest_net_netG.pth
│   ├── latest_net_netD64.pth
│   ├── latest_net_netD128.pth
│   ├── latest_net_netD256.pth
```

### Visualization of results
You can download our sample data and corresponding results from [here](https://drive.google.com/open?id=1Ia8YJrtYTvNRwBfcKK7iBSAf5vb8gkqw)

## License and Citation
The use of this software follows **MIT License**.
```
@inproceedings{OneShotFace2019,
  title={One-shot Face Reenactment},
  author={Zhang, Yunxuan and Zhang, Siwei and He, Yue and Li, Cheng and Loy, Chen Change and Liu, Ziwei},
  booktitle={British Machine Vision Conference (BMVC)},
  year={2019}
}
```
