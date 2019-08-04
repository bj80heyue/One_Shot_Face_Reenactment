简介：
	此项目的意义是针对Face2Face的生成结果,利用reference进行纹理的融合.

算法输入：
	生成图像img_gen
	生成图像106点+40点眼睛(optional)

	参考图像img_ref
	参考图像106点+40点眼睛(optional)

算法输出:
	融合后的图像


算法选项：
	1.alpha blending
	2.泊松融合
	2.NCC mask net
