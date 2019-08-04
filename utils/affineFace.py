from utils.points2heatmap import *
from utils.calcAffine import *


def affineface(img, src_pt, dst_pt, heatmapSize=256):
    # naive mode
    curves_src, _ = points2curves(src_pt)
    pts_fivesense_src = np.vstack(curves_src[1:])
    curves_dst, _ = points2curves(dst_pt)
    pts_fivesense_dst = np.vstack(curves_dst[1:])
    affine_mat = calAffine(pts_fivesense_src, pts_fivesense_dst)

    pt_aligned = affinePts(affine_mat, src_pt)
    img_aligned = affineImg(img, affine_mat)
    return img_aligned, pt_aligned


if __name__ == '__main__':
    pass

