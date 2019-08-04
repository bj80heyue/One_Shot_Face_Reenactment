from __future__ import print_function
import torch
import numpy as np
import inspect
import re
import numpy as np
import os
import collections

import cv2

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array


def th_affine2d(x, matrix, output_img_width, output_img_height, center=True, is_landmarks=False):
    """
    2D Affine image transform on torch.Tensor

    """
    assert(matrix.ndim == 2)
    matrix = matrix[:2, :]
    transform_matrix = matrix
    src = x
    if is_landmarks:
        dst = np.empty((x.shape[0], 2), dtype=np.float32)
        for i in range(src.shape[0]):
            dst[i, :] = AffinePoint(np.expand_dims(
                src[i, :], axis=0), transform_matrix)

    else:
        # cols, rows, channels = src.shape
        dst = cv2.warpAffine(src, transform_matrix, (output_img_width, output_img_height),
                             cv2.INTER_AREA, cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
        # for gray image
        if dst.ndim == 2:
            dst = np.expand_dims(np.asarray(dst), axis=2)

    return dst


def AffinePoint(point, affine_mat):
    """
    Affine 2d point
    """
    assert(affine_mat.shape[0] == 2)
    assert(affine_mat.shape[1] == 3)
    assert(point.shape[1] == 2)

    point_x = point[0, 0]
    point_y = point[0, 1]
    result = np.empty((1, 2), dtype=np.float32)
    result[0, 0] = affine_mat[0, 0] * point_x + \
        affine_mat[0, 1] * point_y + \
        affine_mat[0, 2]
    result[0, 1] = affine_mat[1, 0] * point_x + \
        affine_mat[1, 1] * point_y + \
        affine_mat[1, 2]

    return result


def exchange_landmarks(input_tf, corr_list):
    """
    Exchange value of pair of landmarks
    """
    #print(corr_list.shape)
    for i in range(corr_list.shape[0]):
        temp = input_tf[corr_list[i][0], :].copy()
        input_tf[corr_list[i][0], :] = input_tf[corr_list[i][1], :]
        input_tf[corr_list[i][1], :] = temp

    return input_tf
