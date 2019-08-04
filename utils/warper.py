import numpy as np
import scipy.spatial as spatial
from builtins import range
import cv2
from matplotlib import pyplot as plt


def warping(img, src_bound, dst_bound, size=(256, 256)):
    d = 254
    bound = np.array([0, 0, 0, d, d, 0, d, d]).reshape(-1, 2)
    src_bound = np.vstack([src_bound, bound]).astype(np.int32)
    dst_bound = np.vstack([dst_bound, bound]).astype(np.int32)
    src_bound[src_bound > d] = d
    dst_bound[dst_bound > d] = d
    src_bound = src_bound.astype(np.int32)
    dst_bound = dst_bound.astype(np.int32)
    res = warp_image(img, src_bound, dst_bound, size)
    return res


def bilinear_interpolate(img, coords):
    """ Interpolates over every image channel
    http://en.wikipedia.org/wiki/Bilinear_interpolation

    :param img: max 3 channel image
    :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
    :returns: array of interpolated pixels with same shape as coords
    """
    int_coords = np.int32(coords)
    x0, y0 = int_coords
    dx, dy = coords - int_coords

    # 4 Neighour pixels
    q11 = img[y0, x0]
    q21 = img[y0, x0+1]
    q12 = img[y0+1, x0]
    q22 = img[y0+1, x0+1]

    btm = q21.T * dx + q11.T * (1 - dx)
    top = q22.T * dx + q12.T * (1 - dx)
    inter_pixel = top * dy + btm * (1 - dy)

    return inter_pixel.T


def grid_coordinates(points):
    """ x,y grid coordinates within the ROI of supplied points

    :param points: points to generate grid coordinates
    :returns: array of (x, y) coordinates
    """
    xmin = np.min(points[:, 0])
    xmax = np.max(points[:, 0]) + 1
    ymin = np.min(points[:, 1])
    ymax = np.max(points[:, 1]) + 1
    return np.asarray([(x, y) for y in range(ymin, ymax)
                       for x in range(xmin, xmax)], np.uint32)


def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
    """
    Warp each triangle from the src_image only within the
    ROI of the destination image (points in dst_points).
    """
    roi_coords = grid_coordinates(dst_points)
    # indices to vertices. -1 if pixel is not in any triangle
    roi_tri_indices = delaunay.find_simplex(roi_coords)

    for simplex_index in range(len(delaunay.simplices)):
        coords = roi_coords[roi_tri_indices == simplex_index]
        num_coords = len(coords)
        out_coords = np.dot(tri_affines[simplex_index],
                            np.vstack((coords.T, np.ones(num_coords))))
        x, y = coords.T
        result_img[y, x] = bilinear_interpolate(src_img, out_coords)

    return None


def triangular_affine_matrices(vertices, src_points, dest_points):
    """
    Calculate the affine transformation matrix for each
    triangle (x,y) vertex from dest_points to src_points

    :param vertices: array of triplet indices to corners of triangle
    :param src_points: array of [x, y] points to landmarks for source image
    :param dest_points: array of [x, y] points to landmarks for destination image
    :returns: 2 x 3 affine matrix transformation for a triangle
    """
    ones = [1, 1, 1]
    for tri_indices in vertices:
        src_tri = np.vstack((src_points[tri_indices, :].T, ones))
        dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
        mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
        yield mat


def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
    # Resultant image will not have an alpha channel
    num_chans = 3
    src_img = src_img[:, :, :3]

    rows, cols = dest_shape[:2]
    result_img = np.zeros((rows, cols, num_chans), dtype)

    delaunay = spatial.Delaunay(dest_points)
    tri_affines = np.asarray(list(triangular_affine_matrices(
        delaunay.simplices, src_points, dest_points)))

    process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

    return result_img


if __name__ == "__main__":
    pass
