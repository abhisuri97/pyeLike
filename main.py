from __future__ import division
import cv2
import numpy as np
import math
from scipy.misc import toimage
import time
from queue import *

# test a possible center (x,y) are coors of possible
# center and gx and gy are the x and y components
# of the gradient of some other point x.
K_WEIGHT_DIVISOR = 1.0
K_FAST_WIDTH = 50
K_GRADIENT_THRESHOLD = 50.0
K_WEIGHT_BLUR_SIZE = 5
K_THRESHOLD_VALUE = 0.60
K_ENABLE_WEIGHT = True
K_POST_PROCESSING = True

# Helpers
def unscale_point(p, orig):
    px, py = p
    height, width = orig.shape
    ratio = K_FAST_WIDTH/width
    x = int(round(px / ratio))
    y = int(round(py / ratio))
    return (x,y)


def scale_to_fast_size(src):
    rows, cols = src.shape
    return cv2.resize(src, (K_FAST_WIDTH, int((K_FAST_WIDTH / cols) * rows)))


def test_possible_centers_formula(x, y, weight, gx, gy, arr):
    rows, cols = np.shape(arr)
    for cy in range(rows):
        for cx in range(cols):
            if x == cx and y == cy:
                continue
            dx = x - cx
            dy = y - cy

            magnitude = math.sqrt((dx * dx) + (dy * dy))
            dx = dx / magnitude
            dy = dy / magnitude
            dot_product = dx * gx + dy * gy
            dot_product = max(0.0, dot_product)
            if K_ENABLE_WEIGHT == True:
                arr[cy][cx] += dot_product * dot_product * (weight[cy][cx]/K_WEIGHT_DIVISOR)
            else:
                arr[cy][cx] += dot_product * dot_product
    return arr


def matrix_magnitude(mat_x, mat_y):
    rows, cols = np.shape(mat_x)
    res_arr = np.zeros((rows, cols))
    for y in range(rows):
        for x in range(cols):
            gX = mat_x[y][x]
            gY = mat_y[y][x]
            magnitude = math.sqrt((gX * gX) + (gY * gY))
            res_arr[y][x] = magnitude
    return res_arr


def compute_dynamic_threshold(mags_mat, std_dev_factor):
    mean_magn_grad, std_magn_grad = cv2.meanStdDev(mags_mat)
    rows, cols = np.shape(mags_mat)
    stddev = std_magn_grad[0] / math.sqrt(rows * cols)
    return std_dev_factor * stddev + mean_magn_grad[0]


def flood_should_push_point(dir, mat):
    px, py = dir
    rows, cols = np.shape(mat)
    if px >= 0 and px < cols and py >= 0 and py < rows:
        return True
    else:
        return False


def flood_kill_edges(mat):
    rows, cols = np.shape(mat)
    cv2.rectangle(mat, (0,0), (cols, rows), 255)
    mask = np.ones((rows, cols), dtype=np.uint8)
    mask = mask * 255
    to_do = Queue()
    to_do.put((0,0))
    while to_do.qsize() > 0:
        px,py = to_do.get()
        if mat[py][px] == 0:
            continue
        right = (px + 1, py)
        if flood_should_push_point(right, mat):
            to_do.put(right)
        left = (px - 1, py)
        if flood_should_push_point(left, mat):
            to_do.put(left)
        down = (px, py + 1)
        if flood_should_push_point(down, mat):
            to_do.put(down)
        top = (px, py - 1)
        if flood_should_push_point(top, mat):
            to_do.put(top)
        mat[py][px] = 0.0
        mask[py][px] = 0
    return mask


def compute_mat_x_gradient(mat): 
    rows, cols = mat.shape
    out = np.zeros((rows, cols), dtype='float64')
    mat = mat.astype(float)
    for y in range(rows):
        out[y][0] = mat[y][1] - mat[y][1]
        for x in range(cols - 1):
            out[y][x] = (mat[y][x+1] - mat[y][x-1])/2.0
        out[y][cols - 1] = (mat[y][cols - 1] - mat[y][cols - 2])
    return out



def find_eye_center(img):
    # get row and column lengths
    rows, cols = np.asarray(img).shape

    # scale down eye image to manageable size
    resized = scale_to_fast_size(img)
    resized_arr = np.asarray(resized)
    res_rows, res_cols = np.shape(resized_arr)

    # compute gradients for x and y components of each point
    grad_arr_x = compute_mat_x_gradient(resized_arr)
    grad_arr_y = np.transpose(compute_mat_x_gradient(np.transpose(resized_arr)))

    # create a matrix composed of the magnitudes of the x and y gradients
    mags_mat = matrix_magnitude(grad_arr_x, grad_arr_y)

    # find a threshold value to get rid gradients that are below gradient threshold
    gradient_threshold = compute_dynamic_threshold(mags_mat, K_GRADIENT_THRESHOLD)
    # and now set those gradients to 0 if < gradient threshold and scale down other
    # gradients

    for y in range(res_rows):
        for x in range(res_cols):
            gX = grad_arr_x[y][x]
            gY = grad_arr_y[y][x]
            mag = mags_mat[y][x]
            if mag > gradient_threshold: 
                grad_arr_x[y][x] = gX/mag
                grad_arr_y[y][x] = gY/mag
            else:
                grad_arr_x[y][x] = 0.0
                grad_arr_y[y][x] = 0.0

    # create a weighted image that has a gausian blur
    weight = cv2.GaussianBlur(resized, (K_WEIGHT_BLUR_SIZE, K_WEIGHT_BLUR_SIZE), 0, 0)
    weight_arr = np.asarray(weight)
    weight_rows, weight_cols = np.shape(weight_arr)
    # invert the weight matrix
    for y in range(weight_rows):
        for x in range(weight_cols):
            weight_arr[y][x] = 255-weight_arr[y][x]

    # create a matrix to store the results from test_possible_centers_formula
    out_sum = np.zeros((res_rows, res_cols))
    out_sum_rows, out_sum_cols = np.shape(out_sum)

    # call test_possible_centers for each point
    for y in range(weight_rows):
        for x in range(weight_cols):
            gX = grad_arr_x[y][x]
            gY = grad_arr_y[y][x]
            if gX == 0.0 and gY == 0.0:
                continue
            test_possible_centers_formula(x, y, weight_arr, gX, gY, out_sum)
    # average all values in out_sum and convert to float32. assign to 'out' matrix
    num_gradients = weight_rows * weight_cols
    out = out_sum.astype(np.float32)*(1/num_gradients)
    _, max_val, _, max_p = cv2.minMaxLoc(out)
    print max_p
    if K_POST_PROCESSING == True:
        flood_thresh = max_val * K_THRESHOLD_VALUE 
        retval, flood_clone = cv2.threshold(out, flood_thresh, 0.0, cv2.THRESH_TOZERO)
        mask = flood_kill_edges(flood_clone)
        _, max_val, _, max_p = cv2.minMaxLoc(out, mask)
        print max_p
    x, y = unscale_point(max_p, img)
    return x,y


img = cv2.imread('eyeold.jpg',0)
center = find_eye_center(img)
cv2.circle(img, center, 5, (255,0,0))
cv2.imshow('final', img)
cv2.waitKey(0)
