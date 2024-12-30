import cv2
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random

def alpha_blend(img1_row, img2_row, seam_x, blend_window, direction):
    """
    Alpha blending for a single row of pixels.
    """
    blended_row = np.zeros_like(img1_row)
    if direction == 'left':
        for x in range(seam_x - blend_window, seam_x + blend_window):
            alpha = (x - (seam_x - blend_window)) / (2 * blend_window)
            blended_row[x] = img1_row[x] * (1 - alpha) + img2_row[x] * alpha
    else:
        for x in range(seam_x - blend_window, seam_x + blend_window):
            alpha = (x - (seam_x - blend_window)) / (2 * blend_window)
            blended_row[x] = img2_row[x] * (1 - alpha) + img1_row[x] * alpha
    return blended_row

def stitch_images(img1, img2, blending=True):
    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Find keypoints and descriptors with SIFT
    keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

    # Use FLANN based matcher for feature matching
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    # Store all good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # Extract location of good matches
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)

    # Find homography matrix
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Use homography matrix to warp images
    height, width = img1.shape[:2]
    img2_warped = cv2.warpPerspective(img2, H, (width + img2.shape[1], height))

    # Calculate shift
    shift = [int(H[0][2]), int(H[1][2])]

    # Create a pool for parallel processing
    pool = Pool()

    # Stitch images using the provided stitching function
    result = stitching(img1, img2_warped, shift, pool, blending)

    return result

def stitching(img1, img2, shift, pool, blending=True):
    padding = [
        (shift[0], 0) if shift[0] > 0 else (0, -shift[0]),
        (shift[1], 0) if shift[1] > 0 else (0, -shift[1]),
        (0, 0)
    ]
    shifted_img1 = np.pad(img1, padding, 'constant', constant_values=0)

    # cut out unnecessary region
    split = img2.shape[1] + abs(shift[1])
    splited = shifted_img1[:, split:] if shift[1] > 0 else shifted_img1[:, :-split]
    shifted_img1 = shifted_img1[:, :split] if shift[1] > 0 else shifted_img1[:, -split:]

    h1, w1, _ = shifted_img1.shape
    h2, w2, _ = img2.shape
    
    inv_shift = [h1 - h2, w1 - w2]
    inv_padding = [
        (inv_shift[0], 0) if shift[0] < 0 else (0, inv_shift[0]),
        (inv_shift[1], 0) if shift[1] < 0 else (0, inv_shift[1]),
        (0, 0)
    ]
    shifted_img2 = np.pad(img2, inv_padding, 'constant', constant_values=0)

    direction = 'left' if shift[1] > 0 else 'right'

    if blending:
        seam_x = shifted_img1.shape[1] // 2
        tasks = [(shifted_img1[y], shifted_img2[y], seam_x, 30, direction) for y in range(h1)]
        shifted_img1 = pool.starmap(alpha_blend, tasks)
        shifted_img1 = np.asarray(shifted_img1)
        shifted_img1 = np.concatenate((shifted_img1, splited) if shift[1] > 0 else (splited, shifted_img1), axis=1)
    else:
        raise ValueError('I did not implement "blending=False" ^_^')

    return shifted_img1