import numpy as np
import random
import cv2



def computeHomography(pairs):
    """ Computes the homography matrix given a list of point pairs.

    Args:
        pairs (List[List[List]]): List of pairs of (x, y) points.

    Returns:
        np.ndarray: The computed homography.
    """
    A = []
    for x1, y1, x2, y2 in pairs:
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V has shape (9, 9) for any number of input pairs. V[-1] is the eigenvector
    # of (A^T)A with the smalles eigenvalue. Reshape into 3x3 matrix.
    H = np.reshape(V[-1], (3, 3))

    # Normalization
    H = (1 / H.item(8)) * H
    return H


def homographyError(pair, H):
    """Returns the geometric distance between a pair of points given the
    homography H.

    Args:
        pair (List[List]): List of two (x, y) points.
        H (np.ndarray): The homography.

    Returns:
        float: The geometric distance.
    """
    # points in homogeneous coordinates
    p1 = np.array([pair[0], pair[1], 1])
    p2 = np.array([pair[2], pair[3], 1])

    p2_estimate = np.dot(H, np.transpose(p1))
    p2_estimate = (1 / p2_estimate[2]) * p2_estimate

    return np.linalg.norm(np.transpose(p2) - p2_estimate)

def extract_and_match_keypoints(image1, image2):
    sift = cv2.SIFT_create()

    # Keypoint ve descriptor çıkarma
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    # BFMatcher ile eşleşen noktaları bulma
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)

    #sort matches by score
    matches = sorted(matches, key = lambda x:x.distance)


    # point_map oluşturma
    point_map = np.array([
        [kp1[match.queryIdx].pt[0],
         kp1[match.queryIdx].pt[1],
         kp2[match.trainIdx].pt[0],
         kp2[match.trainIdx].pt[1]] for match in matches
    ])

    print("Point Map in extract_and_match_keypoints Length: ", len(point_map))

    return point_map


def createPointMap(image1, image2, verbose=True):
    """Creates a point map of shape (n, 4) where n is the number of matches
    between the two images. Each row contains (x1, y1, x2, y2), where (x1, y1)
    in image1 maps to (x2, y2) in image2.

    sift.detectAndCompute returns
        keypoints: a list of keypoints
        descriptors: a numpy array of shape (num keypoints, 128)

    Args:
        image1 (cv2.Mat): The first image.
        image2 (cv2.Mat): The second image.
        directory (str): The directory to save a .csv file to.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        List[List[List]]: The point map of (x, y) points from image1 to image2.
    """
    if verbose:
        print('Finding keypoints and descriptors for both images...')
    sift = cv2.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(image1, None)
    kp2, desc2 = sift.detectAndCompute(image2, None)

    if verbose:
        print('Determining matches...')
    matches = cv2.BFMatcher(cv2.NORM_L2, True).match(desc1, desc2)

    point_map = np.array([
        [kp1[match.queryIdx].pt[0],
         kp1[match.queryIdx].pt[1],
         kp2[match.trainIdx].pt[0],
         kp2[match.trainIdx].pt[1]] for match in matches
    ])
    print("Point Map createPointMap Length: ", len(point_map))
    return point_map

def back_projection_error(point_map, H):
    """
    point_map = np.array([
    [kp1[match.queryIdx].pt[0],
     kp1[match.queryIdx].pt[1],
     kp2[match.trainIdx].pt[0],
     kp2[match.trainIdx].pt[1]] for match in matches
     Şeklinde oluşturulmuş bir nokta haritası ve bir homografi matrisi alır ve geri projeksiyon hatasını hesaplar.
])
    
    """
    src_points = point_map[:, :2]
    dst_points = point_map[:, 2:]

    # Kaynak noktaları homografi matrisi ile dönüştür
    projected_points = cv2.perspectiveTransform(src_points.reshape(-1, 1, 2), H).reshape(-1, 2)

    # Geri projeksiyon hatasını hesapla
    error = np.linalg.norm(dst_points - projected_points, axis=1)
    return np.mean(error)