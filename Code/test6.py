
from homographyModel import CalculateHomography, homographyError
import numpy as np


def RANSAC_TEST(point_map, NUM_ITERS,threshold,verbose=False):
    """Runs the RANSAC algorithm.

    Args:
        point_map (List[List[List]]): Map of (x, y) points from one image to the
            another image.
        threshold (float, optional): The minimum portion of points that should
            be inliers before the algorithm terminates. Defaults to THRESHOLD.
        verbose (bool, optional): True if additional information should be
            printed. Defaults to True.

    Returns:
        (np.ndarray, set(List[List])): The homography and set of inliers.
    """
    if verbose:
        print(f'Running RANSAC with {len(point_map)} points...')
    bestInliers = set()
    homography = None
    for i in range(NUM_ITERS):
        # randomly choose 4 points from the matrix to compute the homography
        pairs = [point_map[i] for i in np.random.choice(len(point_map), 4)]

        H = CalculateHomography(pairs)
        inliers = {(c[0], c[1], c[2], c[3])
                   for c in point_map if homographyError(c, H) < 500}

        if verbose:
            print(f'\x1b[2K\r└──> iteration {i + 1}/{NUM_ITERS} ' +
                  f'\t{len(inliers)} inlier' + ('s ' if len(inliers) != 1 else ' ') +
                  f'\tbest: {len(bestInliers)}', end='')

        if len(inliers) > len(bestInliers):
            bestInliers = inliers
            homography = H
            if len(bestInliers) > (len(point_map) * threshold):
                break

    if verbose:
        print(f'\nNum matches: {len(point_map)}')
        print(f'Num inliers: {len(bestInliers)}')
        print(f'Min inliers: {len(point_map) * threshold}')

    return homography, bestInliers