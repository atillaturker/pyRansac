import numpy as np

def computeHomography(src_points, dest_points):
    """Kaynak ve hedef noktalarla homografi matrisini hesaplar."""
    A = []
    for (x1, y1), (x2, y2) in zip(src_points, dest_points):
        A.append([x1, y1, 1, 0, 0, 0, -x2 * x1, -x2 * y1, -x2])
        A.append([0, 0, 0, x1, y1, 1, -y2 * x1, -y2 * y1, -y2])
    A = np.array(A)

    # Singular Value Decomposition (SVD)
    U, S, V = np.linalg.svd(A)

    # V[-1]: Homografi matrisini temsil eden özvektör
    H = np.reshape(V[-1], (3, 3))

    # Normalizasyon
    H = (1 / H.item(8)) * H
    return H


def dist(src_points, dest_points, H):
    """Returns the geometric distance between pairs of points given the
    homography H.

    Args:
        src_points (List[List]): List of source (x, y) points.
        dest_points (List[List]): List of destination (x, y) points.
        H (np.ndarray): The homography.

    Returns:
        List[float]: List of geometric distances for each pair.
    """
    distances = []
    for (x1, y1), (x2, y2) in zip(src_points, dest_points):
        # points in homogeneous coordinates
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])

        p2_estimate = np.dot(H, np.transpose(p1))
        p2_estimate = (1 / p2_estimate[2]) * p2_estimate

        distances.append(np.linalg.norm(np.transpose(p2) - p2_estimate))
    return distances

