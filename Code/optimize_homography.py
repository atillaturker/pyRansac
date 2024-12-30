from scipy.optimize import least_squares
import numpy as np

def optimize_homography(data, initial_H):
    """
    Optimize homography matrix using inlier points and least squares method.
    
    Args:
        data (List[List]): List of point pairs [(x1, y1, x2, y2), ...].
        initial_H (np.ndarray): Initial homography matrix (3x3).

    Returns:
        np.ndarray: Optimized homography matrix.
    """
    def residuals(H_flat, data):
        """
        Compute residuals (geometric distances) for all point pairs.
        
        Args:
            H_flat (np.ndarray): Flattened homography matrix (1D array with 9 elements).
            data (List[List]): List of point pairs [(x1, y1, x2, y2), ...].
        
        Returns:
            np.ndarray: Residuals (geometric distances) for each point pair.
        """
        H = H_flat.reshape((3, 3))
        residuals = []
        for x1, y1, x2, y2 in data:
            p1 = np.array([x1, y1, 1])
            p2 = np.array([x2, y2, 1])
            
            # Estimate the second point using H
            p2_estimate = np.dot(H, p1)
            p2_estimate /= p2_estimate[2]  # Normalize to homogeneous coordinates
            
            # Compute the geometric distance
            distance = np.linalg.norm(p2 - p2_estimate)
            residuals.append(distance)
        return np.array(residuals)

    # Flatten the initial homography matrix
    initial_H_flat = initial_H.flatten()
    
    # Run least squares optimization
    result = least_squares(
        fun=residuals,
        x0=initial_H_flat,
        args=(data,)
    )
    
    # Reshape the optimized H back to 3x3
    optimized_H = result.x.reshape((3, 3))
    
    # Normalize to make H[2, 2] = 1
    optimized_H /= optimized_H[2, 2]
    
    return optimized_H
