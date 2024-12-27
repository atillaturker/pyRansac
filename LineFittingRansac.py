import cv2
import numpy as np
import random
from ransac import RANSAC
from homographyModel import computeHomography, dist
import numpy as np
import matplotlib.pyplot as plt
from generate_points import generate_points_with_outliers




def visualize_points(points, slope=None, intercept=None, additional_slope=None, additional_intercept=None):
        """
        Oluşturulan noktaları ve opsiyonel olarak bir veya daha fazla doğruyu görselleştirir.

        Args:
            points (np.ndarray): Nokta kümesi (x, y).
            slope (float, optional): Doğrunun eğimi (m).
            intercept (float, optional): Doğrunun y kesişimi (b).
            additional_lines (list of tuples, optional): Ekstra çizilecek doğruların eğim ve kesişim değerleri [(slope1, intercept1), (slope2, intercept2), ...].
        """
        x, y = points[:, 0], points[:, 1]
        
        # visualize_points
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Points (Inliers + Outliers)', alpha=0.7)
        
        # if slope and intercept are provided, plot the line
        if slope is not None and intercept is not None:
            x_line = np.linspace(min(x), max(x), 500)
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, color='red', label=f'OpenCV Truth Line (y = {slope:.2f}x + {intercept:.2f})')
        
        # if additional_lines are provided, plot them
        if additional_slope is not None and additional_intercept is not None:
            x_line = np.linspace(min(x), max(x), 500)
            y_line = additional_slope * x_line + additional_intercept
            plt.plot(x_line, y_line, color='green', label=f'Ransac Truth Line (y = {additional_slope:.2f}x + {additional_intercept:.2f})')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Generated Points with Outliers')
        plt.legend()
        plt.grid()
        plt.show()
        x, y = points[:, 0], points[:, 1]
    
        
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, color='blue', label='Points (Inliers + Outliers)', alpha=0.7)


def fit_line(points):
    """Homojen koordinatlarla doğru uydurma."""
    p1 = np.array([points[0][0], points[0][1], 1])  # Homojen koordinat [x1, y1, 1]
    p2 = np.array([points[1][0], points[1][1], 1])  # Homojen koordinat [x2, y2, 1]
    
    # İki nokta arasındaki doğru denklemi (cross product)
    line = np.cross(p1, p2)
    
    # Normalizasyon (isteğe bağlı)
    line = line / np.linalg.norm(line[:2])  # a^2 + b^2 = 1 olacak şekilde normalize et

    slope = -line[0] / line[1]
    intercept = -line[2] / line[1]
    return line 


def line_error(point, line):
    """Noktanın doğruya olan mesafesini homojen doğrularla hesaplar."""
    x, y = point  # Nokta [x, y]
    a, b, c = line  # Doğru denklemi: ax + by + c = 0
    
    # Mesafe formülü
    distance = abs(a * x + b * y + c) / np.sqrt(a**2 + b**2)
    return distance
    


def openCv_Fit_line(points, distType=cv2.DIST_L2, param=0, reps=0.01, aeps=0.01):
    """Parameters
            points = Input vector of 2D or 3D points, stored in std::vector<> or Mat.
            line = Output line parameters. In case of 2D fitting, it should be a vector of 4 elements (like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line and (x0, y0, z0) is a point on the line.
            distType = Distance used by the M-estimator, see DistanceTypes
            param = Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value is chosen.
            reps = Sufficient accuracy for the radius (distance between the coordinate origin and the line).
            aeps = Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps."""
    points = np.array(points, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points, distType, param, reps, aeps)
    slope = vy / vx
    intercept = y0 - slope * x0
    return slope[0], intercept[0]


def line_to_slope_intercept(line):
    """
    Homojen doğru denklemini slope ve intercept'e çevirir.

    Args:
        line (array): [a, b, c] homojen doğru denklemi.

    Returns:
        tuple: (slope, intercept)
    """
    a, b, c = line

    # Dik doğru kontrolü (b == 0 ise slope sonsuz olur)
    if b == 0:
        slope = float('inf')  # Dikey doğru
        intercept = -c / a    # x-intercept
    else:
        slope = -a / b
        intercept = -c / b

    return slope, intercept



slope = random.uniform(-10, 10)
intercept = random.uniform(-10, 10)
points_map = generate_points_with_outliers(slope, intercept, num_points=100, outlier_ratio=0.2)


best_line, inliers = RANSAC(
    points_map,
    fit_line,
    line_error,
    sample_size=2,
    threshold=1.0,
    num_iters=100,
    verbose=True
)


fit_slope, fit_intercept = line_to_slope_intercept(best_line)
best_line_opencv = openCv_Fit_line(points_map)
openCV_slope, openCV_intercept = best_line_opencv


visualize_points(points_map, slope=openCV_slope, intercept=openCV_intercept, additional_slope=fit_slope, additional_intercept=fit_intercept)


