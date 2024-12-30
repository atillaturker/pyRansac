import numpy as np
import cv2
import matplotlib.pyplot as plt
import random



def fit_line(points):
    """Homojen koordinatlarla doğru uydurma."""
    point1 = np.array([points[0][0], points[0][1], 1])  # Homojen koordinat [x1, y1, 1]
    point2 = np.array([points[1][0], points[1][1], 1])  # Homojen koordinat [x2, y2, 1]
    
    # İki nokta arasındaki doğru denklemi (cross product)
    line = np.cross(point1, point2)
    
    # Normalizasyon (isteğe bağlı)
    line = line / np.linalg.norm(line[:2])  # a^2 + b^2 = 1 olacak şekilde dönüştürme
    return line 


def line_error(point, line):
    """Bir noktadan bir doğruya olan dik uzaklığı hesaplar."""
    x, y = point  # Nokta [x, y]
    a, b, c = line  # Doğru denklemi: ax + by + c = 0
    
    numerator = abs(a * x + b * y + c)
    denominator = np.sqrt(a**2 + b**2)
    distance = numerator / denominator
    distance_least_square = (distance * distance) / 2
    return distance_least_square


def openCv_Fit_line(points, distType=cv2.DIST_L2):
    """Parameters
    points = Input vector of 2D or 3D points, stored in std::vector<> or Mat.
    line = Output line parameters. In case of 2D fitting, it should be a vector of 4 elements (like Vec4f) - (vx, vy, x0, y0), where (vx, vy) is a normalized vector collinear to the line and (x0, y0) is a point on the line. In case of 3D fitting, it should be a vector of 6 elements (like Vec6f) - (vx, vy, vz, x0, y0, z0), where (vx, vy, vz) is a normalized vector collinear to the line and (x0, y0, z0) is a point on the line.
    distType = Distance used by the M-estimator, see DistanceTypes
    param = Numerical parameter ( C ) for some types of distances. If it is 0, an optimal value is chosen default 0. .
    reps = Sufficient accuracy for the radius (distance between the coordinate origin and the line) default 0.01.
    aeps = Sufficient accuracy for the angle. 0.01 would be a good default value for reps and aeps. default 0.01
    """
    points = np.array(points, dtype=np.float32)
    [vx, vy, x0, y0] = cv2.fitLine(points, distType, param=0 , reps=0.01, aeps=0.01)
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

def generate_points_with_outliers(num_points, outlier_ratio, x_range=(0, 100), noise_std=0):
    """
    Rastgele bir doğrusal ilişki ve belirli bir oranda outlier içeren bir nokta kümesi oluşturur.

    Args:
        slope (float): Doğrunun eğimi (m).
        intercept (float): Doğrunun y kesişimi (b).
        num_points (int): Toplam nokta sayısı (inliers + outliers).
        outlier_ratio (float): Outliers oranı (0.0 - 1.0).
        x_range (tuple): X değerlerinin aralığı (min, max).
        noise_std (float): Inliers için eklenen Gaussian gürültü standard sapması.

    Returns:
        np.ndarray: Noktalar (x, y).
    """

    slope = random.uniform(-10, 10)
    intercept = random.uniform(-10, 10)

    num_outliers = int(num_points * outlier_ratio)
    num_inliers = num_points - num_outliers

    # Inliers oluşturma
    x_inliers = np.random.uniform(x_range[0], x_range[1], num_inliers)
    Gaussian_noise = np.random.normal(0, noise_std, size=num_inliers)
    y_inliers = slope * x_inliers + intercept + Gaussian_noise

    # Outliers oluşturma
    x_outliers = np.random.uniform(x_range[0], x_range[1], size=num_outliers)
    y_outliers = np.random.uniform(min(y_inliers), max(y_inliers), size=num_outliers)

    # Noktaları birleştirme
    x_points = np.concatenate([x_inliers, x_outliers])
    y_points = np.concatenate([y_inliers, y_outliers])

    # Noktaları karıştırma
    points = np.column_stack((x_points, y_points))
    np.random.shuffle(points)

    return points


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