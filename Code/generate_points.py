import numpy as np
import numpy as np

def generate_points_with_outliers(slope, intercept, num_points, outlier_ratio, x_range=(0, 100), noise_std=0):
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
    num_outliers = int(num_points * outlier_ratio)
    num_inliers = num_points - num_outliers

    # Inliers oluşturma
    x_inliers = np.random.uniform(x_range[0], x_range[1], num_inliers)
    Gaussian_noise = np.random.normal(0, noise_std, size=num_inliers)
    print("Gaussian_noise",Gaussian_noise)
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