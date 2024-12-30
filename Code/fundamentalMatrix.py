import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from homographyRansac import createPointMap

from ransac import RANSAC  # RANSAC fonksiyonu import edildi

def fit_fundamental_matrix(points):
    """
    Rastgele seçilen 8 nokta ile Fundamental matrisi hesaplar.
    
    Args:
        points (list): Eşleşen noktalar [x1, y1, x2, y2].
    
    Returns:
        np.ndarray: 3x3 Fundamental matrisi.
    """
    A = []
    for x1, y1, x2, y2 in points:
        A.append([x1 * x2, y1 * x2, x2, x1 * y2, y1 * y2, y2, x1, y1, 1])
    A = np.array(A)

    # SVD ile Fundamental matrisi bulma
    _, _, V = np.linalg.svd(A)
    F = V[-1].reshape(3, 3)

    # Rank-2 zorlaması
    U, S, Vt = np.linalg.svd(F)
    S[-1] = 0  # En küçük singular değer sıfırlanır
    F = np.dot(U, np.dot(np.diag(S), Vt))
    return F

def fundamental_error(point, F):
    """
    Calculates the epipolar error for a single point pair.
    
    Args:
        point (list): [x1, y1, x2, y2] pair of points.
        F (np.ndarray): Fundamental matrix.
    
    Returns:
        float: Epipolar error.
    """
    x1, y1, x2, y2 = point
    p1 = np.array([x1, y1, 1])  # Homogeneous coordinates for first point
    p2 = np.array([x2, y2, 1])  # Homogeneous coordinates for second point

    # Epipolar line for p1 in second image
    line = np.dot(F, p1)
    line /= np.linalg.norm(line[:2])  # Normalize line coefficients

    # Distance from p2 to the epipolar line
    error = abs(np.dot(line, p2))  # Epipolar constraint
    return error


def calculate_epipolar_errors(points, F):
    """
    Calculates the epipolar errors for a set of points given a Fundamental matrix.

    Args:
        points (list): List of matching points [x1, y1, x2, y2].
        F (np.ndarray): Fundamental matrix.

    Returns:
        list: Epipolar errors for each point.
    """
    errors = []
    for x1, y1, x2, y2 in points:
        p1 = np.array([x1, y1, 1])  # Homogeneous source point
        p2 = np.array([x2, y2, 1])  # Homogeneous destination point

        # Epipolar line for p1 in the second image
        line = np.dot(F, p1)
        line /= np.linalg.norm(line[:2])  # Normalize the line

        # Distance from p2 to the epipolar line
        error = abs(np.dot(line, p2))
        errors.append(error)

    return errors



# Örnek veri: Eşleşen noktalar [x1, y1, x2, y2]
point_map = [
    [100, 200, 150, 250],
    [120, 220, 170, 270],
    [130, 230, 180, 280],
    [140, 240, 190, 290],
    [150, 250, 200, 300],
    [160, 260, 210, 310],
    [170, 270, 220, 320],
    [180, 280, 230, 330],
    [190, 290, 240, 340],
    [200, 300, 250, 350],
    [210, 310, 260, 360],
    [220, 320, 270, 370],
    [230, 330, 280, 380],
    [240, 340, 290, 390],
    [250, 350, 300, 400]
]

def draw_epipolar_lines(image1, image2, points, F):
    """
    Draws epipolar lines on the images given the Fundamental matrix.

    Args:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
        points (list): Matching points [x1, y1, x2, y2].
        F (np.ndarray): Fundamental matrix.

    Returns:
        (np.ndarray, np.ndarray): Images with epipolar lines drawn.
    """
    import cv2

    # Draw lines on both images
    img1_lines = image1.copy()
    img2_lines = image2.copy()

    for x1, y1, x2, y2 in points:
        p1 = np.array([x1, y1, 1])  # Homogeneous source point
        p2 = np.array([x2, y2, 1])  # Homogeneous destination point

        # Epipolar line in image2 for point in image1
        line = np.dot(F, p1)

        # Line endpoints in image2
        h, w = img2_lines.shape
        x_start, y_start = 0, int(-line[2] / line[1])  # Start of the line
        x_end, y_end = w, int((-line[2] - line[0] * w) / line[1])  # End of the line

        # Draw line and corresponding point
        img2_lines = cv2.line(img2_lines, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
        img2_lines = cv2.circle(img2_lines, (int(x2), int(y2)), 5, (0, 255, 0), -1)

        # Epipolar line in image1 for point in image2 (Optional)
        # Add similar logic for the second image if needed

    return img1_lines, img2_lines



# OpenCV ile Fundamental matrisi hesaplama

# OpenCV ile Fundamental Matris Hesaplama
# src_points = np.array([[x1, y1] for x1, y1, _, _ in point_map], dtype=np.float32)
# dst_points = np.array([[x2, y2] for _, _, x2, y2 in point_map], dtype=np.float32)

image1 = cv2.imread("../images/Blender_Suzanne1.jpg", cv2.IMREAD_GRAYSCALE)
image2 = cv2.imread("../images/Blender_Suzanne2.jpg", cv2.IMREAD_GRAYSCALE)



# Görüntüleri görselleştir
plt.imshow(image1, cmap='gray')
plt.title("Image 1")
plt.show()

plt.imshow(image2, cmap='gray')
plt.title("Image 2")
plt.show()

print(f"Image1 shape: {image1.shape}, dtype: {image1.dtype}")
print(f"Image2 shape: {image2.shape}, dtype: {image2.dtype}")



points2= createPointMap(image1, image2)
src_points = np.array([[x1, y1] for x1, y1, x2, y2 in points2], dtype=np.float32)
dst_points = np.array([[x2, y2] for x1, y1, x2, y2 in points2], dtype=np.float32)

F_opencv, mask = cv2.findFundamentalMat(src_points, dst_points, cv2.FM_RANSAC, 0.01)


# +50 pixel ortalama hata ekleyin
noise = np.random.normal(0, 500, src_points.shape)
src_points_noisy = src_points + noise # Add noise to source points
dst_points_noisy = dst_points + noise # Add noise to destination points

F_opencv_noisy, mask_noisy = cv2.findFundamentalMat(src_points_noisy, dst_points_noisy, cv2.FM_RANSAC, 0.01)

# OpenCV ile bulunan inlier sayısı

inliers_opencv_noisy = np.sum(mask_noisy)
print(f"OpenCV Noisy Inliers: {inliers_opencv_noisy}")





# RANSAC algoritması
best_F, inliers = RANSAC(
    data=points2,
    model_func=fit_fundamental_matrix,
    error_func=fundamental_error,
    sample_size=8,
    threshold=0.01,
    num_iters=100,
    random_seed=42,
)

print("En iyi Fundamental matrisi:")
print(best_F)
print(f"Inlier sayısı: {len(inliers)}")

print("OpenCV Fundamental matrisi:")
print(F_opencv)
inlier_count = np.sum(mask)
print(f"OpenCV Inlier Count: {inlier_count}")

# OpenCV ile bulunan inlier sayısı
inliers_opencv = np.sum(mask)


plt.scatter(src_points[:, 0], src_points[:, 1], color='blue', label='Source Points')
plt.scatter(dst_points[:, 0], dst_points[:, 1], color='red', label='Destination Points')
plt.legend()
plt.title("Matched Points")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

# Kendi fonksiyonunuzun matrisine göre epipolar hatalar
custom_errors = calculate_epipolar_errors(point_map, best_F)

# OpenCV'nin matrisine göre epipolar hatalar
opencv_errors = calculate_epipolar_errors(point_map, F_opencv)

# Ortalama hatalar
custom_mean_error = np.mean(custom_errors)
opencv_mean_error = np.mean(opencv_errors)

print(f"Custom RANSAC Mean Epipolar Error: {custom_mean_error}")
print(f"OpenCV RANSAC Mean Epipolar Error: {opencv_mean_error}")

def calculate_epipolar_errors(points, F):
    """
    Calculates the epipolar errors for a set of points given a Fundamental matrix.

    Args:
        points (list): List of matching points [x1, y1, x2, y2].
        F (np.ndarray): Fundamental matrix.

    Returns:
        list: Epipolar errors for each point.
    """
    errors = []
    for x1, y1, x2, y2 in points:
        p1 = np.array([x1, y1, 1])  # Homogeneous source point
        p2 = np.array([x2, y2, 1])  # Homogeneous destination point

        # Epipolar line for p1 in the second image
        line = np.dot(F, p1)
        line /= np.linalg.norm(line[:2])  # Normalize the line

        # Distance from p2 to the epipolar line
        error = abs(np.dot(line, p2))
        errors.append(error)

    return errors

custom_errors = calculate_epipolar_errors(point_map, best_F)

# OpenCV'nin matrisine göre epipolar hatalar
opencv_errors = calculate_epipolar_errors(point_map, F_opencv)

# Ortalama hatalar
custom_mean_error = np.mean(custom_errors)
opencv_mean_error = np.mean(opencv_errors)

print(f"Custom RANSAC Mean Epipolar Error: {custom_mean_error}")
print(f"OpenCV RANSAC Mean Epipolar Error: {opencv_mean_error}")

def draw_epipolar_lines(image1, image2, points, F):
    """
    Draws epipolar lines on the images given the Fundamental matrix.

    Args:
        image1 (np.ndarray): The first image.
        image2 (np.ndarray): The second image.
        points (list): Matching points [x1, y1, x2, y2].
        F (np.ndarray): Fundamental matrix.

    Returns:
        (np.ndarray, np.ndarray): Images with epipolar lines drawn.
    """
    import cv2

    # Draw lines on both images
    img1_lines = image1.copy()
    img2_lines = image2.copy()

    for x1, y1, x2, y2 in points:
        p1 = np.array([x1, y1, 1])  # Homogeneous source point
        p2 = np.array([x2, y2, 1])  # Homogeneous destination point

        # Epipolar line in image2 for point in image1
        line = np.dot(F, p1)

        # Line endpoints in image2
        h, w = img2_lines.shape
        x_start, y_start = 0, int(-line[2] / line[1])  # Start of the line
        x_end, y_end = w, int((-line[2] - line[0] * w) / line[1])  # End of the line

        # Draw line and corresponding point
        img2_lines = cv2.line(img2_lines, (x_start, y_start), (x_end, y_end), (255, 0, 0), 1)
        img2_lines = cv2.circle(img2_lines, (int(x2), int(y2)), 5, (0, 255, 0), -1)

        # Epipolar line in image1 for point in image2 (Optional)
        # Add similar logic for the second image if needed

    return img1_lines, img2_lines

# Görselleştirme
custom_img1, custom_img2 = draw_epipolar_lines(image1, image2, point_map, best_F)
opencv_img1, opencv_img2 = draw_epipolar_lines(image1, image2, point_map, F_opencv)

# OpenCV pencerelerinde görüntüleyin
cv2.imshow("Custom RANSAC Epipolar Lines (Image 2)", custom_img2)
cv2.imshow("OpenCV RANSAC Epipolar Lines (Image 2)", opencv_img2)
cv2.waitKey(0)
cv2.destroyAllWindows()


print("OpenCV Inliers with Noise:", np.sum(mask))