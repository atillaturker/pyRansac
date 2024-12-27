import cv2
import numpy as np
from matplotlib import pyplot as plt



# Örnek homografi matrisleri
H_opencv = np.array([[1.00138292e+00 , 9.78104318e-04, -4.90786078e-01],
                     [3.17155857e-04 , 1.00094850e+00 ,-1.08663579e-01],
                     [1.19552190e-06 , 1.38466226e-06 , 1.00000000e+00]])

H_ransac = np.array([[1.00185172e+00 , 5.90525082e-03, -2.69083093e+00],
                     [-2.17111411e-03 , 9.99577458e-01  ,1.04261643e+00],
                     [ -2.80163821e-06 , 3.90975620e-06 , 1.00000000e+00]])

# Görüntüleri yükleme
img1 = cv2.imread('left01.jpg')
img2 = cv2.imread('left02.jpg')


# Görüntü boyutları
h, w = img1.shape[:2]

# OpenCV homografi matrisi ile dönüştürme
img1_transformed_opencv = cv2.warpPerspective(img1, H_opencv, (w, h))

# Kendi RANSAC homografi matrisi ile dönüştürme
img1_transformed_ransac = cv2.warpPerspective(img1, H_ransac, (w, h))

# Görüntüleri gösterme
plt.subplot(131), plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
plt.subplot(132), plt.imshow(cv2.cvtColor(img1_transformed_opencv, cv2.COLOR_BGR2RGB)), plt.title('Transformed (OpenCV)')
plt.subplot(133), plt.imshow(cv2.cvtColor(img1_transformed_ransac, cv2.COLOR_BGR2RGB)), plt.title('Transformed (RANSAC)')
plt.show()



def back_projection_error(src_points, dst_points, H):
    errors = []
    for (x1, y1), (x2, y2) in zip(src_points, dst_points):
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])
        p2_estimated = np.dot(H, p1)
        p2_estimated /= p2_estimated[2]  # Normalize et
        errors.append(np.linalg.norm(p2[:2] - p2_estimated[:2]))
    return np.mean(errors)

# Hata hesaplama
src_points = [pair[:2] for pair in point_map_example]
dst_points = [pair[2:] for pair in point_map_example]

error_opencv = back_projection_error(src_points, dst_points, H_opencv)
error_custom = back_projection_error(src_points, dst_points, H_ransac)

print(f"OpenCV Back Projection Error: {error_opencv}")
print(f"Custom RANSAC Back Projection Error: {error_custom}")
