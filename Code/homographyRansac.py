import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from homographyModel import computeHomography, dist
from ransac import RANSAC



# 1. SIFT ile Keypoint ve Descriptor Çıkarımı
def extract_keypoints(image1, image2):
    sift = cv2.SIFT_create()

    # Keypoint ve descriptor çıkarma
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    return kp1, des1, kp2, des2

def match_keypoints(des1, des2, kp1, kp2):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)  # En iyi eşleşmeler önde
    point_map = []

    for match in matches:
        src_point = kp1[match.queryIdx].pt
        dst_point = kp2[match.trainIdx].pt
        point_map.append([src_point[0], src_point[1], dst_point[0], dst_point[1]])

    return point_map


# 3. RANSAC ile Homografi Hesaplama
def fit_homography(pairs):
    """
    Rastgele seçilen nokta çiftlerinden homografi matrisini hesaplar.

    Args:
        pairs (List[List[float]]): Rastgele seçilmiş noktaların listesi. 
                                   Her eleman [x1, y1, x2, y2] formatında.

    Returns:
        np.ndarray: Hesaplanan homografi matrisi (3x3 boyutunda).
    """
    src_points = [pair[:2] for pair in pairs]
    dst_points = [pair[2:] for pair in pairs]
    return computeHomography(src_points, dst_points)

def homography_error(pair, H):
    """
    Bir nokta çiftinin homografi ile tahmin edilen hedef noktadan farkını ölçer.

    Args:
        pair (List[float]): Tek bir nokta çifti [x1, y1, x2, y2] formatında.
        H (np.ndarray): Homografi matrisi.

    Returns:
        float: Geometrik hata (kaynak noktanın hedef noktaya dönüşüm hatası).
    """
    src_point = pair[:2]  # İlk iki eleman kaynak nokta
    dest_point = pair[2:]  # Son iki eleman hedef nokta
    return dist([src_point], [dest_point], H)[0]  # Tek bir çift için mesafeyi al

# 4. OpenCV ile Homografi Hesaplama
def opencv_homography(point_map):
    src_points = np.array([pair[:2] for pair in point_map], dtype=np.float32)
    dst_points = np.array([pair[2:] for pair in point_map], dtype=np.float32)

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H, mask

def back_projection_error(src_points, dst_points, H):
    errors = []
    for (x1, y1), (x2, y2) in zip(src_points, dst_points):
        p1 = np.array([x1, y1, 1])
        p2 = np.array([x2, y2, 1])
        p2_estimated = np.dot(H, p1)
        p2_estimated /= p2_estimated[2]  # Normalize et
        errors.append(np.linalg.norm(p2[:2] - p2_estimated[:2]))
    return np.mean(errors)

def warp_image(image, H, shape):
    """
    Bir görüntüyü homografi matrisini kullanarak dönüştürür.

    Args:
        image (np.ndarray): Giriş görüntüsü.
        H (np.ndarray): Homografi matrisi.
        shape (tuple): Çıkış görüntüsünün boyutları (genişlik, yükseklik).

    Returns:
        np.ndarray: Dönüştürülmüş görüntü.
    """
    return cv2.warpPerspective(image, H, shape)


def blend_images(img1, img2, H):
    """
    İki görüntüyü homografi kullanarak birleştirir.

    Args:
        img1 (np.ndarray): Birinci görüntü.
        img2 (np.ndarray): İkinci görüntü.
        H (np.ndarray): Homografi matrisi.

    Returns:
        np.ndarray: Birleştirilmiş görüntü.
    """
    # Birinci görüntüyü dönüştür
    h2, w2 = img2.shape
    warped_img1 = cv2.warpPerspective(img1, H, (w2, h2))

    # İkinci görüntüyü ve dönüştürülmüş birinci görüntüyü birleştir
    combined = np.maximum(warped_img1, img2)
    return combined

# 5. Ana İşlem
def main():
    # Görüntüleri yükleme 
    image1_path = "images/aero1.jpg"
    image2_path = "images/aero3.jpg"

    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        raise ValueError("Görüntüler yüklenemedi. Dosya yollarını kontrol edin.")

    # Keypoint çıkarma
    kp1, des1, kp2, des2 = extract_keypoints(img1, img2)

    # Keypoint eşleşmelerini bul
    point_map = match_keypoints(des1, des2, kp1, kp2)

    # Hata hesaplama
    src_points = [pair[:2] for pair in point_map]
    dst_points = [pair[2:] for pair in point_map]

    
    
   
    # RANSAC ile homografi hesaplama

    H_opencv, mask_opencv = opencv_homography(point_map)
    

    best_homography, inliers = RANSAC(
        point_map,
        fit_homography,
        homography_error,
        sample_size=4,
        threshold=5.0,
        num_iters=1000,
        verbose=True
    )

    print(f"OpenCV Homography Matrix:\n{H_opencv}")
    print(f"Custom RANSAC Homography Matrix:\n{best_homography}")
  
    H_ransac = best_homography


    


   


    # Hata hesaplama
    error_opencv = back_projection_error(src_points, dst_points, H_opencv)
    error_custom = back_projection_error(src_points, dst_points, H_ransac)

    print(f"OpenCV Back Projection Error: {error_opencv}")
    print(f"Custom RANSAC Back Projection Error: {error_custom}")


    # h2, w2 = img2.shape

    # # OpenCV'nin Homografisi
    # result_opencv = warp_image(img1, H_opencv, (w2, h2))

    # # Kendi RANSAC Homografisi
    # result_custom = warp_image(img1, H_ransac, (w2, h2))

    # # Görüntüleri göster
    # cv2.imshow("OpenCV Result", result_opencv)
    # cv2.imshow("Custom RANSAC Result", result_custom)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # OpenCV ile birleştirme
    combined_opencv = blend_images(img1, img2, H_opencv)

    # Kendi RANSAC ile birleştirme
    combined_custom = blend_images(img1, img2, H_ransac)

    # Görüntüleri görselleştir
    cv2.imshow("Combined OpenCV", combined_opencv)
    cv2.imshow("Combined Custom RANSAC", combined_custom)
    cv2.waitKey(0)
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    main()
