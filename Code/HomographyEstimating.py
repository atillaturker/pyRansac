import cv2
import numpy as np

def create_mosaic(image1, image2, H):
    """
    Performs mosaicking using a homography matrix.

    Args:
        image1 (np.ndarray): The first image (base image).
        image2 (np.ndarray): The second image to warp.
        H (np.ndarray): Homography matrix.

    Returns:
        np.ndarray: Mosaic image.
    """
    # Boyutları belirleyin
    h1, w1 = image1.shape
    h2, w2 = image2.shape

    # Köşe noktalarını dönüştür
    corners_image2 = np.array([
        [0, 0, 1],  # Top-left
        [w2, 0, 1],  # Top-right
        [0, h2, 1],  # Bottom-left
        [w2, h2, 1]  # Bottom-right
    ])
    transformed_corners = np.dot(H, corners_image2.T)
    transformed_corners = (transformed_corners / transformed_corners[2]).T[:, :2]

    # Yeni boyutları hesapla
    all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [0, h1], [w1, h1]]))
    x_min, y_min = np.min(all_corners, axis=0).astype(int)
    x_max, y_max = np.max(all_corners, axis=0).astype(int)

    # Homografiyi base image için ayarla
    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    new_width = x_max - x_min
    new_height = y_max - y_min

    # Görüntüleri yeni alana warp et
    result = cv2.warpPerspective(image2, translation_matrix @ H, (new_width, new_height))
    result[-y_min:h1 - y_min, -x_min:w1 - x_min] = image1

    return result

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

def create_mosaic_color(image1, image2, H):
    """
    Performs mosaicking for color images using a homography matrix.

    Args:
        image1 (np.ndarray): The first color image (base image).
        image2 (np.ndarray): The second color image to warp.
        H (np.ndarray): Homography matrix.

    Returns:
        np.ndarray: Mosaic color image.
    """
    # Boyutları belirleyin
    h1, w1, _ = image1.shape
    h2, w2, _ = image2.shape

    # Köşe noktalarını dönüştür
    corners_image2 = np.array([
        [0, 0, 1],  # Top-left
        [w2, 0, 1],  # Top-right
        [0, h2, 1],  # Bottom-left
        [w2, h2, 1]  # Bottom-right
    ])
    transformed_corners = np.dot(H, corners_image2.T)
    transformed_corners = (transformed_corners / transformed_corners[2]).T[:, :2]

    # Yeni boyutları hesapla
    all_corners = np.vstack((transformed_corners, [[0, 0], [w1, 0], [0, h1], [w1, h1]]))
    x_min, y_min = np.min(all_corners, axis=0).astype(int)
    x_max, y_max = np.max(all_corners, axis=0).astype(int)

    # Homografiyi base image için ayarla
    translation_matrix = np.array([
        [1, 0, -x_min],
        [0, 1, -y_min],
        [0, 0, 1]
    ])
    new_width = x_max - x_min
    new_height = y_max - y_min

    # Görüntüleri yeni alana warp et
    result = cv2.warpPerspective(image2, translation_matrix @ H, (new_width, new_height))
    result[-y_min:h1 - y_min, -x_min:w1 - x_min] = image1

    return result