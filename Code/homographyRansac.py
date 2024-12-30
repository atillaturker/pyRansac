import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from homographyModel import computeHomography , homographyError , extract_and_match_keypoints , back_projection_error
from ransac import RANSAC
from optimize_homography import optimize_homography
from HomographyEstimating import create_mosaic_color , warp_image , blend_images , create_mosaic_color
from test5 import stitch_images

def opencvHomography(point_map):
    src_points = point_map[:, :2]
    dst_points = point_map[:, 2:]

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    print("OpenCV İnliers:",len(mask))  
    return H, mask

def main():
    img1 = cv2.imread("../images/library/1.jpg", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("../images/library/2.jpg", cv2.IMREAD_GRAYSCALE)

    img_color1 = cv2.imread("../images/library/1.jpg")
    img_color2 = cv2.imread("../images/library/2.jpg")

    
    # Keypoint çıkarma ve eşleştirme
    point_map = extract_and_match_keypoints(img1, img2)

    H_opencv, mask_opencv = opencvHomography(point_map)

    H_ransac, inliers = RANSAC(
        point_map,
        computeHomography,
        homographyError,
        sample_size=4,
        threshold=10,
        num_iters=1000,
        random_seed=66,
        verbose=True
    )

    H_optimized = optimize_homography(inliers, H_ransac)

    print(f"OpenCV Homography Matrix:\n{H_opencv}")
    print("OpenCV Inliers:",len(mask_opencv))
    print(f"Custom RANSAC Homography Matrix:\n{H_ransac}")
    print(f"Custom RANSAC Inliers: {len(inliers)}")
    print(f"Optimized Homography Matrix:\n{H_optimized}")



    # Hata hesaplama
    error_opencv = back_projection_error(point_map,H_opencv)
    error_custom = back_projection_error(point_map, H_ransac)

    print(f"OpenCV Back Projection Error: {error_opencv}")
    print(f"Custom RANSAC Back Projection Error: {error_custom}")



    # # OpenCV homografisi ile mosaicking
    mosaic_opencv = create_mosaic_color(img_color1, img_color1, H_opencv)

    # Görselleştir
    plt.figure(figsize=(10, 5))
    plt.title("OpenCV Homography Mosaic")
    plt.imshow(mosaic_opencv, cmap='gray')
    plt.axis('off')
    plt.show()

    # Custom homografisi ile mosaicking
    mosaic_custom = create_mosaic_color(img_color1, img_color2, H_ransac)

    # Görselleştir
    plt.figure(figsize=(10, 5))
    plt.title("Custom RANSAC Homography Mosaic")
    plt.imshow(mosaic_custom, cmap='gray')
    plt.axis('off')
    plt.show()

    # Görüntüleri birleştir

    # # Görüntüleri yükle
    # img1 = cv2.imread("../images/library/1.jpg")
    # img2 = cv2.imread("../images/library/2.jpg")
    # # Stitch images
    # result = stitch_images(img1, img2)

    # # Save and display the result
    # cv2.imwrite('stitched_image.jpg', result)
    # cv2.imshow('Stitched Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    


    


if __name__ == "__main__":
    main()
