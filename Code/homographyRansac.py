import cv2
import numpy as np
import matplotlib.pyplot as plt
import random
from homographyModel import CalculateHomography , homographyError , extract_and_match_keypoints , back_projection_error
from ransac import RANSAC
from optimize_homography import optimize_homography
from HomographyEstimating import stitch_images
from test6 import RANSAC_TEST

def opencvHomography(point_map):
    src_points = point_map[:, :2]
    dst_points = point_map[:, 2:]

    H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
    return H, mask

def main():
    img1 = cv2.imread("../images/boat1.jpg",)
    img2 = cv2.imread("../images/boat2.jpg",) 

    img_color1 = cv2.imread("../images/library/1.jpg")
    img_color2 = cv2.imread("../images/library/2.jpg")

    
    # Keypoint çıkarma ve eşleştirme
    point_map = extract_and_match_keypoints(img1, img2)

    H_opencv, mask_opencv = opencvHomography(point_map)

    H_ransac, inliers = RANSAC(
        point_map,
        CalculateHomography,
        homographyError,
        sample_size=4,
        threshold=1,
        num_iters=100,
        random_seed=11,
        verbose=True
    )

    # H_ransac_test , best_inliers = RANSAC_TEST(point_map, 1000, 0.5, False)
    # print(f"Best Inliers: {len(best_inliers)}")
    # print(f"H_Ransac Test :  Homography Matrix:\n{H_ransac_test}")
    print(f"Custom RANSAC Homography Matrix:\n{H_ransac}")

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



    # # # OpenCV homografisi ile mosaicking
    # mosaic_opencv = create_mosaic_color(img_color1, img_color1, H_opencv)

    # # Görselleştir
    # plt.figure(figsize=(10, 5))
    # plt.title("OpenCV Homography Mosaic")
    # plt.imshow(mosaic_opencv, cmap='gray')
    # plt.axis('off')
    # plt.show()

    # # Custom homografisi ile mosaicking
    # mosaic_custom = create_mosaic_color(img_color1, img_color2, H_ransac)

    # # Görselleştir
    # plt.figure(figsize=(10, 5))
    # plt.title("Custom RANSAC Homography Mosaic")
    # plt.imshow(mosaic_custom, cmap='gray')
    # plt.axis('off')
    # plt.show()



   
    
    
    # Test için farklı bir resme ait homografi matrisi. 
    H_bad = np.array([[1.10043962e+00, 5.82166400e-02, -2.88056455e+01],
                        [2.90733756e-02, 1.10205966e+00, -2.18930460e+01],
                        [8.17233366e-05, 1.15504814e-04, 1.00000000e+00]])
    
 

    stitch_images(img1, img2,H_opencv,H_optimized)

    # # Save and display the result
    # cv2.imwrite('stitched_image.jpg', result)
    # cv2.imshow('Stitched Image', result)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    
    
    


    


if __name__ == "__main__":
    main()
