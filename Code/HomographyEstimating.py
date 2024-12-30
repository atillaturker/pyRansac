import cv2
import numpy as np
import cv2
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import random


def stitch_images(img1, img2,HOpenCV,HCustom):
    # Birinci görüntüyü homografiyle dönüştürün
    height, width, channels = img2.shape

    print("OpenCV Homography Matrix:\n", HOpenCV)
    warped_img1_opencv = cv2.warpPerspective(img1, HOpenCV, (width, height))
    warped_img1_custom = cv2.warpPerspective(img1, HCustom, (width, height))

    # Görüntüleri birleştirin
    stitched_img_opencv = np.maximum(warped_img1_opencv, img2)
    stitched_img_custom = np.maximum(warped_img1_custom, img2)

    # Görüntüleri yan yana göster
    plt.figure(figsize=(20, 15))

    plt.subplot(1, 2, 1)
    plt.title("OpenCV Homografi ile")
    plt.imshow(cv2.cvtColor(stitched_img_opencv, cv2.COLOR_BGR2RGB))

    plt.subplot(1, 2, 2)
    plt.title("Custom Homografi ile")
    plt.imshow(cv2.cvtColor(stitched_img_custom, cv2.COLOR_BGR2RGB))

    plt.show()