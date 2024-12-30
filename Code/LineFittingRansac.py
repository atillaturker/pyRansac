import cv2
import numpy as np
import random
from ransac import RANSAC

import numpy as np
import matplotlib.pyplot as plt
from LineFittingModel import fit_line, line_error, openCv_Fit_line, line_to_slope_intercept , visualize_points, generate_points_with_outliers



points_map = generate_points_with_outliers(num_points=100, outlier_ratio=0.3)

customRansacLine, inliers = RANSAC(
    points_map,
    fit_line,
    line_error,
    sample_size=2,
    threshold=0.01,
    num_iters=100,
    random_seed=42,
    verbose=False
)

customRansacSlope, customRansacIntercept = line_to_slope_intercept(customRansacLine)
openCV_slope, openCV_intercept = openCv_Fit_line(points_map)


visualize_points(points_map, slope=openCV_slope, intercept=openCV_intercept, additional_slope=customRansacSlope, additional_intercept=customRansacIntercept)


