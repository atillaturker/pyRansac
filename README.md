# Computer Vision Project

This project demonstrates the use of computer vision techniques to compute homographies and fit lines using RANSAC (Random Sample Consensus). The project includes various scripts for generating points, computing homographies, and visualizing results.

---

## Project Structure

- Code/ - generate_points.py  - homographyModel.py - homographyRansac.py LineFittingRansac.py ransac.py 
- images/


---

## Files Description

- **`generate_points.py`**: Contains functions to generate points with outliers for line fitting.
- **`homographyModel.py`**: Implements functions to compute homography matrices and calculate geometric distances.
- **`homographyRansac.py`**: Main script to compute homographies using RANSAC and OpenCV, and blend images using the computed homographies.
- **`LineFittingRansac.py`**: Script to fit lines using RANSAC and visualize the results.
- **`ransac.py`**: Contains the RANSAC algorithm implementation.
- **`test.py`**: Script to download datasets using Kaggle API.
- **`test2.py`**: Script to compute back projection error for homographies.
- **`test3.py`**: Script to compute and print back projection errors for OpenCV and custom RANSAC homographies.

---


