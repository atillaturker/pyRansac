import numpy as np
import random


def fit_parabola(points):
    """
    Rastgele seçilen noktalarla parabol uydurur.
    
    Args:
        points (np.ndarray): [x, y] formatında noktalar.
    
    Returns:
        np.ndarray: [a, b, c] polinom katsayıları.
    """
    x = points[:, 0]
    y = points[:, 1]
    A = np.vstack([x**2, x, np.ones(len(x))]).T
    coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    return coeffs


def parabola_error(point, coeffs):
    """
    Bir noktanın eğriye olan mesafesini hesaplar.
    
    Args:
        point (np.ndarray): [x, y] formatında bir nokta.
        coeffs (np.ndarray): [a, b, c] polinom katsayıları.
    
    Returns:
        float: Noktanın eğriye olan mesafesi.
    """
    x, y = point
    y_estimate = coeffs[0] * x**2 + coeffs[1] * x + coeffs[2]
    return abs(y - y_estimate)
