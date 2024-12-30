import numpy as np
import random

def fit_circle(points):
    """
    Rastgele seçilen 3 nokta ile çember uydurur.
    
    Args:
        points (np.ndarray): [x, y] formatında noktalar.
    
    Returns:
        tuple: (h, k, r) çemberin merkezi ve yarıçapı.
    """
    x1, y1 = points[0]
    x2, y2 = points[1]
    x3, y3 = points[2]
    # Çember merkezini ve yarıçapını bulmak için determinantlar
    A = np.array([
        [x1, y1, 1],
        [x2, y2, 1],
        [x3, y3, 1]
    ])
    D = np.linalg.det(A)
    if D == 0:
        raise ValueError("Seçilen noktalar bir çember tanımlamıyor.")

    B = np.array([
        [x1**2 + y1**2, y1, 1],
        [x2**2 + y2**2, y2, 1],
        [x3**2 + y3**2, y3, 1]
    ])
    C = np.array([
        [x1**2 + y1**2, x1, 1],
        [x2**2 + y2**2, x2, 1],
        [x3**2 + y3**2, x3, 1]
    ])
    D = np.array([
        [x1**2 + y1**2, x1, y1],
        [x2**2 + y2**2, x2, y2],
        [x3**2 + y3**2, x3, y3]
    ])

    h = 0.5 * np.linalg.det(B) / np.linalg.det(A)
    k = -0.5 * np.linalg.det(C) / np.linalg.det(A)
    r = np.sqrt(h**2 + k**2 + np.linalg.det(D) / np.linalg.det(A))
    return h, k, r


def circle_error(point, params):
    """
    Bir noktanın çembere olan mesafesini hesaplar.
    
    Args:
        point (np.ndarray): [x, y] formatında bir nokta.
        params (tuple): (h, k, r) çemberin merkezi ve yarıçapı.
    
    Returns:
        float: Noktanın çembere olan mesafesi.
    """
    x, y = point
    h, k, r = params
    return abs(np.sqrt((x - h)**2 + (y - k)**2) - r)


