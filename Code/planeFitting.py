import numpy as np
import random

def fit_plane(points):
    """
    Model Fonksiyonu
    Bir düzlem 𝑎x + 𝑏y + 𝑐z + 𝑑 = 0 denklemi ile ifade edilir. RANSAC'ta düzlemi bulmak için rastgele 3 nokta seçilir ve bu noktalardan bir düzlem denklemi hesaplanır.
    Rastgele seçilen 3 nokta ile düzlem uydurur.
    
    Args:
        points (np.ndarray): [x, y, z] formatında 3D noktalar.
    
    Returns:
        np.ndarray: [a, b, c, d] düzlem denklemi.
    """
    p1, p2, p3 = points
    # Vektörler oluştur
    v1 = p2 - p1
    v2 = p3 - p1
    # Normal vektör (çapraz çarpım)
    normal = np.cross(v1, v2)
    # Düzlem denklemi: ax + by + cz + d = 0
    a, b, c = normal
    d = -np.dot(normal, p1)  # d'yi bul
    return np.array([a, b, c, d])


def plane_error(point, plane):
    """
    Bir noktanın düzleme olan mesafesini hesaplar.
    Bir noktanın düzleme olan mesafesi şu formül ile hesaplanır: |ax + by + cz + d| / √(a^2 + b^2 + c^2)
    
    Args:
        point (np.ndarray): [x, y, z] formatında bir nokta.
        plane (np.ndarray): [a, b, c, d] düzlem denklemi.
    
    Returns:
        float: Noktanın düzleme olan mesafesi.
    """
    a, b, c, d = plane
    x, y, z = point
    return abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)



