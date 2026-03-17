import cv2
import numpy as np
from PIL import Image


def reduce_noise(image: np.ndarray) -> np.ndarray:
    '''Reduces noise by averages pixels with its neighbors in 3x3 grid.'''
    return cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)


def apply_clahe(image: np.ndarray) -> np.ndarray:
    '''Adjust brightness accordingly with respect to photo actuall brightness'''
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.merge([l, a, b])
    
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def preprocess(image_path: str) -> Image.Image:
    '''Preprocess the image into a numpy ndarray without noise.'''
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f'Image not found ar path: {image_path}')
    
    img = reduce_noise(img)
    img = apply_clahe(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return Image.fromarray(img)