import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
import os

def awgn(img, std, seed):
    mean = 0.0
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    return attacked

def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked

def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked

def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked

def resizing(img, scale):
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    return attacked

def jpeg_compression(img, QF):
    img_pil = Image.fromarray(img)
    img_pil=img_pil.convert('L')
    img_pil.save('tmp.jpg', "JPEG", quality=QF)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8)
    os.remove('tmp.jpg')
    return attacked

# Funzione principale di attacco
def attacks(image, attack_name, param_array):
    
    attacked_image = image.copy()
    
    #check if there is one or more attack
    if isinstance(attack_name, str):
        attack_name = [attack_name]
        param_array = [param_array]
    
    # perform the attacks in the input
    for attack, params in zip(attack_name, param_array):
        if attack == 'awgn':
            std, seed = params
            attacked_image = awgn(attacked_image, std, seed)
        
        elif attack == 'blur':
            sigma = params
            attacked_image = blur(attacked_image, sigma)
        
        elif attack == 'sharpening':
            sigma, alpha = params
            attacked_image = sharpening(attacked_image, sigma, alpha)
        
        elif attack == 'median':
            kernel_size = params
            attacked_image = median(attacked_image, kernel_size)
        
        elif attack == 'resize':
            scale = params
            attacked_image = resizing(attacked_image, scale)
        
        elif attack == 'jpeg':
            QF = params
            attacked_image = jpeg_compression(attacked_image, QF)
    
    return attacked_image
