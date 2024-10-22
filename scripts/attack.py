import numpy as np
import cv2
from scipy.ndimage import gaussian_filter
from scipy.signal import medfilt
from skimage.transform import rescale
from PIL import Image
import os

def awgn(img, std, seed, option = 0):
    mean = 0.0
    np.random.seed(seed)
    attacked = img + np.random.normal(mean, std, img.shape)
    attacked = np.clip(attacked, 0, 255)
    attacked = apply_mask(img, attacked, option)
    return attacked

def blur(img, sigma, option = 0):
    attacked = gaussian_filter(img, sigma)
    attacked = apply_mask(img, attacked, option)
    return attacked

def sharpening(img, sigma, alpha, option = 0):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    attacked = apply_mask(img, attacked, option)
    return attacked

def median(img, kernel_size, option):
    attacked = medfilt(img, kernel_size)
    attacked = apply_mask(img, attacked, option)
    return attacked

def resizing(img, scale, option):
    x, y = img.shape
    attacked = rescale(img, scale)
    attacked = rescale(attacked, 1/scale)
    attacked = attacked[:x, :y]
    attacked = apply_mask(img, attacked, option)
    return attacked

def jpeg_compression(img, QF, option):
    img_pil = Image.fromarray(img)
    img_pil=img_pil.convert('L')
    img_pil.save('tmp.jpg', "JPEG", quality=QF)
    attacked = Image.open('tmp.jpg')
    attacked = np.asarray(attacked, dtype=np.uint8)
    attacked = apply_mask(img, attacked, option)
    os.remove('tmp.jpg')
    return attacked

def apply_mask(img, attacked, option):

    mask = np.zeros_like(img)
    th1, th2 = 20, 60 # Parameters for the canny detection

    if option == 0: #Apply filter to the entire image
        return attacked
    elif option == 1: #Mask of the border
        mask = compute_edges(img)
    
    elif option == 2:   #Mask the textured areas
        mask = compute_texture_map(img)

    elif option == 3: #Mask in bright areas
        mask = compute_dark_areas(img)

    elif option == 4: #Mask on the dark areas
        mask = compute_bright_areas(img)

    return np.where(mask, attacked, img)


#Trova i contorni dell'immagine
def compute_edges(image):
    mask = np.zeros_like(image)
    th1, th2 = 20, 60 # Parameters for the canny detection

    img_scaled = np.uint8(image * 255)
    return cv2.Canny(img_scaled, th1, th2)

#Trova le zone con texture
def compute_texture_map(image):

    threshold = 0.5 #Threshold for the texture computation
    mask = np.zeros_like(image)

    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    texture_map = np.sqrt(sobel_x**2 + sobel_y**2)

    max_texture = np.max(texture_map)
    mask[texture_map >= (threshold * max_texture)] = 255

    return mask

#trova zone molto scure
def compute_dark_areas(image):
    mask = np.zeros_like(image)
    dark_factor=0.6 #diminuisci per più restrittivo su scuro

    threshold = np.mean(image)
    dark_threshold=threshold*dark_factor
    dark_areas = np.argwhere(image < dark_threshold)
    mask[tuple(dark_areas.T)] = 255
    return mask

#trova zone molto chiare
def compute_bright_areas(image):
    mask = np.zeros_like(image)
    bright_factor=1.4 #auemnta per più restrittivo su chiaro
    threshold = np.mean(image)

    bright_threshold=threshold*bright_factor
    bright_areas = np.argwhere(image >= bright_threshold)
    mask[tuple(bright_areas.T)] = 255

    return mask

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
            std, seed, option = params + [0] * (3 - len(params))
            attacked_image = awgn(attacked_image, std, seed, option)
        
        elif attack == 'blur':
            sigma, option = params + [0] * (3 - len(params))
            attacked_image = blur(attacked_image, sigma, option)
        
        elif attack == 'sharpening':
            sigma, alpha, option = params + [0] * (3 - len(params))
            attacked_image = sharpening(attacked_image, sigma, alpha, option)
        
        elif attack == 'median':
            kernel_size, option = params + [0] * (3 - len(params))
            attacked_image = median(attacked_image, kernel_size, option)
        
        elif attack == 'resize':
            scale, option = params + [0] * (3 - len(params))
            attacked_image = resizing(attacked_image, scale, option)
        
        elif attack == 'jpeg':
            QF, option = params + [0] * (3 - len(params))
            attacked_image = jpeg_compression(attacked_image, QF, option)
    
    return attacked_image
