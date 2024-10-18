import cv2
import numpy as np
import pywt
from scipy.fft import dct, idct

def compute_texture_map(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude

#trova zone molto chiare o molto scure
def compute_bright_dark_areas(image):
    dark_factor=0.6 #diminuisci per più restrittivo su scuro
    bright_factor=1.4 #auemnta per più restrittivo su chiaro
    threshold = np.mean(image)
    dark_threshold=threshold*dark_factor
    bright_threshold=threshold*bright_factor
    dark_areas = np.argwhere(image < dark_threshold)
    bright_areas = np.argwhere(image >= bright_threshold)
    return dark_areas, bright_areas

#trova contorni
def compute_contours(image):
    """ Calcola i contorni dell'immagine utilizzando il filtro di Canny. """
    edges = cv2.Canny(image, 200, 600)  #aumenta valori per più restrittivo -> secondo deve essere 3x il primo
    return np.argwhere(edges > 0) 

# WPSNR function
def wpsnr(img1, img2):
    img1 = np.float32(img1) / 255.0
    img2 = np.float32(img2) / 255.0
    diff = img1 - img2
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 9999999  # No difference
    max_pixel = 1.0
    return 20 * np.log10(max_pixel / np.sqrt(mse))

# Function to extract watermark based on DWT and significant areas
def extract_watermark(image, watermarked_image, alpha=0.2, texture_threshold=0.3):
    # Apply DWT to both images
    coeffs_image = pywt.dwt2(image, 'haar')
    LL_image, (LH_image, HL_image, HH_image) = coeffs_image
    coeffs_watermarked = pywt.dwt2(watermarked_image, 'haar')
    LL_watermarked, (LH_watermarked, HL_watermarked, HH_watermarked) = coeffs_watermarked

    # Identify significant areas (texture, bright, dark, contours)
    texture_map = compute_texture_map(image)
    dark_areas, bright_areas = compute_bright_dark_areas(image)
    contour_areas = compute_contours(image)
    
    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255
    texture_areas = np.argwhere(texture_mask == 255)

    # Union of all significant areas
    significant_areas = set(map(tuple, np.vstack((bright_areas, dark_areas, contour_areas, texture_areas))))

    # Extract the watermark from the LH coefficients in the significant areas
    watermark_extracted = np.zeros_like(LH_image)
    abs_LH = np.abs(LH_image)

    for loc in np.argwhere(abs_LH > 0):
        if tuple(loc) in significant_areas:
            watermark_extracted[tuple(loc)] = (LH_watermarked[tuple(loc)] / LH_image[tuple(loc)] - 1) / alpha
    
    return watermark_extracted.flatten()

# Function to calculate similarity between two extracted watermarks
def calculate_similarity(watermark1, watermark2):
    similarity_score = np.sum(watermark1 * watermark2) / (
        np.sqrt(np.sum(watermark1 ** 2)) * np.sqrt(np.sum(watermark2 ** 2))
    )
    return similarity_score

# Main detection function
def detection(input1, input2, input3):
    # Load the original, watermarked, and attacked images
    original_img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked_img = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked_img = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None or watermarked_img is None or attacked_img is None:
        raise ValueError("One or more images could not be loaded.")
    
    # Parameters for watermarking
    alpha = 0.2
    
    # Extract watermarks from the watermarked and attacked images using the updated extraction method
    watermark_extracted_w = extract_watermark(original_img, watermarked_img, alpha)
    watermark_extracted_a = extract_watermark(original_img, attacked_img, alpha)
    
    # Calculate similarity using the new function
    similarity_score = calculate_similarity(watermark_extracted_w, watermark_extracted_a)
    
    # Predefined threshold τ
    tau = 0.2  
    
    # Compute WPSNR between watermarked and attacked images
    wpsnr_value = wpsnr(watermarked_img, attacked_img)
    
    # Detection conditions based on the provided rules
    if similarity_score >= tau:
        output1 = 1  # Watermark detected in attacked image
    else:
        output1 = 0  # Watermark destroyed in attacked image
    
    return output1, wpsnr_value