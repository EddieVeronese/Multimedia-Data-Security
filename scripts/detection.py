import pywt
import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt

#trova zone con texture
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

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def wpsnr(img1, img2):
  img1 = np.float32(img1)/255.0
  img2 = np.float32(img2)/255.0
  difference = img1-img2
  same = not np.any(difference)
  if same is True:
      return 150
  w = np.genfromtxt('csf.csv', delimiter=',')
  ew = convolve2d(difference, np.rot90(w,2), mode='valid')
  decibels = 20.0*np.log10(1.0/sqrt(np.mean(np.mean(ew**2))))
  return decibels



def extract_watermark(original_image, test_image):
    # DWT dell'immagine originale
    levels=2
    alpha=0.2
    v='multiplicative'
    texture_threshold=0.3

    coeffs2 = pywt.dwt2(original_image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Calcolo delle aree significative
    texture_map = compute_texture_map(original_image)
    dark_areas, bright_areas = compute_bright_dark_areas(original_image)
    contour_areas = compute_contours(original_image)
    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(original_image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255  # Soglia texture
    texture_areas = np.argwhere(texture_mask == 255)
    significant_areas = set(map(tuple, np.vstack((bright_areas, dark_areas, contour_areas, texture_areas))))

    # DWT dell'immagine di test
    test_coeffs2 = pywt.dwt2(test_image, 'haar')
    LL_test, (LH_test, HL_test, HH_test) = test_coeffs2

    # Estrarre watermark confrontando LH dell'immagine originale e quella di test
    extracted_mark = []
    
    for level in range(levels):
        sign_LH_test = np.sign(LH_test)
        abs_LH_test = abs(LH_test)
        locations_LH_test = np.argsort(-abs_LH_test, axis=None)
        rows_LH_test = LH_test.shape[0]
        locations_LH_test = [(val // rows_LH_test, val % rows_LH_test) for val in locations_LH_test]
        
        for loc in locations_LH_test[1:]:
            if tuple(loc) in significant_areas:
                if v == 'additive':
                    extracted_val = (LH_test[loc] - LH[loc]) / alpha
                elif v == 'multiplicative':
                    extracted_val = (LH_test[loc] / LH[loc] - 1) / alpha
                extracted_mark.append(extracted_val)

        # Aggiorna DWT per livello successivo
        if level < levels - 1:
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')
            LL_test, (LH_test, HL_test, HH_test) = pywt.dwt2(LL_test, 'haar')

    extracted_mark = np.round(extracted_mark).astype(int)
    return extracted_mark



def detection(original_image, watermarked_image, attacked_image):  

    # estraggo watermark da watermarked_image
    w_extracted = extract_watermark(original_image, watermarked_image)
    
    # estraggo watermark da attacked_image
    w_attacked = extract_watermark(original_image, attacked_image)
    
    # calcolo similarità
    sim = similarity(w_extracted, w_attacked)
    
    # calcolo WPSNR tra watermarked e attacked
    wpsnr_value = wpsnr(watermarked_image, attacked_image)
    
    Tau = 0.75  # da ROC

    # controllo se watermark presente
    if sim >= Tau:
        output1 = 1  
    else:
        output1 = 0 

    output2 = wpsnr_value
    
    return output1, output2
