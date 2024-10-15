import numpy as np
import cv2
import pywt
from scipy.signal import convolve2d
from math import sqrt
from skimage.metrics import peak_signal_noise_ratio as psnr

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

def similarity(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def detection(original_image, watermarked_image, attacked_image):
    """ Rileva la presenza del watermark confrontando l'immagine watermarkata e quella attaccata. """
    levels=2

    #dwt su watermarked e attacked
    watermarked_coeffs = pywt.wavedec2(watermarked_image, 'haar', level=levels)
    attacked_coeffs = pywt.wavedec2(attacked_image, 'haar', level=levels)

    similarity_scores = []

    w_LL = watermarked_coeffs[0]
    a_LL = attacked_coeffs[0]

    #for every level
    for level in range(1, levels + 1):
        w_LH, w_HL, w_HH = watermarked_coeffs[level]
        a_LH, a_HL, a_HH = attacked_coeffs[level]

        w_LH_flat = w_LH.flatten()
        a_LH_flat = a_LH.flatten()

        #compute similarity for the level
        sim_value = similarity(w_LH_flat, a_LH_flat)
        similarity_scores.append(sim_value)

    #average similarity
    avg_similarity = np.mean(similarity_scores)
    
    tau = 0.85  #from ROC

    # check if watermark is present
    if avg_similarity >= tau:
        output1 = 1  
    else:
        output1 = 0 

    # WPSNR between watermarked and attacked
    wpsnr_value = wpsnr(watermarked_image, attacked_image)
    output2 = wpsnr_value

    return output1, output2

