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
    levels = 2  #same of embedding
    watermarked_coeffs = pywt.wavedec2(watermarked_image, 'haar', level=levels)
    attacked_coeffs = pywt.wavedec2(attacked_image, 'haar', level=levels)

    w_LL, w_LH, w_HH = watermarked_coeffs[-1]
    a_LL, a_LH, a_HH = attacked_coeffs[-1]

    w_LH_flat = w_LH.flatten()
    a_LH_flat = a_LH.flatten()

    #calculate similarity
    sim_value = similarity(w_LH_flat, a_LH_flat)
    
    tau = 0.85  #from the ROC

    #check if watermark present
    if sim_value >= tau:
        output1 = 1  
    else:
        output1 = 0  

    # calculate WPSNR between watermarkd and attacked
    wpsnr_value = wpsnr(watermarked_image, attacked_image)
    output2 = wpsnr_value

    return output1, output2

