import pywt
import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt
from scipy.ndimage import gaussian_gradient_magnitude

def similaritys(X,X_star):
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



import cv2
import numpy as np
import pywt
from scipy.spatial.distance import cosine

def get_sorted_locations(band):
    """
    Restituisce le posizioni ordinate dei coefficienti della banda in ordine decrescente di magnitudine.
    """
    abs_band = abs(band)
    locations = np.argsort(-abs_band, axis=None)  # Ordinamento decrescente
    rows = band.shape[0]
    sorted_locations = [(val // rows, val % rows) for val in locations]  # Converti in coordinate 2D
    return sorted_locations

def extract_watermark(original_image, watermarked_image):
    """
    Rileva il watermark in un'immagine usando la DWT su più livelli e una maschera percettiva.
    Combina i watermark estratti da ciascun livello calcolando la media.
    """
    mark_size=1024
    levels=2
    alpha=1
    
    # Contenitore per i watermark estratti da ogni livello
    watermark_levels = []
    
    # Applica DWT all'immagine originale e all'immagine watermarked per ogni livello
    coeffs_original = pywt.dwt2(original_image, 'haar')
    LL_or, (LH_or, HL_or, HH_or) = coeffs_original

    coeffs_watermarked = pywt.dwt2(watermarked_image, 'haar')
    LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked

    for level in range(levels):
        ##print(f"Detection at level {level + 1}")

        # Calcola la maschera percettiva per il livello attuale
        mask = calculate_perceptual_mask(LL_or, LH_or, HL_or, HH_or)

        # Ottieni le posizioni ordinate delle bande
        locations_LH = get_sorted_locations(LH_or)
        locations_HL = get_sorted_locations(HL_or)
        locations_HH = get_sorted_locations(HH_or)

        # Estrai il watermark da ciascuna banda (LH, HL, HH)
        w_ex_LH = np.zeros(mark_size, dtype=np.float64)
        w_ex_HL = np.zeros(mark_size, dtype=np.float64)
        w_ex_HH = np.zeros(mark_size, dtype=np.float64)

        # Estrazione da LH
        for idx, loc in enumerate(locations_LH[1:mark_size+1]):
            #w_ex_LH[idx] = (LH_w[loc] - LH_or[loc]) / (alpha * mask[loc]*LH_or[loc])
            w_ex_LH[idx] = (LH_w[loc] - LH_or[loc]) / (alpha *LH_or[loc])
        
        # Estrazione da HL
        for idx, loc in enumerate(locations_HL[1:mark_size+1]):
            #w_ex_HL[idx] = (HL_w[loc] - HL_or[loc]) / (alpha * mask[loc]*HL_or[loc])
            w_ex_HL[idx] = (HL_w[loc] - HL_or[loc]) / (alpha *HL_or[loc])

        # Estrazione da HH
        for idx, loc in enumerate(locations_HH[1:mark_size+1]):
            #w_ex_HH[idx] = (HH_w[loc] - HH_or[loc]) / (alpha * mask[loc]*HH_or[loc])
            w_ex_HH[idx] = (HH_w[loc] - HH_or[loc]) / (alpha*HH_or[loc])

        # Combina i watermark estratti da LH, HL, e HH
        w_ex_level = (w_ex_LH + w_ex_HL + w_ex_HH) / 3

        # Aggiungi il watermark estratto al livello corrente alla lista dei watermark
        watermark_levels.append(w_ex_level)

        # Passa al livello successivo della decomposizione
        if level < levels - 1:
            LL_or, (LH_or, HL_or, HH_or) = pywt.dwt2(LL_or, 'haar')
            LL_w, (LH_w, HL_w, HH_w) = pywt.dwt2(LL_w, 'haar')

    # Calcola la media dei watermark estratti da tutti i livelli
    #final_watermark = sum(watermark_levels) / len(watermark_levels)

    final_watermark=watermark_levels
    return final_watermark


def detection(original, watermarked, attacked):

    
    # Extract watermarks from both the watermarked reference and the attacked image
    watermark_originals = extract_watermark(original, watermarked)
    watermark_attackeds = extract_watermark(original, attacked)
    
    # Initialize an empty list to store similarity scores
    similarities = []
    
    # Compute similarity for each level and store the results
    for watermark_original, watermark_attacked in zip(watermark_originals, watermark_attackeds):
        similarity = similaritys(watermark_original, watermark_attacked)
        similarities.append(similarity)
    
    # Select the best similarity score
    best_similarity = max(similarities)
    
    # Compute WPSNR value
    wpsnr_value = wpsnr(watermarked, attacked)
    
    # Define threshold for determining if the watermark is detected
    threshold_tau = 0.7
    
    # Determine if the watermark is considered present
    watermark_detected = 1 if best_similarity >= threshold_tau else 0
    
    return watermark_detected, wpsnr_value


def calculate_perceptual_mask(LL, LH, HL, HH):
    """
    Calcola la maschera percettiva tenendo conto della luminanza, texture e sensibilità ai bordi.
    """
    band_sensitivity = calculate_band_sensitivity(LH, HL, HH)
    luminance_sensitivity = calculate_luminance_sensitivity(LL)
    texture_sensitivity = calculate_texture_sensitivity(LH, HL, HH)

    # Combina le sensibilità per ottenere la maschera totale (Formula 9)
    mask = band_sensitivity * luminance_sensitivity * texture_sensitivity
    mask = mask / np.max(mask)  # Normalizza la maschera tra 0 e 1

    return mask

def calculate_band_sensitivity(LH, HL, HH):
    """
    Calcola la sensibilità alla banda in base alla frequenza e orientamento.
    """
    band_sensitivity = np.ones_like(LH)
    
    # Assegna sensibilità diversa per bande di diversa frequenza
    band_sensitivity[LH > 0] *= 0.8  # Orientamento orizzontale
    band_sensitivity[HL > 0] *= 0.7  # Orientamento verticale
    band_sensitivity[HH > 0] *= 0.6  # Orientamento diagonale

    return band_sensitivity

def calculate_luminance_sensitivity(LL):
    """
    Calcola la sensibilità alla luminanza basata sulla sottobanda LL (bassa frequenza).
    """
    luminance_sensitivity = np.ones_like(LL)
    luminance_sensitivity[LL > 128] = 0.5  # Aree molto luminose
    luminance_sensitivity[LL < 50] = 0.5   # Aree molto scure

    return luminance_sensitivity

def calculate_texture_sensitivity(LH, HL, HH):
    """
    Calcola la sensibilità alla texture basata sui coefficienti delle bande di dettaglio.
    """
    # Somma i valori assoluti dei coefficienti per catturare la forza della texture
    texture_map = np.abs(LH) + np.abs(HL) + np.abs(HH)
    
    # Aggiungi sensibilità ai bordi (gradiente locale)
    edge_sensitivity = 1 / (1 + gaussian_gradient_magnitude(texture_map, sigma=1))
    
    texture_sensitivity = texture_map * edge_sensitivity
    texture_sensitivity = texture_sensitivity / np.max(texture_sensitivity)  # Normalizza

    return texture_sensitivity
