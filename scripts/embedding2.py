import pywt
import numpy as np
import cv2
from scipy.ndimage import gaussian_gradient_magnitude

def embedding2(image_path, mark):

    #definizione parametri
    levels = 2
    alpha = 1

    #leggo immagine
    image = cv2.imread(image_path, 0)  

    # applico prima dwt
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # memorizza coefficienti decomposizione
    coeffs_list = []

    # esegue per ogni livello
    for level in range(levels):
        ##print(f"Embedding at level {level + 1}")

        # calcola la maschera percettiva livello
        mask = calculate_perceptual_mask(LL, LH, HL, HH)

        # operazioni su LH
        sign_LH = np.sign(LH)
        abs_LH = abs(LH)
        locations_LH = np.argsort(-abs_LH,axis=None) 
        rows_LH = LH.shape[0]
        locations_LH = [(val//rows_LH, val%rows_LH) for val in locations_LH] 

        # operazioni su HL
        sign_HL = np.sign(HL)
        abs_HL = abs(HL)
        locations_HL = np.argsort(-abs_HL,axis=None) 
        rows_HL = HL.shape[0]
        locations_HL = [(val//rows_HL, val%rows_HL) for val in locations_HL]

        # operazioni su HH
        sign_HH = np.sign(HH)
        abs_HH = abs(HH)
        locations_HH = np.argsort(-abs_HH,axis=None) 
        rows_HH = HH.shape[0]
        locations_HH = [(val//rows_HH, val%rows_HH) for val in locations_HH] 
        
        # embed in multiplicative LH
        watermarked_LH = abs_LH.copy()
        for idx, (loc,mark_val) in enumerate(zip(locations_LH[1:], mark)):
            #watermarked_LH[loc] *= 1 + ( alpha * mark_val* mask[loc])
            watermarked_LH[loc] *= 1 + ( alpha * mark_val)
        
        # emend multiplicative in LH
        watermarked_HL = abs_HL.copy()
        for idx, (loc,mark_val) in enumerate(zip(locations_HL[1:], mark)):
            #watermarked_HL[loc] *= 1 + ( alpha * mark_val* mask[loc])
            watermarked_HL[loc] *= 1 + ( alpha * mark_val)
        
        # emned in multiplicative LH
        watermarked_HH = abs_HH.copy()
        for idx, (loc,mark_val) in enumerate(zip(locations_HH[1:], mark)):
            #watermarked_HH[loc] *= 1 + ( alpha * mark_val* mask[loc])
            watermarked_HH[loc] *= 1 + ( alpha * mark_val)

        # ritorna a dominio spaziale
        watermarked_LH *= sign_LH
        watermarked_HL *= sign_HL
        watermarked_HH *= sign_HH

        # memorizza coefficienti per livello
        coeffs_list.append((watermarked_LH, watermarked_HL, watermarked_HH))

        # decomponi ancora immagine
        if level < levels - 1:
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')  

    # ricostruisci immagine
    for level in range(levels - 1, -1, -1):

        watermarked_LH, watermarked_HL, watermarked_HH = coeffs_list[level]
        LL = pywt.idwt2((LL, (watermarked_LH, watermarked_HL, watermarked_HH)), 'haar')

    # riorna immagine finale con watermark
    return np.clip(LL, 0, 255).astype(np.uint8)

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

