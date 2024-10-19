import cv2
import numpy as np
import pywt
from scipy.fft import dct, idct

def compute_texture_map(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude

def compute_bright_dark_areas(image):
    dark_factor = 0.6
    bright_factor = 1.4
    threshold = np.mean(image)
    dark_threshold = threshold * dark_factor
    bright_threshold = threshold * bright_factor
    dark_areas = np.argwhere(image < dark_threshold)
    bright_areas = np.argwhere(image >= bright_threshold)
    return dark_areas, bright_areas

def compute_contours(image):
    edges = cv2.Canny(image, 200, 600)
    return np.argwhere(edges > 0)

def wpsnr(img1, img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    diff = img1 - img2
    mse = np.mean(diff ** 2)
    if mse == 0:
        return 100  # Valore massimo del PSNR
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse))

def extract_watermark(image, watermarked_image, levels=2, alpha=0.2, texture_threshold=0.3, v='multiplicative'):
    image = np.float32(image)
    watermarked_image = np.float32(watermarked_image)

    # Identifica le aree significative nell'immagine originale
    texture_map = compute_texture_map(image)
    dark_areas, bright_areas = compute_bright_dark_areas(image)
    contour_areas = compute_contours(image)

    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255
    texture_areas = np.argwhere(texture_mask == 255)

    # Unione delle aree significative
    significant_areas = set(map(tuple, np.vstack((bright_areas, dark_areas, contour_areas, texture_areas))))

    # Inizializza la lista dei coefficienti DWT per i vari livelli
    coeffs_image = []
    coeffs_watermarked = []

    # Applica la DWT per ogni livello e memorizza i coefficienti
    current_image = image.copy()
    current_watermarked = watermarked_image.copy()
    for level in range(levels):
        coeffs_i = pywt.dwt2(current_image, 'haar')
        coeffs_w = pywt.dwt2(current_watermarked, 'haar')

        coeffs_image.append(coeffs_i)
        coeffs_watermarked.append(coeffs_w)

        current_image = coeffs_i[0]  # LL
        current_watermarked = coeffs_w[0]  # LL

    # Inizializza il watermark estratto
    watermark_extracted = []

    # Partendo dall'ultimo livello
    for level in reversed(range(levels)):
        LL_i, (LH_i, HL_i, HH_i) = coeffs_image[level]
        LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked[level]

        # Ridimensiona le coordinate delle aree significative per questo livello
        scale_factor = 2 ** (level + 1)
        significant_areas_dwt = set()
        for i, j in significant_areas:
            significant_areas_dwt.add((i // scale_factor, j // scale_factor))

        # Estrai il watermark dai coefficienti LH
        watermark_level = np.zeros_like(LH_i)

        # Evita divisioni per zero
        epsilon = 1e-5
        abs_LH_i = np.abs(LH_i) + epsilon

        for loc in np.argwhere(abs_LH_i > 0):
            loc_tuple = tuple(loc)
            if loc_tuple in significant_areas_dwt:
                if v == 'multiplicative':
                    ratio = LH_w[loc_tuple] / LH_i[loc_tuple]
                    watermark_level[loc_tuple] = (ratio - 1) / alpha
                elif v == 'additive':
                    diff = LH_w[loc_tuple] - LH_i[loc_tuple]
                    watermark_level[loc_tuple] = diff / alpha

        # Aggiungi il watermark estratto da questo livello alla lista
        watermark_extracted.append(watermark_level.flatten())

    # Combina i watermark estratti dai vari livelli
    watermark_extracted = np.concatenate(watermark_extracted)

    return watermark_extracted

def calculate_similarity(watermark1, watermark2):
    numerator = np.sum(watermark1 * watermark2)
    denominator = np.sqrt(np.sum(watermark1 ** 2)) * np.sqrt(np.sum(watermark2 ** 2))
    if denominator == 0:
        return 0
    similarity_score = numerator / denominator
    return similarity_score

def detection(input1, input2, input3):
    # Carica le immagini
    original_img = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    watermarked_img = cv2.imread(input2, cv2.IMREAD_GRAYSCALE)
    attacked_img = cv2.imread(input3, cv2.IMREAD_GRAYSCALE)
    
    if original_img is None or watermarked_img is None or attacked_img is None:
        raise ValueError("Una o più immagini non possono essere caricate.")
    
    # Parametri utilizzati durante l'embedding
    alpha = 0.2
    levels = 2
    texture_threshold = 0.3
    v = 'multiplicative'
    
    # Estrai i watermark
    watermark_original = extract_watermark(original_img, watermarked_img, levels, alpha, texture_threshold, v)
    watermark_attacked = extract_watermark(original_img, attacked_img, levels, alpha, texture_threshold, v)
    
    # Calcola la similarità
    similarity_score = calculate_similarity(watermark_original, watermark_attacked)
    
    # Soglia predefinita τ
    tau = 0.2  
    
    # Calcola il WPSNR tra l'immagine watermarked e quella attaccata
    wpsnr_value = wpsnr(watermarked_img, attacked_img)
    
    # Decisione sulla presenza del watermark
    if similarity_score >= tau:
        output1 = 1  # Watermark rilevato nell'immagine attaccata
    else:
        output1 = 0  # Watermark distrutto nell'immagine attaccata
    
    return output1, wpsnr_value
