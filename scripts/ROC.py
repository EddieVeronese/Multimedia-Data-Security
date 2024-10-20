import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
import glob
import pywt
from scipy.ndimage import gaussian_filter

# Funzione per calcolare la mappa delle texture
def compute_texture_map(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude

# Funzione per trovare le aree scure e chiare
def compute_bright_dark_areas(image):
    dark_factor = 0.6
    bright_factor = 1.4
    threshold = np.mean(image)
    dark_threshold = threshold * dark_factor
    bright_threshold = threshold * bright_factor
    dark_areas = np.argwhere(image < dark_threshold)
    bright_areas = np.argwhere(image >= bright_threshold)
    return dark_areas, bright_areas

# Funzione per calcolare i contorni
def compute_contours(image):
    # Assicuriamoci che l'immagine sia di tipo uint8
    if image.dtype != np.uint8:
        image_uint8 = np.uint8(np.clip(image, 0, 255))
    else:
        image_uint8 = image
    edges = cv2.Canny(image_uint8, 200, 600)
    return np.argwhere(edges > 0)

# Funzione di embedding aggiornata
def embedding(image, mark):
    levels = 2  # Più aumenta, meno qualità
    alpha = 0.2  # Più aumenta, meno qualità
    v = 'multiplicative'
    texture_threshold = 0.3  # Soglia per la texture

    # Assicurati che l'immagine sia in scala di grigi e di tipo uint8
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = np.uint8(image)
    
    # Trova le zone significative
    texture_map = compute_texture_map(image)
    dark_areas, bright_areas = compute_bright_dark_areas(image)
    contour_areas = compute_contours(image)
    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255
    texture_areas = np.argwhere(texture_mask == 255)

    # Unisce le zone significative
    significant_areas = set(map(tuple, np.vstack((bright_areas, dark_areas, contour_areas, texture_areas))))

    # Esegue la DWT sull'immagine
    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2

    # Inizializza il contatore per i bit incorporati
    num_bits_embedded = 0

    # Ripete per ogni livello
    for level in range(levels):
        print(f"Embedding at level {level + 1}")

        sign_LH = np.sign(LH)
        abs_LH = np.abs(LH)
        locations_LH = np.argsort(-abs_LH, axis=None)
        rows_LH, cols_LH = LH.shape
        locations_LH = [(val // cols_LH, val % cols_LH) for val in locations_LH]

        # Inserisce il watermark solo nelle aree significative
        watermarked_LH = abs_LH.copy()
        mark_idx = 0 
        for loc in locations_LH:
            # Ridimensiona le coordinate per le aree significative
            scale_factor = 2 ** (level + 1)
            img_i, img_j = loc[0] * scale_factor, loc[1] * scale_factor
            if (img_i, img_j) in significant_areas:
                if mark_idx >= len(mark):
                    break
                mark_val = mark[mark_idx % len(mark)]
                if v == 'additive':
                    watermarked_LH[loc] += (alpha * mark_val)
                elif v == 'multiplicative':
                    watermarked_LH[loc] *= 1 + (alpha * mark_val)
                mark_idx += 1
                num_bits_embedded += 1

        watermarked_LH *= sign_LH

        # Ricostruisce LL usando l'IDWT
        LL = pywt.idwt2((LL, (watermarked_LH, HL, HH)), 'haar')

        # Aggiorna per il prossimo livello
        if level < levels - 1:
            coeffs2 = pywt.dwt2(LL, 'haar')
            LL, (LH, HL, HH) = coeffs2

    watermarked_image = np.uint8(np.clip(LL, 0, 255))
    return watermarked_image, num_bits_embedded

# Funzione di detection aggiornata
def detection(original_image, watermarked_image, num_bits_embedded, levels=2, alpha=0.2, texture_threshold=0.3, v='multiplicative'):
    # Assicuriamoci che le immagini siano in scala di grigi e di tipo float32
    if len(original_image.shape) == 3:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    original_image = np.float32(original_image)

    if len(watermarked_image.shape) == 3:
        watermarked_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2GRAY)
    watermarked_image = np.float32(watermarked_image)

    # Identifica le aree significative nell'immagine originale
    texture_map = compute_texture_map(original_image)
    dark_areas, bright_areas = compute_bright_dark_areas(original_image)
    contour_areas = compute_contours(original_image)

    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(original_image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255
    texture_areas = np.argwhere(texture_mask == 255)

    significant_areas = set(map(tuple, np.vstack((bright_areas, dark_areas, contour_areas, texture_areas))))

    # Esegue la DWT
    coeffs_image = []
    coeffs_watermarked = []

    current_image = original_image.copy()
    current_watermarked = watermarked_image.copy()

    for level in range(levels):
        coeffs_i = pywt.dwt2(current_image, 'haar')
        coeffs_w = pywt.dwt2(current_watermarked, 'haar')

        coeffs_image.append(coeffs_i)
        coeffs_watermarked.append(coeffs_w)

        current_image = coeffs_i[0]  # LL
        current_watermarked = coeffs_w[0]  # LL

    # Estrazione del watermark
    watermark_extracted = []

    for level in reversed(range(levels)):
        LL_i, (LH_i, HL_i, HH_i) = coeffs_image[level]
        LL_w, (LH_w, HL_w, HH_w) = coeffs_watermarked[level]

        # Ridimensiona le coordinate delle aree significative
        scale_factor = 2 ** (level + 1)
        significant_areas_dwt = set()
        for i, j in significant_areas:
            significant_areas_dwt.add((i // scale_factor, j // scale_factor))

        # Estrazione del watermark dai coefficienti LH
        bits_extracted = 0
        for (i, j) in significant_areas_dwt:
            if i < LH_i.shape[0] and j < LH_i.shape[1]:
                if bits_extracted >= num_bits_embedded:
                    break
                if v == 'multiplicative':
                    ratio = LH_w[i, j] / (LH_i[i, j] + 1e-5)
                    wm_val = (ratio - 1) / alpha
                elif v == 'additive':
                    diff = LH_w[i, j] - LH_i[i, j]
                    wm_val = diff / alpha
                watermark_extracted.append(wm_val)
                bits_extracted += 1

    # Converti in array NumPy e limita la dimensione
    watermark_extracted = np.array(watermark_extracted[:num_bits_embedded])
    return watermark_extracted

# Funzione per calcolare la similarità normalizzata
def similarity(X, X_star):
    # Normalizzazione dei vettori
    X_norm = (X - np.mean(X)) / np.std(X)
    X_star_norm = (X_star - np.mean(X_star)) / np.std(X_star)

    numerator = np.sum(X_norm * X_star_norm)
    denominator = np.sqrt(np.sum(X_norm ** 2)) * np.sqrt(np.sum(X_star_norm ** 2))
    if denominator == 0:
        return 0
    s = numerator / denominator
    return s

# Funzione di attacco fornita
def attack(img, sigma, th1, th2, n_layer):
    # Se l'immagine è in scala di grigi, convertila in BGR
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    redBlur = selective_blur(img[:,:,0], sigma, th1, th2)
    greenBlur = selective_blur(img[:,:,1], sigma, th1, th2)
    blueBlur = selective_blur(img[:,:,2], sigma, th1, th2)

    redBlur = layers_blur(redBlur, sigma, n_layer)
    greenBlur = layers_blur(greenBlur, sigma, n_layer)
    blueBlur = layers_blur(blueBlur, sigma, n_layer)

    # Unisci i tre canali
    result = cv2.merge([redBlur, greenBlur, blueBlur])

    # Applica un blur agli ultimi tre bit
    result = result & 254

    return result

def selective_blur(img, sigma, th1, th2):
    # Applica un blur all'immagine
    blurred = gaussian_filter(img, sigma)

    # Rilevamento dei bordi con Canny
    mask = canny_edge_detection(img, th1, th2).astype(bool)

    # Combina il blur con la maschera ottenuta dal rilevamento dei bordi
    return np.where(mask, blurred, img)

def canny_edge_detection(img, th1, th2):
    d = 3  # Blur gaussiano

    edgeresult = img.copy()
    edgeresult = cv2.GaussianBlur(edgeresult, (2*d+1, 2*d+1), -1)

    return cv2.Canny(edgeresult, th1, th2)

def layers_blur(img, sigma, n_layer):
    blurred = gaussian_filter(img, sigma)

    tot = 0
    for i in range(n_layer):
        tot += 2**i

    b2 = blurred.astype(np.uint8) & tot

    t2 = 255 - tot
    b3 = b2 | t2

    img = img & b3

    return img

# Funzione principale per stimare la soglia τ
def compute_similarity_threshold(num_images=101):
    alpha = 0.2
    levels = 2
    texture_threshold = 0.3
    v = 'multiplicative'
    np.random.seed(seed=124)

    # Inizializza array per i punteggi e le etichette
    scores = []
    labels = []

    # Elenca i file BMP dalla cartella
    images = sorted(glob.glob('Multimedia-Data-Security/sample_images/*.bmp'))
    print(images)
    # Limita il numero di immagini se necessario
    images = images[:num_images]

    # Seleziona una sola immagine per visualizzare i risultati
    image_sample_path = images[0] if images else None

    # Loop su ciascuna immagine
    for img_path in images:
        image = cv2.imread(img_path)
        if image is None:
            continue

        #Leggere il watermark
        mark = np.load('Multimedia-Data-Security/mark.npy')
        

        # Incorporare Watermark e ottenere il numero di bit incorporati
        watermarked, num_bits_embedded = embedding(image, mark)

        # Aggiorna la dimensione del watermark in base ai bit effettivamente incorporati
        mark = mark[:num_bits_embedded]

        sample = 0
        while sample < 10:
            # Genera un watermark falso della stessa dimensione
            fakemark = np.random.choice([-1, 1], size=num_bits_embedded)
            fakemark = (fakemark + 1) / 2

            # Applica l'attacco all'immagine con watermark
            res_att = attack(watermarked, sigma=1.0, th1=100, th2=200, n_layer=3)
            res_att = np.uint8(np.clip(res_att, 0, 255))

            # Estrai il watermark dall'immagine attaccata
            wat_attacked = detection(image, res_att, num_bits_embedded)
            # Estrai il watermark dall'immagine watermarked originale
            wat_extracted = detection(image, watermarked, num_bits_embedded)

            # Limita i watermark estratti alla dimensione corretta
            wat_extracted = wat_extracted[:num_bits_embedded]
            wat_attacked = wat_attacked[:num_bits_embedded]

            # Calcola la similarità H1 (watermark estratto vs attaccato)
            s1 = similarity(wat_extracted, wat_attacked)
            scores.append(s1)
            labels.append(1)

            # Calcola la similarità H0 (fake watermark vs attaccato)
            s0 = similarity(fakemark, wat_attacked)
            scores.append(s0)
            labels.append(0)

            sample += 1

            # Se questa è l'immagine campione, salviamo le immagini per visualizzazione
            if img_path == image_sample_path and sample == 1:
                original_image_sample = image
                watermarked_image_sample = watermarked
                attacked_image_sample = res_att

    # Calcola la curva ROC
    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # Traccia la curva ROC con AUC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()

    # Trova il punto sulla curva ROC per FPR ≈ 0.05
    target_fpr = 0.05
    fpr_diffs = fpr - target_fpr
    idx_tpr = np.where(fpr_diffs == min(i for i in fpr_diffs if i > 0))[0]
    if len(idx_tpr) > 0:
        idx = idx_tpr[0]
        print('Per un FPR ≈ 0.05, il TPR corrispondente è %0.2f' % tpr[idx])
        print('La soglia corrispondente è %0.2f' % thresholds[idx])
        print('FPR effettivo: %0.2f' % fpr[idx])
        best_threshold = thresholds[idx]
    else:
        print("Non è possibile trovare un FPR ≈ 0.05")
        best_threshold = thresholds[-1]

    return best_threshold

# Calcola la soglia di similarità (τ)
best_tau = compute_similarity_threshold()

# Stampa la soglia per uso futuro
print("Soglia τ calcolata:", best_tau)
