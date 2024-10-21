import pywt
import os
from scipy.fft import dct, idct
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.signal import convolve2d
from math import sqrt


def embedding(image_path, mark):

    levels=2 #più aumenta meno qualità
    alpha=0.5 #più aumenta meno qualità
    v='multiplicative'

    image = cv2.imread(image_path, 0)

    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2


    #ripete per ogni livello
    for level in range(levels):
        print(f"Embedding at level {level + 1}")

        sign_LH = np.sign(LH)
        abs_LH = abs(LH)
        locations_LH = np.argsort(-abs_LH, axis=None)
        rows_LH = LH.shape[0]
        locations_LH = [(val // rows_LH, val % rows_LH) for val in locations_LH]

        # mette watermark ma solo in aree specifiche
        threshold = 0.6 * np.max(abs_LH)
        watermarked_LH = abs_LH.copy()

        mark_idx=0
        for i, loc in enumerate(locations_LH):
            if abs_LH[loc] > threshold:
                mark_val = mark[mark_idx % len(mark)]
                if v == 'additive':
                    watermarked_LH[loc] += (alpha * mark_val)
                elif v == 'multiplicative':
                    watermarked_LH[loc] *= 1 + (alpha * mark_val)
                mark_idx += 1

        watermarked_LH *= sign_LH
    
        LL = pywt.idwt2((LL, (watermarked_LH, HL, HH)), 'haar')

        # aggiorna livello
        if level < levels - 1:
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')

    watermarked_image = LL
    return watermarked_image



def plot_identified_areas(image):
   
    texture_threshold=0.3

    texture_map = compute_texture_map(image)
    dark_areas, bright_areas = compute_bright_dark_areas(image)
    contour_areas = compute_contours(image)

    dark_mask = np.zeros_like(image)
    bright_mask = np.zeros_like(image)
    contour_mask = np.zeros_like(image)

    dark_mask[tuple(dark_areas.T)] = 255
    bright_mask[tuple(bright_areas.T)] = 255
    contour_mask[tuple(contour_areas.T)] = 255

    max_texture = np.max(texture_map)
    texture_mask = np.zeros_like(image)
    texture_mask[texture_map >= (texture_threshold * max_texture)] = 255
    texture_areas = np.argwhere(texture_mask == 255)

    significant_areas = set(map(tuple, np.vstack((texture_areas, bright_areas, dark_areas, contour_areas))))

    significant_mask = np.zeros_like(image)
    for (i, j) in significant_areas:
        significant_mask[i, j] = 255

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 3, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(texture_mask, cmap='gray')
    plt.title(f'Textured Areas (Threshold={texture_threshold})')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(contour_mask, cmap='gray')
    plt.title('Contours')
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(dark_mask, cmap='gray')
    plt.title('Dark Areas')
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(bright_mask, cmap='gray')
    plt.title('Bright Areas')
    plt.axis('off')

    

    plt.subplot(2, 3, 6)
    plt.imshow(significant_mask, cmap='gray')
    plt.title('Significant Areas (Union of all)')
    plt.axis('off')

    plt.tight_layout()
    plt.show()