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
    alpha=0.1 #più aumenta meno qualità
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
        threshold = 0.5 * np.max(abs_LH)
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



