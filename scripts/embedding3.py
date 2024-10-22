import pywt
import numpy as np
import cv2
from scipy.ndimage import gaussian_gradient_magnitude, generic_filter

def embedding3(image_path, mark):
    levels = 2
    alpha = 0.5
    image = cv2.imread(image_path, 0)

    coeffs2 = pywt.dwt2(image, 'haar')
    LL, (LH, HL, HH) = coeffs2
    coeffs_list = []

    for level in range(levels):
        print(f"Embedding at level {level + 1}")
        
        mask = calculate_perceptual_mask(LL, LH, HL, HH)
        sign_LH, watermarked_LH = embed_watermark(LH, mark, alpha, mask)
        sign_HL, watermarked_HL = embed_watermark(HL, mark, alpha, mask)
        sign_HH, watermarked_HH = embed_watermark(HH, mark, alpha, mask)

        coeffs_list.append((watermarked_LH, watermarked_HL, watermarked_HH))

        if level < levels - 1:
            LL, (LH, HL, HH) = pywt.dwt2(LL, 'haar')  

    for level in range(levels - 1, -1, -1):
        watermarked_LH, watermarked_HL, watermarked_HH = coeffs_list[level]
        LL = pywt.idwt2((LL, (watermarked_LH, watermarked_HL, watermarked_HH)), 'haar')

    return np.clip(LL, 0, 255).astype(np.uint8)

def calculate_perceptual_mask(LL, LH, HL, HH):
    texture_sensitivity = calculate_texture_sensitivity(LL, LH, HL, HH)
    edge_sensitivity = calculate_edge_sensitivity(LL, LH, HL, HH)

    mask = texture_sensitivity * edge_sensitivity
    mask = mask / np.max(mask)

    return mask

def calculate_texture_sensitivity(LL, LH, HL, HH):
    texture = np.abs(LH) + np.abs(HL) + np.abs(HH)
    return texture / np.max(texture)

def calculate_edge_sensitivity(LL, LH, HL, HH):
    edges = gaussian_gradient_magnitude(LL, sigma=3)
    return 1 / (1 + edges)

def embed_watermark(subband, mark, alpha, mask):
    sign_subband = np.sign(subband)
    abs_subband = np.abs(subband)
    sorted_indices = np.argsort(-abs_subband, axis=None)
    flat_subband = abs_subband.flatten()

    for i, idx in enumerate(sorted_indices[:len(mark)]):
        flat_subband[idx] *= (1 + alpha * mark[i] * mask.flat[idx])

    abs_subband = flat_subband.reshape(subband.shape)
    watermarked_subband = abs_subband * sign_subband

    return sign_subband, watermarked_subband
