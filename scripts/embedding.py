import cv2
import numpy as np
from scipy.fft import dct, idct

def embedding(input1, input2):
   
   #open image
    image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Immagine non trovata: {input1}")

    # open watermark
    watermark = np.load(input2)
    if watermark.shape != image.shape:
       watermark = cv2.resize(watermark, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

    # apply dct
    dct_image = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

    dct_abs = np.abs(dct_image)
    locations = np.argsort(-dct_abs, axis=None) 
    rows, cols = dct_image.shape
    locations = [(val // cols, val % cols) for val in locations] 

    watermarked_dct = dct_image.copy()

    alpha = 0.2

    # insert watermark
    for i, (loc, mark_val) in enumerate(zip(locations, watermark.flatten())):
        watermarked_dct[loc] *= (1 + alpha * mark_val) 

    # apply inverse DCT
    watermarked_image = idct(idct(watermarked_dct, axis=1, norm='ortho'), axis=0, norm='ortho')
    output1 = np.uint8(np.clip(watermarked_image, 0, 255)) 

    return output1