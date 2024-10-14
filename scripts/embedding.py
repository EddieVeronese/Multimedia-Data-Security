import numpy as np
import pywt
import cv2
import matplotlib.pyplot as plt

#trova zone con texture
def compute_texture_map(image):
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3) 
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    return magnitude

#trova zone molto chiare o molto scure
def compute_bright_dark_areas(image):
    dark_factor=0.6 #diminuisci per più restrittivo su scuro
    bright_factor=1.4 #auemnta per più restrittivo su chiaro
    threshold = np.mean(image)
    dark_threshold=threshold*dark_factor
    bright_threshold=threshold*bright_factor
    dark_areas = np.argwhere(image < dark_threshold)
    bright_areas = np.argwhere(image >= bright_threshold)
    return dark_areas, bright_areas

def compute_contours(image):
    """ Calcola i contorni dell'immagine utilizzando il filtro di Canny. """
    edges = cv2.Canny(image, 200, 600)  #aumenta valori per più restrittivo -> secondo deve essere 3x il primo
    return np.argwhere(edges > 0) 

def adaptive_multi_level_embedding(image_path, watermark):
    """ Inserisce un watermark nell'immagine utilizzando DWT in aree ad alta texture, contorni e aree scure/chiare. """
    alpha = 0.3 #più aumenti più embed, ma meno qualità
    v = 'multiplicative' #oppure additive
    levels=2 #più livelli metti più embed, ma meno qualità

    image = cv2.imread(image_path, 0)

    # texture map
    texture_map = compute_texture_map(image)
    texture_threshold = np.percentile(texture_map, 90)  # aumenta per più restrittivo -> max 100
    high_texture_locations = np.argwhere(texture_map >= texture_threshold)

    # area chiaro scuro
    dark_areas, bright_areas = compute_bright_dark_areas(image)

    # area contorni
    contour_locations = compute_contours(image)

    #unisci tutte le aree trovato
    all_selected_locations = np.vstack((high_texture_locations, dark_areas, bright_areas, contour_locations)) #togli quelle che vuoi per fare prove


    """decommenta per vedere le zone trovare dove applicare il watermark (esegui solo su un'immagine)"""
    """
    def visualize_locations(image, locations, title, color):
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        for loc in locations:
            y, x = loc  
            cv2.circle(image_color, (x, y), radius=3, color=color, thickness=-1) 
        return image_color

    image_high_texture = visualize_locations(image, high_texture_locations, "High Texture Areas", (0, 255, 0))
    image_dark_areas = visualize_locations(image, dark_areas, "Dark Areas", (255, 0, 0))
    image_bright_areas = visualize_locations(image, bright_areas, "Bright Areas", (0, 0, 255))
    image_contours = visualize_locations(image, contour_locations, "Contour Areas", (255, 255, 0))

    plt.figure(figsize=(12, 12))

    plt.subplot(2, 2, 1)
    plt.imshow(image_high_texture)
    plt.title("High Texture Areas")
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(image_dark_areas)
    plt.title("Dark Areas")
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(image_bright_areas)
    plt.title("Bright Areas")
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(image_contours)
    plt.title("Contour Areas")
    plt.axis('off')

    plt.tight_layout()
    plt.show() 
    """

    """finisce qui"""

    # DWT multi livello
    current_image = image.copy()
    
    for level in range(levels):
        coeffs = pywt.dwt2(current_image, 'haar')
        LL, (LH, HL, HH) = coeffs
        
        # maschera percettiva -> cerca cosa è?
        mask_LH = np.abs(LH) / np.max(np.abs(LH))

        # Embed in LH
        watermarked_LH = np.abs(LH).copy()

        min_length = min(len(watermark), len(all_selected_locations))
        
        for idx in range(min_length):
            loc = all_selected_locations[idx]
            mark_val = watermark[idx]
            
            if loc[0] < LH.shape[0] and loc[1] < LH.shape[1]:
                if v == 'additive':
                    watermarked_LH[loc[0], loc[1]] += (alpha * mark_val * mask_LH[loc[0], loc[1]])
                elif v == 'multiplicative':
                    watermarked_LH[loc[0], loc[1]] *= (1 + (alpha * mark_val * mask_LH[loc[0], loc[1]]))

        watermarked_LH *= np.sign(LH)
        coeffs_watermarked = (LL, (watermarked_LH, HL, HH))
        
        # ricostruisco immagine per livello dopo
        current_image = pywt.idwt2(coeffs_watermarked, 'haar')

    return current_image





