import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from scipy.fft import dct, idct
import random
import glob
import numpy as np
import cv2
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

# Funzione per calcolare la similarità normalizzata
def similarity(X, X_star):
    # Calcola la somiglianza normalizzata tra il watermark originale e quello estratto
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s



# Embedding function 
def embedding(input1, input2):
    # open image
    #image = cv2.imread(input1, cv2.IMREAD_GRAYSCALE)
    image = input1

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




def detection(input_image, watermarked_image, alpha, watermark_size):
    # Carica l'immagine originale e quella con watermark
    original_image = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError(f"Immagine non trovata: {input_image}")
    
    # Calcola la DCT dell'immagine originale
    dct_original = dct(dct(original_image, axis=0, norm='ortho'), axis=1, norm='ortho')
    
    # Calcola la DCT dell'immagine watermarked
    dct_watermarked = dct(dct(watermarked_image, norm='ortho'), axis=1, norm='ortho')

    # Trova i coefficienti DCT più significativi nell'immagine originale
    dct_abs = np.abs(dct_original)
    locations = np.argsort(-dct_abs, axis=None)
    rows, cols = dct_original.shape
    locations = [(val // cols, val % cols) for val in locations]

    # Estrai il watermark dall'immagine watermarked
    extracted_watermark = np.zeros(watermark_size, dtype=np.float64)

    # Recupera il watermark dai coefficienti DCT più significativi
    for i, loc in enumerate(locations[:watermark_size]):
        extracted_watermark[i] = (dct_watermarked[loc] - dct_original[loc]) / (alpha * dct_original[loc])

    return extracted_watermark

random.seed(3)
def awgn(img, std, seed):
  mean = 0.0   # some constant
  #np.random.seed(seed)
  attacked = img + np.random.normal(mean, std, img.shape)
  attacked = np.clip(attacked, 0, 255)
  return attacked

def blur(img, sigma):
  from scipy.ndimage.filters import gaussian_filter
  attacked = gaussian_filter(img, sigma)
  return attacked

def sharpening(img, sigma, alpha):
  import scipy
  from scipy.ndimage import gaussian_filter
  import matplotlib.pyplot as plt

  #print(img/255)
  filter_blurred_f = gaussian_filter(img, sigma)

  attacked = img + alpha * (img - filter_blurred_f)
  return attacked

def median(img, kernel_size):
  from scipy.signal import medfilt
  attacked = medfilt(img, kernel_size)
  return attacked

def resizing(img, scale):
  from skimage.transform import rescale
  x, y = img.shape
  attacked = rescale(img, scale)
  attacked = rescale(attacked, 1/scale)
  attacked = attacked[:x, :y]
  return attacked

def jpeg_compression(img, QF):
  from PIL import Image
  img = Image.fromarray(img)
  img.save('tmp.jpg',"JPEG", quality=QF)
  attacked = Image.open('tmp.jpg')
  attacked = np.asarray(attacked,dtype=np.uint8)
  os.remove('tmp.jpg')

  return attacked


# Funzione per eseguire attacchi casuali
def random_attack(img):
    # Simula un attacco casuale sull'immagine
    i = random.randint(1, 7)
    if i == 1:
        attacked = awgn(img, 3., 123)
    elif i == 2:
        attacked = blur(img, [3, 3])
    elif i == 3:
        attacked = sharpening(img, 1, 1)
    elif i == 4:
        attacked = median(img, [3, 3])
    elif i == 5:
        attacked = resizing(img, 0.8)
    elif i == 6:
        attacked = jpeg_compression(img, 75)
    elif i == 7:
        attacked = img  # Nessun attacco
    return attacked

# Funzione principale per stimare la soglia
def compute_similarity_threshold( num_images=101, attack_func=random_attack):
    mark_size = 1024
    alpha = 0.1
    v = 'multiplicative'
    np.random.seed(seed=124)

    # Inizializza array per i punteggi e le etichette
    scores = []
    labels = []

    # Elenca i file BMP dalla cartella
    images = sorted(glob.glob('Multimedia-Data-Security/sample_images/*.bmp'))
    print(images)
    # Loop su ciascuna immagine
    for img_path in images:
        image = cv2.imread(img_path, 0)
        
        # Incorporare Watermark
        watermarked = embedding(image, 'mark.npy')

        sample = 0
        while sample < 10:
            # Genera un watermark casuale (H0)
            fakemark = np.random.uniform(0.0, 1.0, mark_size)
            fakemark = np.uint8(np.rint(fakemark))
            
            # Applica un attacco casuale all'immagine con watermark
            res_att = random_attack(watermarked)
            
            # Estrai il watermark attaccato e l'originale
            wat_attacked = detection(img_path, res_att, alpha, mark_size)
            wat_extracted = detection(img_path, watermarked, alpha, mark_size)
            
            # Calcola la similarità H1 (watermark estratto vs attaccato)
            scores.append(similarity(wat_extracted, wat_attacked))
            labels.append(1)
            
            # Calcola la similarità H0 (fake watermark vs attaccato)
            scores.append(similarity(fakemark, wat_attacked))
            labels.append(0)

            sample += 1

    # Stampa i risultati
    print('Array dei punteggi (scores): ', scores)
    print('Array delle etichette (labels): ', labels)

    # Calcola la curva ROC
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Scegli la soglia ottimale (τ) dove FPR ∈ [0, 0.1]
    fpr_limit = 0.1
    valid_thresholds = thresholds[fpr <= fpr_limit]
    best_threshold = valid_thresholds[-1]
    print('Miglior soglia τ per FPR ∈ [0, 0.1]:', best_threshold)

    # Salva la curva ROC
    plt.plot(fpr, tpr, label='Curva ROC')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    plt.show()
    # Path alla cartella con immagini BMP




# Calcola la soglia di similarità (τ)
best_tau = compute_similarity_threshold()

# Stampa la soglia per uso futuro
print(best_tau)