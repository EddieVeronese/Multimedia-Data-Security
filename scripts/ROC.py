import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from scipy.fft import dct, idct
import random

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

# Funzioni di attacco (già definite)
def random_attack(img):
    i = random.randint(1,7)
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
    else:
        attacked = img
    return attacked

from scipy.fft import dct, idct

def embedding(image, mark_size, alpha, v='multiplicative'):
    # Calcola la trasformata DCT bidimensionale dell'immagine
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Ottiene i coefficienti più significativi
    sign = np.sign(ori_dct)
    ori_dct = np.abs(ori_dct)
    locations = np.argsort(-ori_dct, axis=None)  # Ordine decrescente dei valori DCT
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]  # Coordinate dei coefficienti

    # Carica il watermark
    mark = np.random.uniform(0.0, 1.0, mark_size)
    mark = np.uint8(np.rint(mark))

    # Embedding del watermark nei coefficienti DCT
    watermarked_dct = ori_dct.copy()
    for idx, (loc, mark_val) in enumerate(zip(locations[1:], mark)):
        if v == 'additive':
            watermarked_dct[loc] += (alpha * mark_val)
        elif v == 'multiplicative':
            watermarked_dct[loc] *= 1 + (alpha * mark_val)

    # Ripristina il segno e ritorna all'immagine spaziale
    watermarked_dct *= sign
    watermarked = np.uint8(idct(idct(watermarked_dct, axis=1, norm='ortho'), axis=0, norm='ortho'))

    return mark, watermarked


def detection(image, watermarked, alpha, mark_size, v='multiplicative'):
    # Calcola la DCT dell'immagine originale e di quella watermarked
    ori_dct = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    wat_dct = dct(dct(watermarked, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Ordina i coefficienti per importanza percettiva
    ori_dct = np.abs(ori_dct)
    wat_dct = np.abs(wat_dct)
    locations = np.argsort(-ori_dct, axis=None)  # Ordine decrescente dei valori DCT
    rows = image.shape[0]
    locations = [(val // rows, val % rows) for val in locations]

    # Estrarre il watermark
    w_extracted = np.zeros(mark_size, dtype=np.float64)
    for idx, loc in enumerate(locations[1:mark_size + 1]):
        if v == 'additive':
            w_extracted[idx] = (wat_dct[loc] - ori_dct[loc]) / alpha
        elif v == 'multiplicative':
            w_extracted[idx] = (wat_dct[loc] - ori_dct[loc]) / (alpha * ori_dct[loc])

    return w_extracted


def similarity(X, X_star):
    # Calcola la similitudine tra due watermark
    s = np.sum(np.multiply(X, X_star)) / np.sqrt(np.sum(np.multiply(X_star, X_star)))
    return s


def calculate_threshold(images_folder, mark_file, mark_size=1024, alpha=0.1, v='multiplicative'):
    np.random.seed(124)  # Seed per riproducibilità

    # Carico il watermark salvato
    mark = np.load(mark_file)

    # Array per punteggi (scores) e etichette (labels)
    scores = []
    labels = []

    # Leggo tutte le immagini dalla cartella
    image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if f.endswith('.bmp')]

    # Embedding del watermark in ciascuna immagine
    for img_file in image_files:
        img = cv2.imread(img_file, 0)  # Legge l'immagine in scala di grigi

        # Applica l'embedding del watermark all'immagine
        _, watermarked_img = embedding(img, mark_size, alpha, v)

        # Ciclo per attaccare le immagini e calcolare la soglia
        for _ in range(10):  # Attacca ogni immagine 10 volte
            attacked_img = random_attack(watermarked_img)
            extracted_attacked = detection(img, attacked_img, alpha, mark_size, v)
            extracted_original = detection(img, watermarked_img, alpha, mark_size, v)

            # Similitudine tra watermark estratto e watermark originale (Ipotesi vera)
            scores.append(similarity(mark, extracted_attacked))
            labels.append(1)

            # Genera un watermark casuale (Ipotesi falsa)
            fake_mark = np.random.uniform(0.0, 1.0, mark_size)
            fake_mark = np.uint8(np.rint(fake_mark))
            scores.append(similarity(fake_mark, extracted_attacked))
            labels.append(0)

    # Calcolo della ROC e della soglia ottimale τ
    fpr, tpr, thresholds = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
    roc_auc = auc(fpr, tpr)

    # Selezionare la soglia che corrisponde a un FPR ∈ [0, 0.1]
    optimal_idx = np.where((fpr <= 0.1))[0][-1]
    tau_optimal = thresholds[optimal_idx]

    # Visualizzazione della curva ROC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

    return tau_optimal

