import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import random
from embedding import embedding
from detection import detection
from detection import similarity

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

def random_attack(img):
  i = random.randint(1,7)
  if i==1:
    attacked = awgn(img, 3., 123)
  elif i==2:
    attacked = blur(img, [3, 3])
  elif i==3:
    attacked = sharpening(img, 1, 1)
  elif i==4:
    attacked = median(img, [3, 3])
  elif i==5:
    attacked = resizing(img, 0.8)
  elif i==6:
    attacked = jpeg_compression(img, 75)
  elif i ==7:
     attacked = img
  return attacked

# Parametri generali
mark_size = 1024
alpha = 0.1
v = 'multiplicative'
np.random.seed(seed=124)

# Percorso alla cartella immagini
image_folder = './../images/'

# Inizializza gli array per score e label
scores = []
labels = []

# Lista di nomi file delle immagini numerate da 001 a 100
image_filenames = [f'{i:03}.bmp' for i in range(1, 101)]  # 001.bmp, 002.bmp, ..., 100.bmp

sample = 0
while sample < 500:  # Numero totale di iterazioni
    for filename in image_filenames:
        # Percorso completo all'immagine
        im_path = os.path.join(image_folder, filename)
        
        # Carica l'immagine in scala di grigi
        im = cv2.imread(im_path, 0)
        
        # Verifica se l'immagine è stata caricata correttamente
        if im is None:
            print(f"Immagine {filename} non trovata o non leggibile.")
        else:
            print(f"Immagine {filename} caricata con successo.")
        
        # Embed il watermark nell'immagine
        watermarked = embedding(im_path)
        
        # Crea un watermark casuale per H0 (ipotesi negativa)
        fakemark = np.random.uniform(0.0, 1.0, mark_size)
        fakemark = np.uint8(np.rint(fakemark))

        # Applica un attacco casuale all'immagine watermarkata
        res_att = random_attack(watermarked)

        # Estrai il watermark dall'immagine attaccata
        wat_attacked = detection(im, res_att)
        wat_extracted = detection(im, watermarked)

        # Calcola la similarità H1 (con l'immagine attaccata)
        scores.append(similarity(wat_extracted, wat_attacked))
        labels.append(1)

        # Calcola la similarità H0 (con il watermark casuale)
        scores.append(similarity(fakemark, wat_attacked))
        labels.append(0)

        # Incrementa il contatore dei sample
        sample += 1
        if sample >= 500:  # Limita il numero totale di sample
            break

# Generazione della curva ROC e calcolo dell'AUC
fpr, tpr, tau = roc_curve(np.asarray(labels), np.asarray(scores), drop_intermediate=False)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([-0.01, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()

# Trova la soglia corrispondente ad un FPR di circa 0.05
idx_tpr = np.where((fpr-0.05)==min(i for i in (fpr-0.05) if i > 0))
print('Per un FPR ≈ 0.05, TPR = %0.2f' % tpr[idx_tpr[0][0]])
print('Per un FPR ≈ 0.05, soglia = %0.2f' % tau[idx_tpr[0][0]])
print('FPR verificato: %0.2f' % fpr[idx_tpr[0][0]])
