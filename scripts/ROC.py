import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from embedding2 import *
from detection import *
from attack import *
import random

def similaritys(X,X_star):
    #Computes the similarity measure between the original and the new watermarks.
    s = np.sum(np.multiply(X, X_star)) / (np.sqrt(np.sum(np.multiply(X, X))) * np.sqrt(np.sum(np.multiply(X_star, X_star))))
    return s

def save_image_temp(image, image_id, folder="temp_images"):
    """
    Salva temporaneamente un'immagine su disco per passare il percorso a embedding2.
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    temp_filename = os.path.join(folder, f"temp_image_{image_id}.bmp")
    cv2.imwrite(temp_filename, image)
    return temp_filename

def load_images_from_folder(folder_path, file_format="bmp", num_images=50):
    """
    Carica un insieme di immagini dalla cartella specificata e restituisce una lista di immagini.
    """
    images = []
    for i in range(1, num_images + 1):
        filename = f"{folder_path}/{i:04d}.{file_format}"
        image = cv2.imread(filename, 0)  # Leggi l'immagine in scala di grigi
        if image is not None:
            images.append(image)
        else:
            print(f"Image {filename} could not be loaded.")
    return images

def generate_random_watermark(mark_size):
    """
    Genera un watermark casuale di una data dimensione.
    """
    return np.random.choice([0, 1], size=mark_size)

def random_attack(watermarked_image):
    """
    Seleziona casualmente un attacco e lo applica all'immagine.
    """
    attack_number = random.randint(1, 7)  # Genera un numero casuale tra 1 e 6

    
    # Switch per selezionare l'attacco in base al numero casuale
    if attack_number == 1:
        print(f"Applico blur")
        sigma = random.choice([0.2, 0.5, 0.7, 1, 1.2])
        attacked_image = attacks(watermarked_image, 'blur', sigma)
        
    elif attack_number == 2:
        print(f"Applico AWGN")
        std = random.randint(5,30)
        attacked_image = attacks(watermarked_image, 'awgn',[std, 42])
        
    elif attack_number == 3:
        print(f"Applico sharpening")
        sigma = random.randint(5, 25)
        alpha = random.randint(5, 25)
        attacked_image = attacks(watermarked_image, 'sharpening', [sigma/10, alpha/10])
        #attacked_image = watermarked_image
    elif attack_number == 4:
        print(f"Applico median")
        attacked_image = attacks(watermarked_image, 'median', 3)
        #attacked_image = watermarked_image
    elif attack_number == 5:
        print(f"Applico resize")
        scale = random.choice([0.5, 0.75, 0.875, 0.9375, 1])
        attacked_image = attacks(watermarked_image, 'resize', scale)
        
    elif attack_number == 6:
        print(f"Applico JPEG compression")
        compression_rate = random.randint(50, 100)
        attacked_image = attacks(watermarked_image, 'jpeg', compression_rate)
    
    elif attack_number == 7:
        print("Applico no attack")
        attacked_image = watermarked_image


    return attacked_image

def estimate_threshold(images, original_watermark):
    """
    Stima la soglia (threshold) basata su ROC curve.
    
    Parametri:
    - images: lista di immagini su cui eseguire embedding e attacco.
    - original_watermark: il watermark originale da inserire.
    - mark_size: dimensione del watermark.
    - detection_func: funzione per estrarre il watermark e fare la rilevazione.
    - similarity_func: funzione per calcolare la similarità tra due watermark.
    
    Ritorna:
    - threshold: soglia ottimale basata su ROC.
    """

    mark_size=1024
    scores = []
    labels = []
    num_repeats=10
    level=-1
    # Loop sulle immagini
    for image_id, image in enumerate(images):
        level+=1
        ranger=-1
        for _ in range(num_repeats):
          ranger+=1
          print(f"{level}{ranger}")
        # 1. Salva l'immagine temporaneamente e passa il percorso a embedding2
          temp_image_path = save_image_temp(image, image_id)

          # 2. Embed il watermark originale
          watermarked_image = embedding2(temp_image_path, original_watermark)
          
          # 3. Attacco l'immagine
          attacked_image = random_attack(watermarked_image)
          
          # 4. Estrai il watermark dall'immagine attaccata
          extracted_watermark = extract_watermark(image, attacked_image)
          
          # 5. Calcola la similarità (True Positive)
          score_tp = similaritys(original_watermark, extracted_watermark)
          scores.append(score_tp)
          labels.append(1)  # True Positive
          
          # 6. Genera un watermark casuale e confrontalo
          random_watermark = generate_random_watermark(mark_size)
          score_tn = similaritys(random_watermark, extracted_watermark)
          scores.append(score_tn)
          labels.append(0)  # True Negative

    fpr, tpr, thresholds = roc_curve(labels, scores, drop_intermediate=False)
    
    # Calcolo dell'AUC
    roc_auc = auc(fpr, tpr)
    
    # Trova il valore del threshold per FPR ≈ 0.05
    idx_tpr = np.where((fpr-0.05) == min(i for i in (fpr-0.05) if i > 0))
    
    # Mostra la ROC curve
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='AUC = %0.2f' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([-0.01, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    
    # Stampa il TPR e il threshold corrispondente a FPR ≈ 0.05
    print('Per FPR ≈ 0.05, il TPR corrispondente è: %0.2f' % tpr[idx_tpr[0][0]])
    print('Per FPR ≈ 0.05, la soglia corrispondente è: %0.2f' % thresholds[idx_tpr[0][0]])

# Alla fine del tuo ciclo di stima threshold, chiama la funzione plot_roc_and_find_threshold
   