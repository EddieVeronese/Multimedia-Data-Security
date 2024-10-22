import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
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

def load_images_from_folder(folder_path, file_format="bmp", num_images=100):
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
    attack_number = random.randint(1, 6)  # Genera un numero casuale tra 1 e 6
    
    # Switch per selezionare l'attacco in base al numero casuale
    if attack_number == 1:
        print("Applico blur")
        attacked_image = attacks(watermarked_image, 'blur', 3)
        
    elif attack_number == 2:
        print("Applico AWGN")
        attacked_image = attacks(watermarked_image, 'awgn', [10, 42])
        
    elif attack_number == 3:
        print("Applico sharpening")
        attacked_image = attacks(watermarked_image, 'sharpening', [0.5, 0.7])
        
    elif attack_number == 4:
        print("Applico median")
        attacked_image = attacks(watermarked_image, 'median', 3)
        
    elif attack_number == 5:
        print("Applico resize")
        attacked_image = attacks(watermarked_image, 'resize', 0.9)
        
    elif attack_number == 6:
        print("Applico JPEG compression")
        attacked_image = attacks(watermarked_image, 'jpeg', 5)
    
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
    level=0
    # Loop sulle immagini
    for image_id, image in enumerate(images):
        level+=1
        ranger=0
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

    # Genera la ROC curve
    fpr, tpr, thresholds = roc_curve(labels, scores)

    # Seleziona la soglia ottimale per un FPR ∈ [0, 0.1]
    optimal_idx = np.where((fpr >= 0) & (fpr <= 0.1))[0][-1]
    optimal_threshold = thresholds[optimal_idx]

    # Plot della ROC curve (facoltativo)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Watermark Detection')
    plt.show()

    return optimal_threshold