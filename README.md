# Multimedia-Data-Security
Il codice contiene 4 funzioni principali: [embedding](/scripts/embedding.py), [ROC](/scripts/ROC.py), [detection](/scripts/detection.py) e [attack](/scripts/attack.py). Tutte e quattro possono essere testate attraverso demo.ipynb.
Al fine di eseguire correttamente il drive è necessario inserire questo progetto dentro una cartella con già installato l'ambiente usato a lezioni ed inserire dentro la cartella [images](/images/) le immagini scaricate da [qui](https://drive.google.com/file/d/1-n9bmQFDBShRN4Tr_WpmCf1jFHOS5MTO/view)

## Embedding
Inserisce un watermark in un'immagine mantenendo alta la qualità dell'immagine stessa, deve essere robusto e non percepibile visivamente.
Il codice ha questa struttura:
```
def embedding(input1, input2):
    CODE
    return output1
```
Dove:
- input1: percorso dell'immagine originale
- input2: percorso del watermark
- output1: immagine con matermark

## ROC
Calcola la soglia di similarità tra il watermark originale e quello estratto dopo un attacco
Biosgna seguire questi passaggi:
- inserire il watermark in un set di immagini
- attaccare le immagini ed estrarre il watermark
- confrontare il watermark estratto con quello originale
- capire la soglia τ migliore per capire se il watermark è presente


## Detection
Controlla se il watermark è ancora presente in un'immagine attaccata, utilizza il τ calcolato prima
Il codice ha questa struttura:
```
def detection(input1, input2, input3):
    CODE
    return output1, output2
```
- input1: percorso dell'immagine originale
- input2: percorso dell'immagine con matermark
- input3: percorso dell'immagine attaccata
- output1: 1 se il watermark è presente, 0 se non lo è
- output2: WPSNR tra l'immagine con watermark e quella attaccata

## Attack
Attacca un0'immagine al fine di rimuovere il watermark, mantenendo alta la qualità dell'immagine
Il codice ha questa struttura:
```
def attacks(input1, attack_name, param_array):
    CODE
    return output1
```
- input1: percorso dell'immagine con watermark
- attack_name: nome dell'attacco
- param_array: parametri per l'attacco
- output1: immagine attaccata