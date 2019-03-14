from localbinarypatterns import LocalBinaryPatterns
import matplotlib.pyplot as plt
import metody   # user-defined
import numpy as np
import pickle
import cv2

# ---------------------------------------------------
# PROJEKT Z SNR - Zadanie 1. w 3 aktach
# by Krzysztof Krupiński 2018
# ---------------------------------------------------
# AKT 1: Ekstrakcja cech obrazów za pomocą deskryptora 

# Ścieżka główna do folderu z liśćmi
trainingMainPath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# paths - wszystkie (pełne) ścieżki do zdjęć liści
paths = metody.getListOfFiles(trainingMainPath)  # Lista wszystkich plików w folderze
desc = LocalBinaryPatterns(16, 2)  # Obiekt klasy LBP - deksryptor LBP 8 - liczba próbek w sąsiedztwie, 2 - promień sąsiedztwa
data = []
labels = []

licznik = 0  # Do wyświetlania postępou ekstracji
allFiles =  len(paths)

print("Ekstrakcja cech\n")
for imagePath in paths:
	image = cv2.imread(imagePath)	# Wczytanie obrazu
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Skala szarości
	hist = desc.describe(gray) # LBP wykonane na obrazie - zwraca histogram

	labels.append(imagePath.split("/")[-2]) # Tylko nazwa kwiatu
	data.append(hist)
	print("Wykonano: " + str( licznik / allFiles*100 ) + " %\n")
	licznik += 1
	
data = np.array(data, dtype = "float32")
labels = np.array(labels)
# Zapis wyesktrachowanych cech do pliku o łatwym dostępie
with open('LBPdata16_2.pckl', 'wb') as f:
	pickle.dump([data, labels], f)
	
