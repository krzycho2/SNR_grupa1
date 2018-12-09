from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import glob
from sklearn.svm import LinearSVC
import metody   # user-defined


import metody
# Ścieżka główna do folderu z liśćmi
trainingMainPath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# paths - wszystkie (pełne) ścieżki do zdjęć liści
paths = metody.getListOfFiles(trainingMainPath)
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
print("Ekstrakcja cech\n")
licznik = 0

for imagePath in paths:
    print(imagePath + "\n")
    image = cv2.imread(imagePath)
    print(str(image.shape) + "\n")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Skala szarości
    hist = desc.describe(gray) # LBP na obrazie - zwraca histogram

	# extract the label from the image path, then update the
	# label and data lists
    labels.append(imagePath.split("/")[-2])
    data.append(hist)
    licznik += 1
    print("Wykonano: " + str(int( licznik / len(trainingMainPath) )))
