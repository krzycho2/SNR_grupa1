from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import glob
from sklearn.svm import LinearSVC
from sklearn.svm import LinearSVC
import pickle
import metody

with open('LBPdata8_2.pckl', 'rb') as f:
	data,labels = pickle.load(f)
print("Zaimportowano dane\n")
# data = np.array(data, dtype = "float32")

model = LinearSVC(C=100.0, random_state=42)
model.fit(data,labels)
desc = LocalBinaryPatterns(8, 2)
trainingMainPath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# paths - wszystkie (pełne) ścieżki do zdjęć liści
paths = metody.getListOfFiles(trainingMainPath)

# Koniec aktu 1: Mamy cechy obrazu. Następnie można użyć ich do rozpoznawania obrazów
# Akt 2 
print("Uczenie SVC\n")
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)
print("Testowanie\n")
# Testing

for imagePath in paths:
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))
	
	# display the image and the prediction

	cv2.putText(image, prediction[0], (300, 300), cv2.FONT_HERSHEY_SIMPLEX,
		10.0, (0, 0, 255), 10)
	print(imagePath + "\n")
	plt.imshow(image)
	plt.show()
	

