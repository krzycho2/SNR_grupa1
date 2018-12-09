from pyimagesearch.localbinarypatterns import LocalBinaryPatterns
from sklearn.svm import LinearSVC
import cv2
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile,join
import glob
from sklearn.svm import LinearSVC


trainingPath  = "/home/krzysztof/Dokumenty/LBP/images/training"
keyboard = glob.glob(trainingPath + "/keyboard" + "/*.*")
carpet = glob.glob(trainingPath + "/carpet" + "/*.*")
desc = LocalBinaryPatterns(24, 8)
data = []
labels = []
paths = carpet + keyboard
print("Ekstrakcja cech\n")
for imagePath in paths:
	# load the image, convert it to grayscale, and describe it
    # print(imagePath + "\n")
    image = cv2.imread(imagePath) # Wczytujemy obraz
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Skala szarości
    hist = desc.describe(gray) # LBP na obrazie - zwraca histogram

	# extract the label from the image path, then update the
	# label and data lists
    labels.append(imagePath.split("/")[-2])
    data.append(hist)

# Koniec aktu 1: Mamy cechy obrazu. Następnie można użyć ich do rozpoznawania obrazów
# Akt 2 
print("Uczenie SVC\n")
model = LinearSVC(C=100.0, random_state=42)
model.fit(data, labels)
print("Testowanie\n")
# Testing
keyboardTestPath = "/home/krzysztof/Dokumenty/LBP/images/testing/keyboard.png"
carpetTestPath = "/home/krzysztof/Dokumenty/LBP/images/testing/carpet.png"
testPaths = [keyboardTestPath,carpetTestPath] 
for imagePath in testPaths:
	# load the image, convert it to grayscale, describe it,
	# and classify it
	image = cv2.imread(imagePath)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	hist = desc.describe(gray)
	prediction = model.predict(hist.reshape(1, -1))
	
	# display the image and the prediction
	cv2.putText(image, prediction[0], (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
		1.0, (0, 0, 255), 3)

	plt.imshow(image)
	plt.show()

