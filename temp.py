from sklearn.svm import LinearSVC
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import metody
with open('LBPdata16_2.pckl', 'rb') as f:
	data,labels = pickle.load(f)
print("Zaimportowano dane\n")
trainingMainPath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# paths - wszystkie (pełne) ścieżki do zdjęć liści
paths = metody.getListOfFiles(trainingMainPath)  # Lista wszystkich plików w folderze

# Wyświetlenie przykładowego histogramu
# os_x = np.linspace(1,18,18)
# p1 = plt.bar(os_x, data[0,:])
# plt.xticks(os_x)
# plt.title("Wynik ekstrakcji cech z obrazu dla p=16, r=2")
# plt.ylabel("Częstość występowania")
# plt.xlabel("Próbka")

im = mpimg.imread(paths[0])
plt.imshow(im)
plt.title(labels[0])


plt.show()

