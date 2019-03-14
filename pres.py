import pickle
import numpy as np
import metody
import matplotlib.pyplot as plt
with open('LBPdata16_2.pckl', 'rb') as f:
	data,labels = pickle.load(f)
print("Zaimportowano dane\n")
data = np.array(data, dtype = "float32")
data1 = data[1]
data100 = data[100]

# ind = np.linspace(1,18,18)
# p1 = plt.bar(ind, data100)
# plt.xticks(ind)
# plt.title('Histogram dla drzewa oliwnego')
# plt.xlabel('Wzorzec')
# plt.ylabel('Czestosc wystepowania')
# plt.show(labels[100])
# plt.show()

trainingMainPath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# paths - wszystkie (pełne) ścieżki do zdjęć liści
paths = metody.getListOfFiles(trainingMainPath)
plt.show(paths[100])