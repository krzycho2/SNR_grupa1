import metody
import cv2
import matplotlib.pyplot as plt
# path = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio"
# files = metody.getListOfFiles(path)

impath = "/home/krzysztof/Dokumenty/SNR_grupa1/Folio Leaf Dataset/Folio/barbados cherry/20150324_165844.jpg"
image = cv2.imread(impath)
fontType = cv2.FONT_HERSHEY_PLAIN
fontScale = 10
cv2.putText(image, "Jakis tekst", (300, 300), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 255), 10)
plt.imshow(image)
plt.show()

# Zabawa z MNIST
mainFolderPath = "/home/krzysztof/Dokumenty/SNR/MNIST"
paths= [mainFolderPath  + "/t10k-images-idx3-ubyte",
        mainFolderPath  + "/t10k-labels-idx1-ubyte",
        mainFolderPath  + "/train-images-idx3-ubyte",
        mainFolderPath  + "/train-labels.idx1-ubyte"]

testImages  = metody.getMNISTdata(paths[0])
testLabels  = metody.getMNISTdata(paths[1])
trainImages = metody.getMNISTdata(paths[2])
trainLabels = metody.getMNISTdata(paths[3])


# Ewentualna konwersja do int64
testImages = metody.uint2int(testImages)
testLabels = metody.uint2int(testLabels)
trainImages = metody.uint2int(trainImages)
trainLabels = metody.uint2int(trainLabels)