from skimage import feature
import numpy as np

class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints # Liczba punktów w sąsiedztwa
		self.radius = radius		# Odległość piksela od sąsiadów

	def describe(self, image, eps=1e-7):
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))

		# Normalizacja histogramu
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)

		# Zwróć histogram deskryptora LBP
		return hist