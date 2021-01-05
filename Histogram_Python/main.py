import cv2 as cv
from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = [13, 13]

img = cv.imread("source/test.jpeg")
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
color = ('r', 'g', 'b')

plt.subplot(2, 2, 1)
plt.imshow(img)
plt.axis("off")
plt.title("Color image", fontsize=18)

plt.subplot(2, 2, 2)
plt.imshow(gray, cmap="gray")
plt.axis("off")
plt.title("Gray scale image", fontsize=18)

plt.subplot(2, 2, 3)
for i, c in enumerate(color):
    hist_c = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_c, color=c)
    plt.xlim([0, 256])
plt.title("Histogram color", fontsize=18)

plt.subplot(2, 2, 4)
hist_g = cv.calcHist([gray], [0], None, [256], [0, 256])
plt.plot(hist_g)
plt.xlim([0, 256])
plt.title("Histogram gray", fontsize=18)

plt.show()
