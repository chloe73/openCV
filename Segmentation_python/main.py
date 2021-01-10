import numpy as np
import cv2 as cv

img = cv.imread("source/coins.jpg")
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
gray = cv.GaussianBlur(gray, (5, 5), 0)
ret, binar = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
border = cv.dilate(binar, np.ones((2, 2), np.uint8), iterations=2)
dt = cv.distanceTransform(binar, cv.DIST_L2, 5)
dt = ((dt-dt.min())/(dt.max()-dt.min())*255).astype(np.uint8)
ret, dt = cv.threshold(dt, 180, 255, cv.THRESH_BINARY)
dt = np.uint8(dt)
unknown = cv.subtract(border, dt)
ret, markers = cv.connectedComponents(dt)
markers = markers+1
markers[unknown == 255] = 0

markers = cv.watershed(img, markers)
img[markers == -1] = [255, 0, 0]
img[markers == 1] = [255, 255, 0]

cv.imshow("dt", dt)
cv.imshow("unknown", unknown)
cv.imshow("result", img)
cv.waitKey(0)
cv.destroyAllWindows()