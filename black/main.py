import numpy as np
import cv2

cat = cv2.imread("pic/cutecat.jpg")
gray_cat = cv2.cvtColor(cat, cv2.COLOR_BGR2GRAY)

cv2.imshow("cutecat", cat)
cv2.imshow("gray_cat", gray_cat)
cv2.waitKey()