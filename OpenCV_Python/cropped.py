import cv2
import numpy as np
from matplotlib import pyplot as plt
# 흑백 이미지로 로드
image =cv2.imread("C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg",cv2.IMREAD_GRAYSCALE)

#모든 행과 열의 중간
image_cropped = image[:,:999]

#이미지 출력
plt.imshow(image_cropped, cmap="gray"),plt.axis("off")
plt.show()