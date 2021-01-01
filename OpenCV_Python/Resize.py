import cv2
import numpy as np
from matplotlib import pyplot as plt
# 흑백 이미지로 로드
image =cv2.imread("C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg",cv2.IMREAD_GRAYSCALE)

#이미지를  50*50로 크기변경한 이미지로 저장함
image_50x50 = cv2.resize(image,(50,50))

#이미지 출력
plt.imshow(image_50x50, cmap="gray"),plt.axis("off")
plt.show()