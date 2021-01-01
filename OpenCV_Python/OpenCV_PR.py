import cv2
import numpy as np
from matplotlib import pyplot as plt
#C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg

#흑백 이미지로 로드
image = cv2.imread("C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg", cv2.IMREAD_GRAYSCALE)

#이미지 출력
plt.imshow(image, cmap ="gray"),plt.axis("off") 
plt.show()


#데이터 타입확인
type(image)

#이미지 데이터를 확인
image

#이미지의 차원을 확인(해상도)
image.shape