import cv2
import numpy as np
from matplotlib import pyplot as plt
#C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg

#컬러 이미지로 로드
image_bgr = cv2.imread("C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg", cv2.IMREAD_COLOR)

#픽셀확인
image_bgr[0,0]

#RGB로 변환
image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

#이미지 출력
plt.imshow(image_bgr), plt.axis("off") 

plt.show()


#데이터 타입확인
type(image_bgr)

#이미지 데이터를 확인
image_bgr

#이미지의 차원을 확인(해상도)
image_bgr.shape