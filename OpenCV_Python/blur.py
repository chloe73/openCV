import cv2
import numpy as np
from matplotlib import pyplot as plt
#C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg

#흑백 이미지로 로드
image = cv2.imread("C:/Users/ParkSangHoon/Pictures/Saved Pictures/g_CF006626.jpg", cv2.IMREAD_GRAYSCALE)


#각 픽셀 주변의 5 X 5 커널 평균값으로 이미지를 흐리게한다.
image_blurry = cv2.blur(image,(5,5))

#이미지 출력
plt.imshow(image_blurry, cmap="gray"), plt.axis("off")
plt.show()

#커널 크기의 영향을 강조하기 위해서 
#100 X 100 커널로 같은 이미지를 흐리게만든다
image_very_blurry = cv2.blur(image,(100,100))

#이미지 출력하기
plt.imshow(image_very_blurry,cmap="gray"),plt.yticks([])
plt.show()