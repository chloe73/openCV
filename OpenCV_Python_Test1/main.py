import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

path = "source/test.jpeg"

# 컬러 이미지로 로드
image = cv.imread(path, cv.IMREAD_COLOR)
cv.imshow("Origin image", image) # 이미지 출력하기
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 1. 흑백화
# 이미지를 흑백으로 만드는 방법은 2가지가 있다
# 두 방법은 비슷해 보여도 다른 결과값을 가져온다

# 흑백 이미지로 로드
gray1 = cv.imread(path, cv.IMREAD_GRAYSCALE)
cv.imshow("Gray scale roading", gray1)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 이미 로딩한 컬러 이미지를 흑백으로 변환
gray2 = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
cv.imshow("Convert to gray scale", gray2)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 2. 이미지 크기 변경
# 이미지의 크기를 줄여 메모리 사용량을 크게 감소시킬 수 있다
image_540 = cv.resize(gray1, (540, 540))
cv.imshow("Resize 540X540", image_540)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

image_540_color = cv.resize(image, (540, 540))
cv.imshow("Resize 540X540 color", image_540_color)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 3. 이미지 크롭
# 이미지에서 필요한 특정 부분을 행과 열을 선택하여 자른다
crop = image_540[270:, 270:]
cv.imshow("Cropped image", crop)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

crop_color = image_540_color[270:, 270:]
cv.imshow("Cropped image color", crop_color)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 4. 블러
# 블러를 사용하여 이미지의 노이즈를 제거할 수 있다.
# 사람의 눈에는 이미지가 흐려지지만 영상처리에는 더 수월해진다
# GaussianBlur()를 사용하기
blur = cv.GaussianBlur(crop, (5, 5), 0)
cv.imshow("GaussianBlur", blur)
cv.waitKey(0)
cv.destroyAllWindows()

# 커널을 이용하여 블러처리
n = 64
kernel = np.array([[1, 2, 3, 2, 1],
                   [2, 3, 4, 3, 2],
                   [3, 4, 5, 4, 3],
                   [2, 3, 4, 3, 2],
                   [1, 2, 3, 2, 1]], np.float)
arr = np.array([[n, n, n, n, n],
                [n, n, n, n, n],
                [n, n, n, n, n],
                [n, n, n, n, n],
                [n, n, n, n, n]])
kernel = kernel/arr
blur_color = cv.filter2D(crop_color, -1, kernel)
cv.imshow("Kernel Blur", blur_color)
cv.waitKey(0)
cv.destroyAllWindows()

# 5. 이미지 대비 높이기
# 히스토그램 평활화는 객체의 형태가 두드러 지도록 만들어주는 이미지 처리 도구이다
hist = cv.equalizeHist(crop)
cv.imshow("Histogram", hist)
cv.waitKey(0)
cv.destroyAllWindows()
# 컬러 이미지는 YUV로 변환해야한다
hist_yuv = cv.cvtColor(crop_color, cv.COLOR_BGR2YUV)
hist_yuv[:, :, 0] = cv.equalizeHist(hist_yuv[:, :, 0])
hist_color = cv.cvtColor(hist_yuv, cv.COLOR_YUV2BGR)
cv.imshow("Histogram color", hist_color)
cv.waitKey(0)
cv.destroyAllWindows()

# matplotilb을 이용하여 이미지 출력하기
# openCV는 BGR를 사용하지만 matplotilb은 RGB를 사용한다

plotilb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # BGR를 RGB로 변환
# 이미지 출력
plt.imshow(plotilb)
plt.axis("off")
plt.show()

