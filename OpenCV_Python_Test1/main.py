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

# 2. 이미지 크롭
# 이미지에서 필요한 특정 부분을 행과 열을 선택하여 자른다
crop = gray1[:, 164:796]
cv.imshow("Cropped image", crop)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

crop_color = image[:, 164:796]
cv.imshow("Cropped image color", crop_color)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

# 3. 이미지 크기 변경
# 이미지의 크기를 줄여 메모리 사용량을 크게 감소시킬 수 있다
image_360 = cv.resize(crop, (360, 360))
cv.imshow("Resize 360X360", image_360)
cv.waitKey(0) # 키 입력 기다리기
cv.destroyAllWindows() # 모든 사진 닫기

image_360_color = cv.resize(crop_color, (360, 360))
cv.imshow("Resize 360X360 color", image_360_color)
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
kernel = kernel/n
blur_color = cv.filter2D(crop_color, -1, kernel)
cv.imshow("Kernel Blur", blur_color)
cv.waitKey(0)
cv.destroyAllWindows()

# 5. 이미지 대비 높이기
# 히스토그램 평활화는 객체의 형태가 두드러 지도록 만들어주는 이미지 처리 도구이다
equ = cv.equalizeHist(crop)
cv.imshow("Histogram equalized", equ)
cv.waitKey(0)
cv.destroyAllWindows()
# 컬러 이미지는 YUV로 변환해야한다
equ_yuv = cv.cvtColor(crop_color, cv.COLOR_BGR2YUV)
equ_yuv[:, :, 0] = cv.equalizeHist(equ_yuv[:, :, 0])
equ_color = cv.cvtColor(equ_yuv, cv.COLOR_YUV2BGR)
cv.imshow("Histogram equalized color", equ_color)
cv.waitKey(0)
cv.destroyAllWindows()

# 6. 경계선 감지
# 경계선 감지를 사용하여 정보가 적은 영역을 제거하고 대부분의 정보가 담긴 이미지 영역을 구분할 수 있다
# 경계선 감지를 하기 전에 평활화와 블러를 사용하면 더 좋은 결과를 얻을 수 있다
# 이번 실습은 threshold를 이미지 픽셀 강도의 중간값을 이용하여 설정
pre = cv.GaussianBlur(equ, (5, 5), 0)
median = np.median(pre)
lower = int(max(0, (1.0 - 0.5) * median))
upper = int(min(255, (1.0 + 0.5) * median))
canny = cv.Canny(pre, lower, upper)
cv.imshow("Canny", canny)
cv.waitKey(0)
cv.destroyAllWindows()

# 7. 모서리 감지
corner = cv.imread("source/corner.jpeg", cv.IMREAD_COLOR)
corner_gray = cv.cvtColor(corner, cv.COLOR_BGR2GRAY)
corner_gray = cv.GaussianBlur(corner_gray, (5, 5), 0)
n = cv.goodFeaturesToTrack(corner_gray, 30, 0.01, 30)
n = np.int0(n)
cv.imshow("Origin", corner)
for i in n:
    x, y = i.ravel()
    cv.circle(corner, (x, y), 5, 255, -1)

cv.imshow("Corner", corner)
cv.waitKey(0)
cv.destroyAllWindows()

# 8. 이진화
max_output = 255
neighborhood = 99
subtract = 10
binar = cv.adaptiveThreshold(image_360, max_output, cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                             cv.THRESH_BINARY, neighborhood, subtract)
cv.imshow("Binarized", binar)
cv.waitKey(0)
cv.destroyAllWindows()

# matplotilb을 이용하여 이미지 출력하기
# openCV는 BGR를 사용하지만 matplotilb은 RGB를 사용한다

plotilb = cv.cvtColor(image, cv.COLOR_BGR2RGB) # BGR를 RGB로 변환
# 이미지 출력
plt.imshow(plotilb)
plt.axis("off")
plt.show()
