# 1/7 computer vision study : segmentation
import numpy as np
import cv2
import matplotlib.pyplot as plt

# imgFile = "resource/coin.png" # 파일 위치 저장
img = cv2.imread("resource/coin.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

plt.imshow(thresh, cmap='gray')
plt.axis('off')
plt.show()

kernel = np.ones((3,3),np.uint8)
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

# 배경이 확실한 영역
sure_bg = cv2.dilate(opening,kernel,iterations=3)

# 전경이 확실한 영역 찾기
dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

# 모르겠는 영역 찾기
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg,sure_fg)

plt.figure(figsize=(12,8))
plt.subplot(221), plt.imshow(opening,cmap='gray')
plt.title("Noise Removed"), plt.axis('off')
plt.subplot(222), plt.imshow(sure_bg,cmap='gray')
plt.title("Sure Background"), plt.axis('off')
plt.subplot(223), plt.imshow(dist_transform,cmap='gray')
plt.title("Distance Transform"), plt.axis('off')
plt.subplot(224), plt.imshow(sure_fg,cmap='gray')
plt.title("Threshold"), plt.axis('off')
plt.show()


# commit
# ========== 히스토그램 균일화(평활화) : Histogram Equalization ===========
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# imgFile = "resource/111.jpg" # 파일 위치 저장
# img = cv2.imread(imgFile, 0)
#
#
# hist, bins = np.histogram(img.ravel(), 256,[0,256]) # 히스토그램 구하기
# cdf = hist.cumsum()  # numpy 배열을 1차원으로 하고 더한 값을 누적하여 배열 생성
#
# cdf_m = np.ma.masked_equal(cdf, 0)  # cdf에서 값이 0인 부분을 mask하여 제외 처리
#
# cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())  # 여기가 균일화 방정식
#
# cdf = np.ma.filled(cdf_m, 0).astype('uint8')  # mask로 제외했던 0값을 복원
#
# img2 = cdf[img]
#
# cv2.imshow('Equalization', img2)
#
# cv2.waitKey(0)
#
# cv2.destroyAllWindows()

# =========================================================================
# import numpy as np
# import cv2
# from matplotlib import pyplot as plt
#
# imgFile = "resource/img.jpg" # 파일 위치 저장
# img = cv2.imread(imgFile, 0)
#
# color = ('b','g','r')
#
# for i,col in enumerate(color):
#     histr = cv2.calcHist([img],[i],None,[256],[0,256])
#     plt.plot(histr, color=col)
#     plt.xlim([0,256])
# plt.show()

# =======openCV_histogram 실습 1=============
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# imgFile = "resource/img.jpg" # 파일 위치 저장
# img = cv2.imread(imgFile,0)
#
# plt.hist(img.ravel(), 256, [0,256]);
# plt.show()

# ======================================================================

# import cv2
# import numpy as np
# import matplotlib.pyplot as plt

# import cv2
#
# def showImage():
#     imgFile = "resource/img.jpg" # 파일 위치 저장
#
#     img = cv2.imread(imgFile, cv2.IMREAD_GRAYSCALE) #필요한 파일을 읽어옵니다.
#     # 읽기 플래그
#     # cv2.IMREAD_COLOR 컬러 이미지로 로드함
#     # cv2.IMREAD_GRAYSCALE 흑백 이미지로 로드함
#     # cv2.IMREAD_UNCHANGED 알파 채널을 포함하여 이미지 그대로 로드함
#
#     resize_img= cv2.resize(img, (500,500))
#
#     image_blur = cv2.blur(resize_img, (100,100)) # 이미지 블러 처리
#
#     cv2.imshow('image', resize_img)
#     cv2.imshow('blur', image_blur)
#     cv2.waitKey(0) # (ms 초 동안 유지된다.) 0이면 키보드 클릭시에 없어짐
#     cv2.destroyAllWindows() # 모든 화면 꺼짐 showImage()
#
# showImage()

# import cv2
#
# # 이미지 불러오기 & 이미지 사이즈 조절
# def showImage():
#     imgFile = "resource/img.jpg"  # 파일 위치 저장
#
#     img = cv2.imread(imgFile, cv2.IMREAD_COLOR)  #필요한 파일을 읽어옵니다. & 컬러사진
#
#     # 읽기 플래그
#     # cv2.IMREAD_COLOR 컬러 이미지로 로드함
#     # cv2.IMREAD_GRAYSCALE 흑백 이미지로 로드함
#     # cv2.IMREAD_UNCHANGED 알파 채널을 포함하여 이미지 그대로 로드함
#
#     resize_img = cv2.resize(img, (300,300))
#
#     cv2.imshow('img', resize_img)
#
#     cv2.waitKey(0)  # (ms 초 동안 유지된다.) 0이면 키보드 클릭시에 없어짐
#     cv2.destroyAllWindows()
#
# showImage()

