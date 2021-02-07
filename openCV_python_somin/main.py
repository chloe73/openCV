# MNIST - 학습시킨 데이터로 테스트하기
import pickle
import sys
import matplotlib.pylab as plt
from dataset.dataset.mnist import load_mnist

sys.path.append("./dataset") # 이때, dataset 폴더는 실행하는 py 파일의 경로와 일치해야 한다.
(train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(flatten = True, normalize = False)


def sigmoid(x):  # sigmoid 함수
    return 1 / (1 + plt.np.exp(-x))


def softmax(matrix):  # softmax 함수
    maximum_of_matrix = plt.np.max(matrix)
    difference_from_maximum = matrix - maximum_of_matrix
    exponential_of_difference = plt.np.exp(difference_from_maximum)
    sum_of_exponential = plt.np.sum(exponential_of_difference)
    y = exponential_of_difference / sum_of_exponential
    return y


def get_data():  # mnist 데이터를 불러옴. 여기서는 이 중에 test 변수만을 사용할 것이다.
    (image_train, label_train), (image_test, label_test) = load_mnist(flatten=True, normalize=False)
    return image_test, label_test


def init_network():  # sample_weight 를 불러와서 신경망 구성
    with open('./dataset/sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):  # 테스트 케이스들을 테스트
    # hidden data 2개
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = plt.np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = plt.np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = plt.np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


images, labels = get_data()
network = init_network()

accuracy_cnt = 0
for i in range(len(images)):  # 각 테스트 케이스들에 대해
    y = predict(network, images[i])  # 실행 결과 output 10개가 나온다
    # 각 0~9 별로 비슷 정도에 대한 수치이다
    p = plt.np.argmax(y)  # 가장 가능성이 높은(값이 큰) 것을 선택
    if p == labels[i]:  # 실제 값과 비교하여, 예측과 실제가 맞으면 카운트
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(images)))

# MNIST 데이터 가져오기
# import sys
#
# from dataset.dataset.mnist import load_mnist
#
# sys.path.append("./dataset") # 이때, dataset 폴더는 실행하는 py 파일의 경로와 일치해야 한다.
#
# (train_image_data, train_label_data), (test_image_data, test_label_data) = load_mnist(flatten = True, normalize = False)
#
# import matplotlib.pylab as plt
#
# # n(0~59,999)을 입력값으로 주면, 그 번호에 맞는 label과 image를 가져와서, 그걸 그림으로 나타내는 함수이다.
# def mnist_show(n) :
#     image = train_image_data[n]
#     image_reshaped = image.reshape(28, 28)
#     image_reshaped.shape
#     label = train_label_data[n]
#     plt.figure(figsize = (4, 4))
#     plt.title("sample of " + str(label))
#     plt.imshow(image_reshaped, cmap="gray")
#     plt.show()
#
# mnist_show(2747)

# 텐서플로우 테스트 코드
# import tensorflow as tf
# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train),(x_test, y_test) = mnist.load_data()
# x_train, x_test = x_train / 255.0, x_test / 255.0
#
# model = tf.keras.models.Sequential([
#   tf.keras.layers.Flatten(input_shape=(28, 28)),
#   tf.keras.layers.Dense(128, activation='relu'),
#   tf.keras.layers.Dropout(0.2),
#   tf.keras.layers.Dense(10, activation='softmax')
# ])
#
# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])
#
# model.fit(x_train, y_train, epochs=5)
# model.evaluate(x_test, y_test)

# ==============
# import cv2
#
# im = cv2.imread("resource/QRImage.png")
# row, col = im.shape[:2]
# bottom = im[row-2:row, 0:col]
# mean = cv2.mean(bottom)[0]
#
# bordersize = 10 # 이미지 테두리 크기
# border = cv2.copyMakeBorder(
#     im,
#     top=bordersize,
#     bottom=bordersize,
#     left=bordersize,
#     right=bordersize,
#     borderType=cv2.BORDER_CONSTANT,
#     value=[mean, 1, mean]
# )
#
# # cv2.imshow('image', im)
# # cv2.imshow('bottom', bottom)
# cv2.imshow('border', border)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# computer vision study : qr코드 인식 (여러 각도에서 찍은 qr 인식)
# import cv2
# import numpy as np
#
# inputImage = cv2.imread("resource/QRImage.png")
# inputImage = cv2.resize(inputImage, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
#
# qrDecoder = cv2.QRCodeDetector()
#
# # QR코드를 찾고 디코드해줍니다
# data, bbox, rectifiedImage = qrDecoder.detectAndDecode(inputImage)
# if len(data) > 0:
#     print("Decoded Data : {}".format(data))
#     rectifiedImage = np.uint8(rectifiedImage)
#
# else:
#     print("QR Code not detected")

# computer vision stduy : 이미지 뒤틀기(원근 변환 (perspective.py))
# import cv2
# import numpy as np
#
# file_name = "resource/111.jpg"
# img = cv2.imread(file_name)
# rows, cols = img.shape[:2]
#
# #---① 원근 변환 전 후 4개 좌표
# pts1 = np.float32([[0,0], [0,rows], [cols, 0], [cols,rows]])
# pts2 = np.float32([[100,50], [10,rows-50], [cols-100, 50], [cols-10,rows-50]])
#
# #---② 변환 전 좌표를 원본 이미지에 표시
# cv2.circle(img, (0,0), 10, (255,0,0), -1)
# cv2.circle(img, (0,rows), 10, (0,255,0), -1)
# cv2.circle(img, (cols,0), 10, (0,0,255), -1)
# cv2.circle(img, (cols,rows), 10, (0,255,255), -1)
#
# #---③ 원근 변환 행렬 계산
# mtrx = cv2.getPerspectiveTransform(pts1, pts2)
# #---④ 원근 변환 적용
# dst = cv2.warpPerspective(img, mtrx, (cols, rows))
#
# cv2.imshow("origin", img)
# cv2.imshow('perspective', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# computer vision stduy : 이미지 뒤틀기(어핀 변환 (getAffine.py))
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# file_name = 'resource/111.jpg'
# img = cv2.imread(file_name)
# rows, cols = img.shape[:2]
#
# # ---① 변환 전, 후 각 3개의 좌표 생성
# pts1 = np.float32([[200, 100], [400, 100], [200, 400]])
# pts2 = np.float32([[160, 140], [420, 120], [500, 240]])
#
# # ---② 변환 전 좌표를 이미지에 표시
# cv2.circle(img, (200,100), 5, (255,0), -1)
# cv2.circle(img, (400,100), 5, (0,255,0), -1)
# cv2.circle(img, (200,400), 5, (0,0,255), -1)
#
# #---③ 짝지은 3개의 좌표로 변환 행렬 계산
# mtrx = cv2.getAffineTransform(pts1, pts2)
# #---④ 어핀 변환 적용
# dst = cv2.warpAffine(img, mtrx, (int(cols*1.5), rows))
#
# #---⑤ 결과 출력
# cv2.imshow('origin',img)
# cv2.imshow('affin', dst)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# computer vision study : 이미지 변환 - Flipping(이미지 뒤집기)
# import cv2
#
# image = cv2.imread("resource/111.jpg")
# cv2.imshow("Original", image)
#
# # X축 뒤집기
# flipped = cv2.flip(image, 0)
# cv2.imshow("X axis", flipped)
#
# # Y축 뒤집기
# flipped = cv2.flip(image, 1)
# cv2.imshow("Y axis", flipped)
#
# # X, Y축 동시
# flipped = cv2.flip(image, -1)
# cv2.imshow("Both Flipped", flipped)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()


# computer vision stduy : 이미지 변환 - Scaling - Resize
# import cv2
#
# image = cv2.imread("resource/111.jpg")
# img_scale = cv2.resize(image, (300, 200), interpolation = cv2.INTER_AREA)
# cv2.imshow('Scaling Size', img_scale)
# cv2.waitKey()
# cv2.destroyAllWindows()


# computer vision stduy : 이미지 변환 - Scaling - Cubic Interpolation
# import cv2
#
# image = cv2.imread("resource/111.jpg")
# img_scale = cv2.resize(image, None, fx=0.8, fy=1, interpolation = cv2.INTER_CUBIC)
# cv2.imshow('CUBIC Interpolation', img_scale)
# cv2.waitKey()
# cv2.destroyAllWindows()

# computer vision stduy : 이미지 변환 - Scaling - Linear Interpolation(선형 보간법)
# import cv2
#
# image = cv2.imread("resource/111.jpg")
# img_scale = cv2.resize(image, None, fx=0.8, fy=1, interpolation = cv2.INTER_LINEAR)
# cv2.imshow('Linear Interpolation', img_scale)
# cv2.waitKey()
# cv2.destroyAllWindows()


# computer vision study : 이미지 변환 - Rotation(2)
# import cv2
# import imutils
#
# # load the image and show it
# image = cv2.imread("resource/111.jpg")
# cv2.imshow("Original", image)
#
# # 회전의 중심축을 정의하지 않으면 그림의 중심이 됨
# rotated = imutils.rotate(image, 45)
# cv2.imshow("Rotated by 180 Degrees", rotated)
# cv2.waitKey()
#
# # 회전의 중심 축을 정의하면 해당 중심축으로 회전을 함.
# rotated = imutils.rotate(image, 45, center=(0, 0))  # 회전 중심축 TOP LEFT
# cv2.imshow("Rotated by 180 Degrees", rotated)
# cv2.waitKey()

#  computer vision study : 이미지 변환 - Rotation(1)
# import cv2
#
# # load the image and show it
# image = cv2.imread("resource/111.jpg")
# cv2.imshow("Original", image)
#
# # grab the dimensions of the image and calculate the center of the image
# (h, w) = image.shape[:2]
# (cX, cY) = (w / 2, h / 2)
#
# # rotate our image by 45 degrees
# M = cv2.getRotationMatrix2D((cX, cY), 45, 1.0)
# rotated = cv2.warpAffine(image, M, (w, h))
# cv2.imshow("Rotated by 45 Degrees", rotated)
# cv2.waitKey()


# computer vision study : 이미지 변환 - Translation (상하, 좌우 이동)
# import cv2
# import imutils
#
# image = cv2.imread('resource/111.jpg')
# # X 방향으로 25, Y 방향으로 50 이동할때
# shifted = imutils.translate(image, 25, 50)   # translate는 이해하기 쉬운 용어
#
# cv2.imshow("Shifted Down and Right", shifted)
# cv2.waitKey()
# cv2.destroyAllWindows()


# 1/7 computer vision stduy : labeling
# import sys
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# img = "resource/QRImage.png" # 파일 위치 저장
# # resize_img= cv2.resize(img, (500,500))
# # image =
#
# src = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
#
# resize_img = cv2.resize(src, (500,500)) # 이미지 크기 조절
#
# if resize_img is None:
#     print('Image load failed!')
#     sys.exit()
#
# _, src_bin = cv2.threshold(resize_img, 0, 255, cv2.THRESH_OTSU)
#
# cnt, labels, stats, centroids = cv2.connectedComponentsWithStats(src_bin)
#
# dst = cv2.cvtColor(resize_img, cv2.COLOR_GRAY2BGR)
#
# for i in range(1, cnt): # 각각의 객체 정보에 들어가기 위해 반복문. 범위를 1부터 시작한 이유는 배경을 제외
#     (x, y, w, h, area) = stats[i]
#
#     # 노이즈 제거
#     if area < 20:
#         continue
#
#     cv2.rectangle(dst, (x, y, w, h), (0, 255, 255))
#
# cv2.imshow('src', resize_img)
# cv2.imshow('src_bin', src_bin)
# cv2.imshow('dst', dst)
# cv2.waitKey()
# cv2.destroyAllWindows()

# # 1/7 computer vision study : segmentation
# import numpy as np
# import cv2
# import matplotlib.pyplot as plt
#
# # imgFile = "resource/coin.png" # 파일 위치 저장
# img = cv2.imread("resource/coin.png")
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
#
# plt.imshow(thresh, cmap='gray')
# plt.axis('off')
# plt.show()
# #
# kernel = np.ones((3,3),np.uint8)
# opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
#
# # 배경이 확실한 영역
# sure_bg = cv2.dilate(opening,kernel,iterations=3)
#
# # 전경이 확실한 영역 찾기
# dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
# ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)
#
# # 모르겠는 영역 찾기
# sure_fg = np.uint8(sure_fg)
# unknown = cv2.subtract(sure_bg,sure_fg)
#
# plt.figure(figsize=(12,8))
# plt.subplot(221), plt.imshow(opening,cmap='gray')
# plt.title("Noise Removed"), plt.axis('off')
# plt.subplot(222), plt.imshow(sure_bg,cmap='gray')
# plt.title("Sure Background"), plt.axis('off')
# plt.subplot(223), plt.imshow(dist_transform,cmap='gray')
# plt.title("Distance Transform"), plt.axis('off')
# plt.subplot(224), plt.imshow(sure_fg,cmap='gray')
# plt.title("Threshold"), plt.axis('off')
# plt.show()


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
#
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

