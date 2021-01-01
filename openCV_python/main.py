import cv2

# 이미지 불러오기 & 이미지 사이즈 조절
def showImage():
    imgFile = "resource/img.jpg"  # 파일 위치 저장

    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)  #필요한 파일을 읽어옵니다. & 컬러사진

    # 읽기 플래그
    # cv2.IMREAD_COLOR 컬러 이미지로 로드함
    # cv2.IMREAD_GRAYSCALE 흑백 이미지로 로드함
    # cv2.IMREAD_UNCHANGED 알파 채널을 포함하여 이미지 그대로 로드함

    resize_img = cv2.resize(img, (300,300))

    cv2.imshow('img', resize_img)

    cv2.waitKey(0)  # (ms 초 동안 유지된다.) 0이면 키보드 클릭시에 없어짐
    cv2.destroyAllWindows()

showImage()

