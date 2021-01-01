import cv2

# 이미지 불러오기 & 이미지 사이즈 조절
def showImage():
    imgFile = "resource/img.jpg"

    img = cv2.imread(imgFile, cv2.IMREAD_COLOR)
    resize_img = cv2.resize(img, (300,300))

    cv2.imshow('img', resize_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

showImage()

