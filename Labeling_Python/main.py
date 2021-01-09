import cv2 as cv
import random


def setRandColor(min=0, max=255):

    while True:
        r = random.randrange(0, max)
        g = random.randrange(0, max)
        b = random.randrange(0, max)
        if not(r < min and g < min and b < min):
            return r, g, b


source = "source/test.jpeg"
origin = cv.imread(source, cv.IMREAD_COLOR)
origin = cv.resize(origin, (720, 720))
#cv.imshow("origin", origin)
img = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (9, 9), 0)

ret, binar = cv.threshold(img, 0, 255, cv.THRESH_OTSU)
#cv.imshow("binar", binar)
cnt, labels, stats, centroid = cv.connectedComponentsWithStats(binar)
count = 0
for i in range(0, cnt):
    (x, y, w, h, area) = stats[i]
    if area < 70:
        continue
    if area > 1000:
        continue
    if w > 700:
        continue
    if h > 700:
        continue

    cv.rectangle(origin, (x, y, w, h), (0, 255, 255))
    count += 1
    string = str(count)
    cv.putText(origin, string, (x, y), cv.FONT_HERSHEY_PLAIN, 1.2, setRandColor(128, 255))

cv.imshow("Labeling", origin)
cv.waitKey(0)
cv.destroyAllWindows()