import cv2 as cv

source = "source/test.jpeg"
origin = cv.imread(source, cv.IMREAD_COLOR)
origin = cv.resize(origin, (720, 720))
#cv.imshow("origin", origin)
img = cv.cvtColor(origin, cv.COLOR_BGR2GRAY)
img = cv.GaussianBlur(img, (9, 9), 0)

ret, binar = cv.threshold(img, 100, 255, 1)
cv.imshow("binar", binar)
cnt, labels, stats, centroid = cv.connectedComponentsWithStats(binar)
count = 0
for i in range(1, cnt):
    (x, y, w, h, area) = stats[i]
    if area < 200:
        continue

    cv.rectangle(origin, (x, y, w, h), (0, 255, 255))
    count += 1
    string = str(count)
    cv.putText(origin, string, (x, y+15), cv.FONT_HERSHEY_PLAIN, 1.2, (255, 255, 255))

cv.imshow("Labeling", origin)
cv.waitKey(0)
cv.destroyAllWindows()