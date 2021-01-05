# Script is based on https://github.com/richzhang/colorization/blob/master/colorization/colorize.py
# To download the caffemodel and the prototxt, see: https://github.com/richzhang/colorization/tree/master/colorization/models
# To download pts_in_hull.npy, see: https://github.com/richzhang/colorization/blob/master/colorization/resources/pts_in_hull.npy
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

W_in = 224
H_in = 224

prototxt = "colorization_deploy_v2.prototxt.txt"
caffemodel = "colorization_release_v2.caffemodel"
kernel = "pts_in_hull.npy"
input = "test1.jpg"

# Select desired model
net = cv.dnn.readNetFromCaffe(prototxt, caffemodel)

pts_in_hull = np.load(kernel) # load cluster centers

# populate cluster centers as 1x1 convolution kernel
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull.astype(np.float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full([1, 313], 2.606, np.float32)]

if input:
    cap = cv.VideoCapture(input)
else:
    cap = cv.VideoCapture(0)

width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

imshowSize = (width, height)

hasFrame, frame = cap.read()

img_rgb = (frame[:,:,[2, 1, 0]] * 1.0 / 255).astype(np.float32)

img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
img_l = img_lab[:,:,0] # pull out L channel
(H_orig,W_orig) = img_rgb.shape[:2] # original image size

# resize image to network input size
img_rs = cv.resize(img_rgb, (W_in, H_in)) # resize image to network input size
img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
img_l_rs = img_lab_rs[:,:,0]
img_l_rs -= 50 # subtract 50 for mean-centering

net.setInput(cv.dnn.blobFromImage(img_l_rs))
ab_dec = net.forward()[0,:,:,:].transpose((1,2,0)) # this is our result

(H_out,W_out) = ab_dec.shape[:2]
ab_dec_us = cv.resize(ab_dec, (W_orig, H_orig))
img_lab_out = np.concatenate((img_l[:,:,np.newaxis],ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2RGB), 0, 1)

frame = cv.resize(frame, imshowSize)

plt.rcParams['figure.figsize'] = [13, 13]
plt.subplot(2, 2, 1)
plt.imshow(frame)
plt.axis("off")
plt.title("Origin", fontsize=18)

plt.subplot(2, 2, 2)
gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
plt.imshow(gray, cmap='gray')
plt.axis("off")
plt.title("Gray scale", fontsize=18)

plt.subplot(2, 2, 3)
plt.imshow(img_rgb_out)
plt.axis("off")
plt.title("Colorization", fontsize=18)
plt.imsave("colorization.jpg", img_rgb_out)

img = cv.imread("colorization.jpg")
color = ('b', 'r', 'g')
plt.subplot(2, 2, 4)
for i, c in enumerate(color):
    hist_c = cv.calcHist([img], [i], None, [256], [0, 256])
    plt.plot(hist_c, color=c)
    plt.xlim([0, 256])
plt.grid()
plt.title("Histogram color", fontsize=18)

plt.show()

