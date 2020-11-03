

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# In Python, load 'frames/000000.jpg' using cv2.imread() and show it using cv2.imshow().
img = cv.imread('../cubemaps/cube_map.jpg')
print(img.shape)
#(1536, 2048, 3)
crop_img = img[512:512+512, 0:512]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image1.png', crop_img)

crop_img = img[512:512+512, 512:512+512]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image2.png', crop_img)


crop_img = img[512:512+512, 512*2:512*3]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image3.png', crop_img)

crop_img = img[512:512+512, 512*3:512*4]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image4.png', crop_img)

crop_img = img[0:512, 512:512*2]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image5.png', crop_img)


crop_img = img[512*2:512*3, 512:512*2]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/image6.png', crop_img)


crop_img = img[512:512+512, 0:512*4]
cv.imshow('image1', crop_img)
cv.waitKey(0)
cv.destroyAllWindows()

cv.imwrite('../cubemaps/deconstructed/panorama.png', crop_img)