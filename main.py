import imutils
import cv2
from stitcher import Stitcher 



imageA = cv2.imread('foto1A.jpg')
imageB = cv2.imread('foto1B.jpg')
#resizing images
imageA = imutils.resize(imageA, width=600)
imageB = imutils.resize(imageB, width=600)

# stich
stitcher = Stitcher()
result= stitcher.stitch([imageA, imageB])

# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Result", result)
cv2.waitKey(0)