import cv2

img = cv2.imread('004.png',1)
b, g, r = cv2.split(img)
a= img.shape
# cv2.imshow('image', img)
# cv2.waitKey(2)
s =b-g
print(a)
print(b)
print(s.int())
