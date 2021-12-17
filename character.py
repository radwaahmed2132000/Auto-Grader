import cv2
import imutils
import numpy as np

def make_square(char):          #makes the character into a square by padding
    s = max(char.shape)
    f = np.zeros((s,s),np.uint8)
    ax,ay = (s - char.shape[1])//2,(s - char.shape[0])//2
    f[ay:char.shape[0]+ay,ax:ax+char.shape[1]] = char
    image = cv2.copyMakeBorder(f,14, 14, 14, 14, cv2.BORDER_CONSTANT)
    return image


img = cv2.imread('1.png')
img = imutils.resize(img, width=500)

#Thresholding:
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
T, img_t = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
# 0 and +cv2.THRESH_BINARY_INV are meaningless.
# Otsu (Historgram Shape Analysis) Decides the threshold automatically.
# Probably won't need adaptive thresholding.

#Contours (only need the external contour):
cnts, _ = cv2.findContours(img_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
characters = []
for i, c in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c)
    character = img_t[y:y+h, x:x+w]
    character = cv2.resize(make_square(character), (28, 28))
    characters.append(character)
    cv2.imwrite(f'characters/Char{i}.jpg', character ) 
    cv2.rectangle(img_t, (x, y), (x + w, y + h), (100,0,0), 2)

img = cv2.bitwise_and(img, img, mask=img_t)
#cv2.imshow('Thresholded Image', img_t )
#cv2.waitKey()

