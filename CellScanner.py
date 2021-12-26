import cv2
import imutils
import numpy as np

#makes the character into a square by padding the contour
def make_square(char):          
    s = max(char.shape)
    f = np.zeros((s,s),np.uint8)
    ax,ay = (s - char.shape[1])//2,(s - char.shape[0])//2
    f[ay:char.shape[0]+ay, ax:ax+char.shape[1]] = char
    image = cv2.copyMakeBorder(f,3, 3, 3, 3, cv2.BORDER_CONSTANT)
    return image


#Reading and resizing the cell
cell = cv2.imread('GG.jpeg')
cell = imutils.resize(cell, width=500)

#Thresholding:
gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
T, cell_t = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

#Finding contours (only need the external contour):
cnts, _ = cv2.findContours(cell_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
characters = []
for i, c in enumerate(cnts):
    x,y,w,h = cv2.boundingRect(c)
    character = cell_t[y:y+h, x:x+w]
    character = cv2.resize(make_square(character), (28, 28))
    characters.append(character)
    cv2.rectangle(cell_t, (x, y), (x + w, y + h), (100,0,0), 2)

#Writing the found contour into a folder
characters.reverse()
for i, char in enumerate(characters):
    cv2.imwrite(f'ScannedCharacters/Char{i}.jpg', char ) 

#Displaying them
cell = cv2.bitwise_and(cell, cell, mask=cell_t)
#cv2.imshow('Thresholded Image', cell_t )
#cv2.waitKey()
