import cv2
import imutils
import numpy as np
from PIL import Image
import pytesseract
from Hexlassifier import predict_hex
from PrintClassifier import predict_print

#makes the character into a square by padding the contour
def make_square(char, p):          
    s = max(char.shape)
    f = np.zeros((s,s),np.uint8)
    ax,ay = (s - char.shape[1])//2,(s - char.shape[0])//2
    f[ay:char.shape[0]+ay, ax:ax+char.shape[1]] = char
    image = cv2.copyMakeBorder(f,p, p, p, p, cv2.BORDER_CONSTANT)
    return image

def cell_processing(cell):
    #Thresholding:
    gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    T, cell_t = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)

    #Finding contours (only need the external contour):
    cnts, _ = cv2.findContours(cell_t, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    characters = []
    for i, c in enumerate(cnts):
        x,y,w,h = cv2.boundingRect(c)
        character = cell_t[y:y+h, x:x+w]
        character = cv2.resize(make_square(character, 6), (28, 28))
        characters.append((character, x))
        cv2.rectangle(cell_t, (x, y), (x + w, y + h), (100,0,0), 2)

    characters.sort(key=lambda by_x: by_x[1])
    characters = [c[0] for c in characters]
    #Writing the found contour into a folder
    for i, char in enumerate(characters):
        cv2.imwrite(f'ScannedCharacters/Char{i}.jpg', char )
    return cell, cell_t, characters


#Reading and resizing the cell
cell = cv2.imread('./Samples/Digits.png')
cell = imutils.resize(cell, width=500)
cell, cell_t, characters = cell_processing(cell)

#Displaying them
cell = cv2.bitwise_and(cell, cell, mask=cell_t)
cv2.imshow('Thresholded Image', cell_t )
cv2.waitKey()

handwritten = False

if(handwritten):
    cell_content = predict_hex(characters, saved=True)
else:
    cell_content = predict_print(characters, saved=True)

cell_content = pytesseract.image_to_string(Image.open('./Samples/Digits.png'))
print(cell_content)